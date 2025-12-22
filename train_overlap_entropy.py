#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import shutil
import numpy as np
import json
import struct
import types
import subprocess
import torch
import torchvision
import wandb
import time
from datetime import datetime
from os import makedirs
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint, random
from utils.loss_utils import l1_loss, ssim
import sys
from gaussian_renderer import network_gui
from scene import Scene
from utils.general_utils import get_expon_lr_func, safe_state, parse_cfg, visualize_depth
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, save_rgba
from argparse import ArgumentParser, Namespace
import yaml
import torch.nn.functional as F
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# GPU Setup
try:
    cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
except:
    pass
os.system('echo $CUDA_VISIBLE_DEVICES')

# LPIPS Setup
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

# ================= [CORE UTILS: FROM DEBUG_OVERLAP_V3] =================

class FrustumCuller:
    """
    [Fixed Logic from debug_overlapV3.py]
    负责计算 3D 锚点与相机视锥体的几何关系。
    """
    @staticmethod
    def filter_anchors_by_frustum(anchors, view_proj_matrix, margin=0.5): 
        # 1. 齐次坐标变换
        p_hom = torch.cat([anchors, torch.ones_like(anchors[:, :1])], dim=1) 
        # [Correct Order]: p_hom @ Matrix for row-major systems in Torch/3DGS
        p_clip = p_hom @ view_proj_matrix 
        
        # 2. W 判断 (Z深度 > epsilon)
        valid_w = p_clip[:, 3] > 0.001
        
        # 3. 透视除法
        denom = p_clip[:, 3] + 1e-6
        p_ndc_x = p_clip[:, 0] / denom
        p_ndc_y = p_clip[:, 1] / denom
        
        # 4. 范围判断
        limit = 1.0 + margin 
        mask_x = (p_ndc_x > -limit) & (p_ndc_x < limit)
        mask_y = (p_ndc_y > -limit) & (p_ndc_y < limit)
        
        return valid_w & mask_x & mask_y

def project_points_to_2d_mask(anchors, mask_indices, view_cam, height, width, dilation=15):
    """
    [Safe Mask Generator]
    将 3D 点投影到 2D 屏幕，生成二值 Mask。
    使用较大的 dilation 来将稀疏的 Level 0 锚点连成片。
    """
    selected_anchors = anchors[mask_indices]
    if selected_anchors.shape[0] == 0:
        return torch.zeros((1, height, width), device="cuda")

    # Projection
    p_hom = torch.cat([selected_anchors, torch.ones_like(selected_anchors[:, :1])], dim=1)
    p_clip = p_hom @ view_cam.full_proj_transform
    
    valid_z = p_clip[:, 3] > 0.001
    p_ndc = p_clip[valid_z, :3] / p_clip[valid_z, 3:4]
    
    # Clip to screen
    valid_screen = (p_ndc[:, 0] > -1.2) & (p_ndc[:, 0] < 1.2) & \
                   (p_ndc[:, 1] > -1.2) & (p_ndc[:, 1] < 1.2)
    p_ndc = p_ndc[valid_screen]
    
    if p_ndc.shape[0] == 0:
        return torch.zeros((1, height, width), device="cuda")

    # Map to pixels
    u = ((p_ndc[:, 0] + 1) * width - 1) * 0.5
    v = ((p_ndc[:, 1] + 1) * height - 1) * 0.5
    
    u = u.long().clamp(0, width - 1)
    v = v.long().clamp(0, height - 1)
    
    mask = torch.zeros((height, width), device="cuda", dtype=torch.float32)
    mask[v, u] = 1.0
    
    # Dilation to fill gaps between Level 0 anchors
    if dilation > 0:
        mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        mask = F.max_pool2d(mask, kernel_size=2*dilation+1, stride=1, padding=dilation)
        mask = mask.squeeze()
        
    return mask # [H, W]

class Level0OverlapManager:
    """
    专门用于管理 Level 0 的重叠锚点。
    """
    def __init__(self):
        self.overlap_indices = None
        self.is_initialized = False

    def precompute(self, scene, gaussians, logger):
        """
        预计算：找出同时在街景和航拍视锥内的 Level 0 锚点索引。
        """
        logger.info("[L0 Manager] Pre-computing Overlap Level 0 Indices...")
        
        # 1. 准备数据
        all_anchors = gaussians.get_anchor.detach()
        all_levels = gaussians.get_level.detach().squeeze()
        
        # 2. 筛选 Level 0
        is_level0 = (all_levels == 0)
        level0_indices_global = torch.nonzero(is_level0).squeeze()
        level0_anchors = all_anchors[is_level0]
        
        n_l0 = level0_anchors.shape[0]
        logger.info(f"[L0 Manager] Total Level 0 Anchors: {n_l0}")

        # 3. 街景覆盖检测
        street_cams = [c for c in scene.getTrainCameras() if c.image_type == 'street']
        if len(street_cams) == 0:
            logger.warning("No street cameras found! Skipping overlap logic.")
            return

        seen_by_street = torch.zeros(n_l0, dtype=torch.bool, device="cuda")
        # 抽样检测以加速 (每5张抽1张)
        for cam in tqdm(street_cams[::5], desc="Checking Street Visibility"):
            mask = FrustumCuller.filter_anchors_by_frustum(level0_anchors, cam.full_proj_transform, margin=0.2)
            seen_by_street |= mask

        # 4. 航拍覆盖检测
        aerial_cams = [c for c in scene.getTrainCameras() if c.image_type == 'aerial']
        if len(aerial_cams) == 0:
            logger.warning("No aerial cameras found! Skipping overlap logic.")
            return

        seen_by_aerial = torch.zeros(n_l0, dtype=torch.bool, device="cuda")
        for cam in tqdm(aerial_cams[::5], desc="Checking Aerial Visibility"):
            mask = FrustumCuller.filter_anchors_by_frustum(level0_anchors, cam.full_proj_transform, margin=0.2)
            seen_by_aerial |= mask
            
        # 5. 取交集
        overlap_mask_local = seen_by_street & seen_by_aerial
        
        # 6. 映射回全局索引
        self.overlap_indices = level0_indices_global[overlap_mask_local]
        self.is_initialized = True
        
        count = self.overlap_indices.shape[0]
        logger.info(f"[L0 Manager] Found {count} Overlap Level 0 Anchors ({count/n_l0*100:.2f}% of L0).")

# ================= [TRAINING LOGIC] =================

def saveRuntimeCode(dst: str) -> None:
    # ... (Keep existing code saving logic) ...
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    if os.path.exists(os.path.join(ROOT, '.gitignore')):
        with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
            for line in gitIgnoreFile:
                if not line.startswith('#'):
                    if line.endswith('\n'): line = line[:-1]
                    if line.endswith('/'): line = line[:-1]
                    ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent
    try:
        shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
        print('Backup Finished!')
    except:
        pass

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    modules = __import__('scene')
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
    
    scene = Scene(dataset, gaussians, shuffle=False, logger=logger, weed_ratio=pipe.weed_ratio)

    # --- 修正相机类型 ---
    logger.info("Classifying camera types...")
    aerial_cnt, street_cnt = 0, 0
    for cam in scene.getTrainCameras():
        if "street" in cam.image_name.lower():
            cam.image_type = "street"
            street_cnt += 1
        else:
            cam.image_type = "aerial"
            aerial_cnt += 1
    scene.add_aerial = (aerial_cnt > 0)
    scene.add_street = (street_cnt > 0)
    logger.info(f"Cameras: Aerial={aerial_cnt}, Street={street_cnt}")

    # --- [NEW] 初始化 Level 0 重叠管理器 ---
    l0_manager = Level0OverlapManager()
    if scene.add_aerial and scene.add_street:
        # 在这里预计算索引
        l0_manager.precompute(scene, gaussians, logger)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    # 随机采样器
    viewpoint_stack = None
    
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    densify_cnt = 0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", leave=True, dynamic_ncols=True, mininterval=0.5)
    first_iter += 1
    modules = __import__('gaussian_renderer')
    
    # --- 熵损失参数 ---
    ENTROPY_START_ITER = 3000   # 给模型一点预热时间
    ENTROPY_WEIGHT = 0.05       # 权重：建议 0.01 - 0.1
    MASK_DILATION = 20          # 膨胀半径，因为Level 0很稀疏，需要连成片
    
    for iteration in range(first_iter, opt.iterations + 1):        
        # ... (GUI Communication Code Omitted for Brevity, keep it if needed) ...
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.add_prefilter, keep_alive = network_gui.receive()
                if custom_cam != None:
                    net_image = getattr(modules, 'render')(custom_cam, gaussians, pipe, scene.background)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        # ... 

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        
        # 1. 随机选择一个相机 (标准训练流程)
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 2. 渲染
        render_pkg = getattr(modules, 'render')(viewpoint_cam, gaussians, pipe, scene.background)
        image, scaling, alpha = render_pkg["render"], render_pkg["scaling"], render_pkg["render_alphas"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        alpha_mask = viewpoint_cam.alpha_mask.cuda()
        
        # 应用 Alpha Mask (如果有)
        image = image * alpha_mask
        gt_image = gt_image * alpha_mask

        # 3. 计算基础 RGB Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # ================= [OVERLAP ENTROPY OPTIMIZATION] =================
        # 目标：在重叠区域(Level 0可见区)，强迫渲染结果变得不透明(实体化)，消除半透明堆叠。
        # 只有在管理器初始化成功，且训练进入中期时才开启
        if l0_manager.is_initialized and (iteration > ENTROPY_START_ITER):
            
            # A. 动态生成当前视角的“安全重叠掩膜”
            # 将预计算的 Level 0 3D 索引投影到当前 2D 屏幕
            overlap_mask_2d = project_points_to_2d_mask(
                gaussians.get_anchor.detach(),  # 传入所有锚点(函数内会按索引取)
                l0_manager.overlap_indices,     # 只投影这些重叠点
                viewpoint_cam, 
                image.shape[1], image.shape[2], 
                dilation=MASK_DILATION
            )
            
            # B. 计算不透明度熵
            # alpha: [1, H, W]
            # 限制范围防止 log(0)
            a_clip = torch.clamp(alpha, 1e-6, 1-1e-6)
            # 熵公式: -p*log(p) - (1-p)*log(1-p). 当 p=0 或 p=1 时熵为 0.
            entropy_map = - (a_clip * torch.log(a_clip) + (1-a_clip) * torch.log(1-a_clip))
            
            # C. 施加约束 (只在 Mask 范围内)
            # 使用 sum() / (sum() + epsilon) 避免除零
            mask_sum = overlap_mask_2d.sum() + 1e-6
            loss_entropy = (entropy_map * overlap_mask_2d).sum() / mask_sum
            
            loss += ENTROPY_WEIGHT * loss_entropy
        # ==================================================================

        # 其他正则化 Loss (保持原样)
        if opt.lambda_dreg > 0:
            if scaling.shape[0] > 0:
                scaling_reg = scaling.prod(dim=1).mean()
            else:
                scaling_reg = torch.tensor(0.0, device="cuda")
            loss += opt.lambda_dreg * scaling_reg
        
        if opt.lambda_sky_opa > 0:
             # 注意：Entropy Loss 可能会和 Sky Opacity Loss 冲突，
             # 但由于我们用了 Mask 限制 Entropy 只在重叠区(地面)，所以理应不冲突。
            o = alpha.clamp(1e-6, 1-1e-6)
            sky = alpha_mask.float()
            loss_sky_opa = (-(1-sky) * torch.log(1 - o)).mean()
            loss += opt.lambda_sky_opa * loss_sky_opa

        # 深度 Loss
        cur_Ll1depth = 0
        if iteration > opt.start_depth and depth_l1_weight(iteration) > 0 and viewpoint_cam.invdepthmap is not None:
            render_depth = render_pkg["render_depth"]
            invDepth = torch.where(render_depth > 0.0, 1.0 / render_depth, torch.zeros_like(render_depth))            
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).sum() / (depth_mask.sum() + 1e-6)
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            cur_Ll1depth = Ll1depth.item()
        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * cur_Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}","Depth": f"{ema_Ll1depth_for_log:.{5}f}","GS":f"{len(gaussians.get_anchor)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, getattr(modules, 'render'), (pipe, scene.background), wandb, logger)
            
            if (iteration in saving_iterations):
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration % pipe.vis_step == 0 or iteration == 1:
                vis_path = os.path.join(scene.model_path, "vis")
                os.makedirs(vis_path, exist_ok=True)
                # 如果开启了 Entropy 优化，顺便保存 Mask 看看有没有切准
                if l0_manager.is_initialized and iteration > ENTROPY_START_ITER:
                    # 生成一张 debug 图：左边是渲染，右边是 mask
                    mask_vis = overlap_mask_2d.unsqueeze(0).repeat(3,1,1)
                    combined = torch.cat([image, mask_vis], dim=2) # 左右拼接
                    torchvision.utils.save_image(combined, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}_entropy.png"))
                else:
                    torchvision.utils.save_image(image, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))
            
            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(opt, render_pkg, image.shape[2], image.shape[1])
                densify_cnt += 1 

                if opt.densification and iteration > opt.update_from and densify_cnt > 0 and densify_cnt % opt.update_interval == 0:
                    gaussians.run_densify(opt, iteration)
            
            elif iteration == opt.update_until:
                gaussians.clean()
                    
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration >= opt.iterations - pipe.no_prefilter_step:
                pipe.add_prefilter = False

            if (iteration in checkpoint_iterations):
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({'train_total_loss':loss, })
    
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in range(0, len(scene.getTrainCameras()), 100)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test_aerial = 0.0
                psnr_test_aerial = 0.0
                aerial_cnt = 0
                l1_test_street = 0.0
                psnr_test_street = 0.0
                street_cnt = 0
                
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    alpha_mask = viewpoint.alpha_mask.cuda()
                    image = image * alpha_mask
                    gt_image = gt_image * alpha_mask
                    
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                    if viewpoint.image_type == "aerial":
                        l1_test_aerial += l1_loss(image, gt_image).mean().double()
                        psnr_test_aerial += psnr(image, gt_image).mean().double()
                        aerial_cnt += 1 
                    else:
                        l1_test_street += l1_loss(image, gt_image).mean().double()
                        psnr_test_street += psnr(image, gt_image).mean().double()
                        street_cnt += 1 
                
                if scene.add_aerial and aerial_cnt > 0:
                    l1_test_aerial /= aerial_cnt
                    psnr_test_aerial /= aerial_cnt    
                    tqdm.write("[ITER {}] Evaluating {} Aerial: L1 {} PSNR {}".format(iteration, config['name'], l1_test_aerial, psnr_test_aerial))
                if scene.add_street and street_cnt > 0:       
                    l1_test_street /= street_cnt
                    psnr_test_street /= street_cnt       
                    tqdm.write("[ITER {}] Evaluating {} Street: L1 {} PSNR {}".format(iteration, config['name'], l1_test_street, psnr_test_street))
                
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', len(scene.gaussians.get_anchor), iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipe, background, add_aerial, add_street):
    if add_aerial:
        aerial_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "renders")
        aerial_error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "errors")
        aerial_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "gt")
        makedirs(aerial_render_path, exist_ok=True)
        makedirs(aerial_error_path, exist_ok=True)
        makedirs(aerial_gts_path, exist_ok=True)
    if add_street:
        street_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "renders")
        street_error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "errors")
        street_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "gt")
        makedirs(street_render_path, exist_ok=True)
        makedirs(street_error_path, exist_ok=True)
        makedirs(street_gts_path, exist_ok=True)

    modules = __import__('gaussian_renderer')

    street_t_list = []
    street_visible_count_list = []
    street_per_view_dict = {}
    street_views = [view for view in views if view.image_type=="street"]
    for idx, view in enumerate(tqdm(street_views, desc="Street rendering progress")):
        torch.cuda.synchronize();t_start = time.time()
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        torch.cuda.synchronize();t_end = time.time()
        street_t_list.append(t_end - t_start)
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        gt = view.original_image.cuda()
        alpha_mask = view.alpha_mask.cuda()
        rendering = torch.cat([rendering, alpha_mask], dim=0)
        gt = torch.cat([gt, alpha_mask], dim=0)
        if gt.device != rendering.device: rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()
        save_rgba(rendering, os.path.join(street_render_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(errormap, os.path.join(street_error_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(gt, os.path.join(street_gts_path, '{0:05d}'.format(idx) + ".png"))
        street_visible_count_list.append(visible_count)
        street_per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    if len(street_views) > 0:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "street", "per_view_count.json"), 'w') as fp:
            json.dump(street_per_view_dict, fp, indent=True)
    
    aerial_t_list = []
    aerial_visible_count_list = []
    aerial_per_view_dict = {}
    aerial_views = [view for view in views if view.image_type=="aerial"]
    for idx, view in enumerate(tqdm(aerial_views, desc="Aerial rendering progress")):
        torch.cuda.synchronize();t_start = time.time()
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        torch.cuda.synchronize();t_end = time.time()
        aerial_t_list.append(t_end - t_start)
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        gt = view.original_image.cuda()
        alpha_mask = view.alpha_mask.cuda()
        rendering = torch.cat([rendering, alpha_mask], dim=0)
        gt = torch.cat([gt, alpha_mask], dim=0)
        if gt.device != rendering.device: rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()
        save_rgba(rendering, os.path.join(aerial_render_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(errormap, os.path.join(aerial_error_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(gt, os.path.join(aerial_gts_path, '{0:05d}'.format(idx) + ".png"))
        aerial_visible_count_list.append(visible_count)
        aerial_per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    if len(aerial_views) > 0:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "per_view_count.json"), 'w') as fp:
            json.dump(aerial_per_view_dict, fp, indent=True)

    return aerial_visible_count, street_visible_count

def render_sets(dataset, opt, pipe, iteration, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        if pipe.no_prefilter_step > 0:
            pipe.add_prefilter = False
        else:
            pipe.add_prefilter = True
        modules = __import__('scene')
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, logger=logger)
        gaussians.eval()

        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            aerial_visible_count, street_visible_count = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, scene.add_aerial, scene.add_street)

        if not skip_test:
            aerial_visible_count, street_visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background, scene.add_aerial, scene.add_street)

    return aerial_visible_count, street_visible_count

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        render_image = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        render_mask = tf.to_tensor(render).unsqueeze(0)[:, 3:4, :, :].cuda()
        render_image = render_image * render_mask
        gt_image = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        gt_mask = tf.to_tensor(gt).unsqueeze(0)[:, 3:4, :, :].cuda()
        gt_image = gt_image * gt_mask
        renders.append(render_image)
        gts.append(gt_image)
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, eval_name, aerial_visible_count=None, street_visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    full_dict = {}
    per_view_dict = {}
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}

        base_method_dir = test_dir / method
        
        # Eval Aerial
        method_dir = base_method_dir / "aerial" 
        if os.path.exists(method_dir):
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

            logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
            logger.info("  AERIAL_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
            
            full_dict[scene_dir][method].update({
                "AERIAL_PSNR": torch.tensor(psnrs).mean().item(),
                "AERIAL_SSIM": torch.tensor(ssims).mean().item(),
                "AERIAL_LPIPS": torch.tensor(lpipss).mean().item(),
                })
            
            per_view_dict[scene_dir][method].update({
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                })

        # Eval Street
        method_dir = base_method_dir / "street" 
        if os.path.exists(method_dir):
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

            logger.info("  STREET_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
            
            full_dict[scene_dir][method].update({
                "STREET_PSNR": torch.tensor(psnrs).mean().item(),
                "STREET_SSIM": torch.tensor(ssims).mean().item(),
                "STREET_LPIPS": torch.tensor(lpipss).mean().item(),
                })

            per_view_dict[scene_dir][method].update({
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                })

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)
    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)
    return logger

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path') 
    parser.add_argument('--ip', type=str, default="127.0.0.1") 
    parser.add_argument('--port', type=int, default=6009) 
    parser.add_argument('--debug_from', type=int, default=-1) 
    parser.add_argument('--detect_anomaly', action='store_true', default=False) 
    parser.add_argument('--use_wandb', action='store_true', default=False) 
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1]) 
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1]) 
    parser.add_argument("--quiet", action="store_true") 
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[]) 
    parser.add_argument("--start_checkpoint", type=str, default = None) 
    parser.add_argument("--gpu", type=str, default = '-1') 
    args = parser.parse_args(sys.argv[1:])
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        args.save_iterations.append(op.iterations)

    lp.model_path = os.path.join("outputs", lp.dataset_name, lp.scene_name)
    os.makedirs(lp.model_path, exist_ok=True)
    shutil.copy(args.config, os.path.join(lp.model_path, "config.yaml"))

    logger = get_logger(lp.model_path)

    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != op.iterations:
        args.test_iterations.append(op.iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
    if len(args.save_iterations) == 0 or args.save_iterations[-1] != op.iterations:
        args.save_iterations.append(op.iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        logger.info(f'using GPU {args.gpu}')
    
    try:
        saveRuntimeCode(os.path.join(lp.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
    
    exp_name = lp.scene_name if lp.dataset_name=="" else lp.dataset_name+"_"+lp.scene_name
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            project=f"Horizon-GS",
            name=exp_name,
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + lp.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp, op, pp, exp_name, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, wandb, logger)

    logger.info("\nTraining complete.")
    logger.info(f'\nStarting Rendering~')
    if lp.eval:
        aerial_visible_count, street_visible_count = render_sets(lp, op, pp, -1, skip_train=True, skip_test=False, wandb=wandb, logger=logger)
    else:
        aerial_visible_count, street_visible_count = render_sets(lp, op, pp, -1, skip_train=False, skip_test=True, wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    logger.info("\n Starting evaluation...")
    eval_name = 'test' if lp.eval else 'train'
    evaluate(lp.model_path, eval_name, aerial_visible_count=16, street_visible_count=41, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")