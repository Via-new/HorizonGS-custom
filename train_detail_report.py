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

import subprocess
# 构建命令查询 nvidia-smi 的显存信息
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# 执行系统命令并获取输出结果，按行分割
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# 解析输出，找到显存使用量最小的GPU索引，并设置环境变量 CUDA_VISIBLE_DEVICES
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# 打印当前使用的 GPU 编号
os.system('echo $CUDA_VISIBLE_DEVICES')

import torch
import torchvision
import json
import wandb
import time
from datetime import datetime
from os import makedirs
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
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

# 初始化 LPIPS 感知损失模型（使用VGG网络），并移动到 CUDA
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    # 尝试导入 Tensorboard 用于可视化训练过程
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    """
    备份当前运行时的代码到指定目录，用于复现实验。
    排除 .git 等无关文件。
    """
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    assert os.path.exists(os.path.join(ROOT, '.gitignore'))
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

# ================= [DIAGNOSIS HELPER FUNCTIONS START] =================
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary_debug(path_to_model_file):
    images = {}
    try:
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_reg_images):
                binary_image_properties = read_next_bytes(fid, 64, "i4d3di")
                image_id = binary_image_properties[0]
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_name = ""
                current_char = read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(fid, 1, "c")[0]
                num_points2D = read_next_bytes(fid, 8, "Q")[0]
                # Skip points data
                fid.seek(24 * num_points2D, 1)
                images[image_id] = {"id": image_id, "name": image_name}
    except Exception as e:
        print(f"[DEBUG READ ERROR] {e}")
    return images
# ================= [DIAGNOSIS HELPER FUNCTIONS END] =================

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, wandb=None, logger=None, ply_path=None):
    """
    主训练函数。
    """
    first_iter = 0
    # 准备输出目录和 Tensorboard writer
    tb_writer = prepare_output_and_logger(dataset)

    # 动态导入 scene 模块并根据配置初始化 Gaussian 对象 (例如 GaussianLoDModel)
    modules = __import__('scene')
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
    
    # ================= [DIAGNOSIS LOGIC START] =================
    # 为了避免刷屏，这里的诊断信息只在初始化时运行一次
    logger.info("\n" + "="*20 + " 深度参数与COLMAP一致性检查 " + "="*20)
    json_path = os.path.join(dataset.source_path, "sparse", "0", "depth_params.json")
    
    if os.path.exists(json_path):
        logger.info(f"[JSON] 发现文件: {json_path}")
        # 简单检查一下文件是否能读取
        try:
            with open(json_path, 'r') as f:
                json.load(f)
            logger.info(f"[JSON] 文件格式正确。")
        except Exception as e:
            logger.error(f"[JSON] 文件损坏: {e}")
    else:
        logger.error(f"[JSON] 警告: 未找到 {json_path}。如果开启了 add_depth，这将导致报错。")
    logger.info("="*60 + "\n")
    # ================= [DIAGNOSIS LOGIC END] =================

    # 初始化场景 (加载相机、点云等数据)
    scene = Scene(dataset, gaussians, shuffle=False, logger=logger, weed_ratio=pipe.weed_ratio)

    # ================= [修复开始] 手动修正相机类型 =================
    logger.info("Fixing camera types based on filenames...")
    aerial_count = 0
    street_count = 0
    
    # [NEW] 统计 Depth 和 Mask 的加载情况
    depth_loaded_count = 0
    mask_loaded_count = 0
    total_train_cams = len(scene.getTrainCameras())

    for cam in scene.getTrainCameras():
        # 获取小写文件名
        img_name = cam.image_name.lower()
        
        if "street" in img_name:
            cam.image_type = "street"
            street_count += 1
        elif "aerial" in img_name:
            cam.image_type = "aerial"
            aerial_count += 1
        else:
            cam.image_type = "aerial"
            aerial_count += 1
        
        # [NEW] 检查 Depth (invdepthmap 存在即表示读取成功)
        if cam.invdepthmap is not None:
            depth_loaded_count += 1
            
        # [NEW] 检查 Mask (alpha_mask 存在即表示读取成功)
        if cam.alpha_mask is not None:
            mask_loaded_count += 1
            
    logger.info(f"Manual Classification Result: Aerial={aerial_count}, Street={street_count}")
    
    # [NEW] 打印 Depth 和 Mask 的统计信息 (只打印一次)
    logger.info("-" * 50)
    logger.info(f"数据加载完整性检查 (训练集):")
    logger.info(f"  总相机数 (Total Cameras) : {total_train_cams}")
    logger.info(f"  深度图 (Depth Maps)      : {depth_loaded_count} / {total_train_cams} ({(depth_loaded_count/total_train_cams)*100:.1f}%)")
    logger.info(f"  掩码图 (Masks)           : {mask_loaded_count} / {total_train_cams} ({(mask_loaded_count/total_train_cams)*100:.1f}%)")
    
    if dataset.add_depth and depth_loaded_count < total_train_cams:
        logger.warning(f"警告: 已开启 add_depth，但有 {total_train_cams - depth_loaded_count} 张图片未加载到深度图！")
    elif dataset.add_depth:
        logger.info("成功: 所有相机的深度图已成功加载。")

    if dataset.add_mask and mask_loaded_count < total_train_cams:
        logger.warning(f"警告: 已开启 add_mask，但有 {total_train_cams - mask_loaded_count} 张图片未加载到掩码！")
    elif dataset.add_mask:
        logger.info("成功: 所有相机的掩码已成功加载。")
    logger.info("-" * 50)

    # 安全检查
    if street_count == 0 and pipe.camera_balance:
        logger.warning("WARNING: No street cameras found even after manual fix! Disabling camera_balance.")
        pipe.camera_balance = False
    
    scene.add_aerial = (aerial_count > 0)
    scene.add_street = (street_count > 0)
    # ================= [修复结束] =================

    # 设置 Gaussian 模型的训练参数 (优化器等)
    gaussians.training_setup(opt)
    
    # 如果有 checkpoint，加载模型参数并恢复训练状态
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置 CUDA 事件用于计时
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    # 获取深度 L1 损失权重的指数衰减函数
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    if pipe.camera_balance:
        aerial_viewpoint_stack = None
        street_viewpoint_stack = None
    else:
        viewpoint_stack = None
        
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    densify_cnt = 0
    
    # [NEW] 优化进度条设置：leave=True(保留进度条), dynamic_ncols=True(自适应宽度), mininterval=0.5(减少刷新频率)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", leave=True, dynamic_ncols=True, mininterval=0.5)
    first_iter += 1
    modules = __import__('gaussian_renderer')
    
    # --- 开始训练循环 ---
    for iteration in range(first_iter, opt.iterations + 1):        
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

        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)
        
        # --- 选择一个随机相机进行训练 ---
        if pipe.camera_balance:
            if not aerial_viewpoint_stack:
                aerial_viewpoint_stack = [cam for cam in scene.getTrainCameras().copy() if cam.image_type == "aerial"]
            if not street_viewpoint_stack:
                street_viewpoint_stack = [cam for cam in scene.getTrainCameras().copy() if cam.image_type == "street"]
            
            aerial_proportion, street_proportion = pipe.camera_proportion.split("-")
            r = float(aerial_proportion) / ( float(aerial_proportion) + float(street_proportion) )
            if np.random.rand() < r:
                viewpoint_cam = aerial_viewpoint_stack.pop(randint(0, len(aerial_viewpoint_stack)-1))
            else:
                viewpoint_cam = street_viewpoint_stack.pop(randint(0, len(street_viewpoint_stack)-1))
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # --- 前向传播：渲染 ---
        render_pkg = getattr(modules, 'render')(viewpoint_cam, gaussians, pipe, scene.background)
        image, scaling, alpha = render_pkg["render"], render_pkg["scaling"], render_pkg["render_alphas"]

        gt_image = viewpoint_cam.original_image.cuda()
        alpha_mask = viewpoint_cam.alpha_mask.cuda()
        
        # 应用掩码
        image = image * alpha_mask
        gt_image = gt_image * alpha_mask
            
        # --- 计算基础损失 ---
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
       
        # --- 计算正则化损失 ---
        if opt.lambda_dreg > 0:
            if scaling.shape[0] > 0:
                scaling_reg = scaling.prod(dim=1).mean()
            else:
                scaling_reg = torch.tensor(0.0, device="cuda")
            loss += opt.lambda_dreg * scaling_reg
        
        if opt.lambda_sky_opa > 0:
            o = alpha.clamp(1e-6, 1-1e-6)
            sky = alpha_mask.float()
            loss_sky_opa = (-(1-sky) * torch.log(1 - o)).mean()
            loss = loss + opt.lambda_sky_opa * loss_sky_opa

        if opt.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o*torch.log(o)).mean()
            loss = loss + opt.lambda_opacity_entropy * loss_opacity_entropy

        if opt.lambda_normal > 0 and iteration > opt.normal_start_iter:
            assert gaussians.render_mode=="RGB+ED" or gaussians.render_mode=="RGB+D"
            normals = render_pkg["render_normals"].squeeze(0).permute((2, 0, 1))
            normals_from_depth = render_pkg["render_normals_from_depth"] * render_pkg["render_alphas"].permute((1, 2, 0)).detach()
            if len(normals_from_depth.shape) == 4:
                normals_from_depth = normals_from_depth.squeeze(0)
            normals_from_depth = normals_from_depth.permute((2, 0, 1))
            normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
            loss += opt.lambda_normal * (normal_error * alpha_mask).mean()

        if opt.lambda_dist and iteration > opt.dist_start_iter:
            loss += opt.lambda_dist * (render_pkg["render_distort"].squeeze(3) * alpha_mask).mean()
        
        # 深度损失
        if iteration > opt.start_depth and depth_l1_weight(iteration) > 0 and viewpoint_cam.invdepthmap is not None:
            assert gaussians.render_mode=="RGB+ED" or gaussians.render_mode=="RGB+D"
            render_depth = render_pkg["render_depth"]
            invDepth = torch.where(render_depth > 0.0, 1.0 / render_depth, torch.zeros_like(render_depth))            
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
    
        # --- 反向传播 ---
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                psnr_log = psnr(image, gt_image).mean().double()
                anchor_prim = len(gaussians.get_anchor) 
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}","Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}","psnr":f"{psnr_log:.{3}f}","GS_num":f"{anchor_prim}","prefilter":f"{pipe.add_prefilter}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, getattr(modules, 'render'), (pipe, scene.background), wandb, logger)
            
            if (iteration in saving_iterations):
                # [NEW] 使用 tqdm.write 避免破坏进度条格式
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration % pipe.vis_step == 0 or iteration == 1:
                other_img = []
                resolution = (int(viewpoint_cam.image_width/5.0), int(viewpoint_cam.image_height/5.0))
                vis_img = F.interpolate(image.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                vis_gt_img = F.interpolate(gt_image.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                vis_alpha = F.interpolate(alpha.repeat(3, 1, 1).unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]

                if iteration > opt.start_depth and viewpoint_cam.invdepthmap is not None:
                    vis_depth = visualize_depth(invDepth) 
                    gt_depth = visualize_depth(mono_invdepth)
                    vis_depth = F.interpolate(vis_depth.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                    vis_gt_depth = F.interpolate(gt_depth.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                    other_img.append(vis_depth)
                    other_img.append(vis_gt_depth)
                
                grid = torchvision.utils.make_grid([
                    vis_img, 
                    vis_gt_img, 
                    vis_alpha,
                ] + other_img, nrow=3)

                vis_path = os.path.join(scene.model_path, "vis")
                os.makedirs(vis_path, exist_ok=True)
                torchvision.utils.save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))
            
            if iteration < opt.update_until and iteration > opt.start_stat:
                if  (viewpoint_cam.image_type == "aerial" and pipe.aerial_densify) \
                    or (viewpoint_cam.image_type == "street" and pipe.street_densify) :
                    gaussians.training_statis(opt, render_pkg, image.shape[2], image.shape[1])
                    densify_cnt += 1 

                if opt.densification and iteration > opt.update_from and densify_cnt > 0 and densify_cnt % opt.update_interval == 0:
                    if dataset.pretrained_checkpoint != "":
                        gaussians.roll_back()
                    gaussians.run_densify(opt, iteration)
            
            elif iteration == opt.update_until:
                if dataset.pretrained_checkpoint != "":
                    gaussians.roll_back()
                gaussians.clean()
                    
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration >= opt.iterations - pipe.no_prefilter_step:
                pipe.add_prefilter = False

            if (iteration in checkpoint_iterations):
                # [NEW] 使用 tqdm.write
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    """
    准备输出目录并初始化 Tensorboard。
    """
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
    """
    记录训练状态，并在测试集上进行评估。
    """
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # --- [修改开始] 评估与日志记录逻辑 ---
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        
        # ==========================================================
        # [NEW] 修复高斯数量统计逻辑
        # ==========================================================
        
        # 1. 获取锚点数量 (静态存储的数量)
        num_anchors = len(scene.gaussians.get_anchor)
        
        # 2. 获取实际高斯球数量 (动态生成的数量)
        # HorizonGS/ScaffoldGS 在渲染时会基于锚点生成 n_offsets 个高斯
        # 我们通过执行一次推理(render)来获取 render_pkg，查看其中 scaling 的维度
        try:
            # 临时取训练集的第一个相机进行一次“试渲染”
            temp_cam = scene.getTrainCameras()[0]
            with torch.no_grad():
                # renderArgs 是 (pipe, scene.background)
                temp_pkg = renderFunc(temp_cam, scene.gaussians, *renderArgs)
                # scaling 的形状通常是 [Num_Gaussians, 3]，这才是真正的生成数量
                num_gaussians = temp_pkg["scaling"].shape[0]
        except Exception as e:
            # 如果出错（例如内存不足），回退到读取 xyz 大小，并打印警告
            num_gaussians = scene.gaussians.get_xyz.shape[0]
            if logger: logger.warning(f"无法计算动态高斯数量，回退到静态数量。错误: {e}")

        # 3. 计算倍率 (每个锚点生成了多少个高斯)
        ratio = num_gaussians / num_anchors if num_anchors > 0 else 0

        if logger:
            logger.info("=" * 40)
            logger.info(f"[ITER {iteration}] 统计信息 (Statistics):")
            logger.info(f"  > 锚点数量 (Anchor Num)   : {num_anchors}")
            logger.info(f"  > 高斯数量 (Gaussian Num) : {num_gaussians} (倍率: {ratio:.1f}x)")
            logger.info("=" * 40)
        # ==========================================================

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
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    alpha_mask = viewpoint.alpha_mask.cuda()
                    image = image * alpha_mask
                    gt_image = gt_image * alpha_mask
                    
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        if wandb:
                            gt_image_list.append(gt_image[None])
                
                    if viewpoint.image_type == "aerial":
                        l1_test_aerial += l1_loss(image, gt_image).mean().double()
                        psnr_test_aerial += psnr(image, gt_image).mean().double()
                        aerial_cnt += 1 
                    else:
                        l1_test_street += l1_loss(image, gt_image).mean().double()
                        psnr_test_street += psnr(image, gt_image).mean().double()
                        street_cnt += 1 
                
                # 使用 logger 记录评估结果到 output.log
                if scene.add_aerial and aerial_cnt > 0:
                    l1_test_aerial /= aerial_cnt
                    psnr_test_aerial /= aerial_cnt     
                    msg = "[ITER {}] Evaluating {} Aerial: L1 {:.6f} PSNR {:.6f}".format(iteration, config['name'], l1_test_aerial, psnr_test_aerial)
                    tqdm.write(msg)
                    if logger: logger.info(msg) 

                if scene.add_street and street_cnt > 0:        
                    l1_test_street /= street_cnt
                    psnr_test_street /= street_cnt        
                    msg = "[ITER {}] Evaluating {} Street: L1 {:.6f} PSNR {:.6f}".format(iteration, config['name'], l1_test_street, psnr_test_street)
                    tqdm.write(msg)
                    if logger: logger.info(msg)
                
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', len(scene.gaussians.get_anchor), iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipe, background, add_aerial, add_street):
    """
    渲染指定的数据集（train/test），保存图像、误差图和真值。
    """
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
        
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
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
        
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()

        save_rgba(rendering, os.path.join(aerial_render_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(errormap, os.path.join(aerial_error_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(gt, os.path.join(aerial_gts_path, '{0:05d}'.format(idx) + ".png"))
        aerial_visible_count_list.append(visible_count)
        aerial_per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    if len(aerial_views) > 0:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "per_view_count.json"), 'w') as fp:
            json.dump(aerial_per_view_dict, fp, indent=True)

    return aerial_visible_count_list, street_visible_count_list

def render_sets(dataset, opt, pipe, iteration, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    """
    包装函数，用于初始化环境并调用 render_set 渲染训练集和测试集。
    """
    with torch.no_grad():
        if pipe.no_prefilter_step > 0:
            pipe.add_prefilter = False
        else:
            pipe.add_prefilter = True
        modules = __import__('scene')
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        # 加载指定迭代的模型
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
    """
    读取渲染图像和真值图像用于评估。
    """
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        # 读取并预处理图像，提取RGB和Alpha
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
    """
    计算评估指标 (PSNR, SSIM, LPIPS) 并保存结果。
    """
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        # 针对特定数据集 "ucgs" 的处理逻辑 (包含特定的训练/测试划分)
        if "ucgs" in model_paths:

            base_method_dir = test_dir / method
            method_dir = base_method_dir / "street" 
            if os.path.exists(method_dir):
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                # 计算指标
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                # 记录并打印不同子集 (Held-out, View等) 的指标
                logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
                logger.info("  Held-out STREET_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs[72:-1]).mean(), ".5"))
                logger.info("  Held-out STREET_SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims[72:-1]).mean(), ".5"))
                logger.info("  Held-out STREET_LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss[72:-1]).mean(), ".5"))
                logger.info("  Held-out STREET_GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(street_visible_count[72:-1]).float().mean(), ".5"))
                
                # ... (省略重复的日志打印部分) ...
                
                # 更新结果字典
                full_dict[scene_dir][method].update({
                "Held-out STREET_PSNR": torch.tensor(psnrs[72:-1]).mean().item(),
                # ...
                })
        else:
            # 标准处理逻辑
            base_method_dir = test_dir / method
            # 评估航拍部分
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
                # ... (日志打印) ...
                print("")
                
                # 更新总指标和每张视图的指标
                full_dict[scene_dir][method].update({
                    "AERIAL_PSNR": torch.tensor(psnrs).mean().item(),
                    "AERIAL_SSIM": torch.tensor(ssims).mean().item(),
                    "AERIAL_LPIPS": torch.tensor(lpipss).mean().item(),
                    "AERIAL_GS_NUMS": torch.tensor(aerial_visible_count).float().mean().item(),
                    })

                per_view_dict[scene_dir][method].update({
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(aerial_visible_count).tolist(), image_names)}
                    })

            # 评估街景部分 (逻辑同上)
            method_dir = base_method_dir / "street" 
            if os.path.exists(method_dir):
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                # ... (计算指标、打印日志、更新字典) ...
                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
                logger.info("  STREET_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
                # ... (日志打印) ...
                print("")
                
                # 更新总指标和每张视图的指标
                full_dict[scene_dir][method].update({
                    "STREET_PSNR": torch.tensor(psnrs).mean().item(),
                    "STREET_SSIM": torch.tensor(ssims).mean().item(),
                    "STREET_LPIPS": torch.tensor(lpipss).mean().item(),
                    "STREET_GS_NUMS": torch.tensor(aerial_visible_count).float().mean().item(),
                    })

                per_view_dict[scene_dir][method].update({
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(aerial_visible_count).tolist(), image_names)}
                    })

    # 将所有评估结果写入 JSON 文件
    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    """
    配置并获取 logger，同时输出到文件和控制台。
    """
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
    # 设置命令行参数解析
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path') # 配置文件路径
    parser.add_argument('--ip', type=str, default="127.0.0.1") # GUI服务器IP
    parser.add_argument('--port', type=int, default=6009) # GUI服务器端口
    parser.add_argument('--debug_from', type=int, default=-1) # 调试起始迭代
    parser.add_argument('--detect_anomaly', action='store_true', default=False) # 检测梯度异常
    parser.add_argument('--use_wandb', action='store_true', default=False) # 使用 WandB
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1]) # 测试迭代点
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1]) # 保存模型迭代点
    parser.add_argument("--quiet", action="store_true") # 静默模式
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[]) # 保存 Checkpoint 迭代点
    parser.add_argument("--start_checkpoint", type=str, default = None) # 加载 Checkpoint 路径
    parser.add_argument("--gpu", type=str, default = '-1') # 指定 GPU
    args = parser.parse_args(sys.argv[1:])
    
    # 解析配置文件 yaml
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        args.save_iterations.append(op.iterations)

    # 设置模型保存路径
    # enable logging
    # cur_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # lp.model_path = os.path.join("outputs", lp.dataset_name, lp.scene_name, cur_time)
    lp.model_path = os.path.join("outputs", lp.dataset_name, lp.scene_name)
    os.makedirs(lp.model_path, exist_ok=True)
    # 备份配置文件
    shutil.copy(args.config, os.path.join(lp.model_path, "config.yaml"))

    # 获取 Logger
    logger = get_logger(lp.model_path)

   # [NEW] 修改这里：将间隔从 10000 改为 500
    # 设置默认的测试和保存迭代点 (如果没有指定)
    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(500, op.iterations + 1, 500)]
    
    # 确保最后一次迭代一定在列表中
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != op.iterations:
        args.test_iterations.append(op.iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [i for i in range(500, op.iterations + 1, 500)]
    
    # 确保最后一次迭代一定在列表中
    if len(args.save_iterations) == 0 or args.save_iterations[-1] != op.iterations:
        args.save_iterations.append(op.iterations)

    # 如果命令行指定了 GPU，覆盖之前的自动选择
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')
    
    # 备份代码
    try:
        saveRuntimeCode(os.path.join(lp.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
    
    # 配置 WandB
    exp_name = lp.scene_name if lp.dataset_name=="" else lp.dataset_name+"_"+lp.scene_name
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Horizon-GS",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + lp.model_path)

    # 初始化系统状态 (随机种子等)
    safe_state(args.quiet)

    # 设置 PyTorch 异常检测
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 1. 执行训练
    training(lp, op, pp, exp_name, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, wandb, logger)

    logger.info("\nTraining complete.")

    # 2. 执行渲染 (渲染最终模型)
    logger.info(f'\nStarting Rendering~')
    if lp.eval:
        aerial_visible_count, street_visible_count = render_sets(lp, op, pp, -1, skip_train=True, skip_test=False, wandb=wandb, logger=logger)
    else:
        aerial_visible_count, street_visible_count = render_sets(lp, op, pp, -1, skip_train=False, skip_test=True, wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # 3. 计算评估指标
    logger.info("\n Starting evaluation...")
    eval_name = 'test' if lp.eval else 'train'
    evaluate(lp.model_path, eval_name, aerial_visible_count=aerial_visible_count, street_visible_count=street_visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")