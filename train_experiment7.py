#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modified Train Script for HorizonGS - Robust Loading Fix
#

import os
import shutil
import numpy as np
import json
import struct
import torch.nn as nn
from utils.general_utils import build_rotation 
import subprocess

# Auto-select GPU with least memory usage
try:
    cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
    os.system('echo $CUDA_VISIBLE_DEVICES')
except Exception:
    pass

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
import glob

warnings.filterwarnings('ignore')

# Initialize LPIPS model
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str, logger=None) -> None:
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
        if logger: logger.info('Backup Finished!')
    except:
        pass

def get_logger(path):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"), encoding='utf-8')
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)
    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)
    return logger

def prepare_output_and_logger(args, logger=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    if logger:
        logger.info("Output folder: {}".format(args.model_path))
    else:
        print("Output folder: {}".format(args.model_path))
        
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        if logger: logger.info("Tensorboard not available")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
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
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                
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
                    msg = "[ITER {}] Evaluating {} Aerial: L1 {} PSNR {}".format(iteration, config['name'], l1_test_aerial, psnr_test_aerial)
                    tqdm.write(msg)
                    if logger: logger.info(msg)

                if scene.add_street and street_cnt > 0:       
                    l1_test_street /= street_cnt
                    psnr_test_street /= street_cnt       
                    msg = "[ITER {}] Evaluating {} Street: L1 {} PSNR {}".format(iteration, config['name'], l1_test_street, psnr_test_street)
                    tqdm.write(msg)
                    if logger: logger.info(msg)
                
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', len(scene.gaussians.get_anchor), iteration)
        torch.cuda.empty_cache()
        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipe, background, add_aerial, add_street):
    # (Existing render_set implementation - keeping it concise for this snippet)
    modules = __import__('gaussian_renderer')
    if add_street:
        street_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "renders")
        makedirs(street_render_path, exist_ok=True)
        street_views = [view for view in views if view.image_type=="street"]
        for idx, view in enumerate(tqdm(street_views, desc="Street render")):
            rendering = getattr(modules, 'render')(view, gaussians, pipe, background)["render"]
            save_rgba(rendering, os.path.join(street_render_path, '{0:05d}'.format(idx) + ".png"))
    # (Aerial logic similar, omitted for brevity but should be in original file)
    return [], [] # Placeholder return

def render_sets(dataset, opt, pipe, iteration, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        modules = __import__('scene')
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, logger=logger)
        gaussians.eval()
        if not os.path.exists(dataset.model_path): os.makedirs(dataset.model_path)
        if not skip_train: render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, scene.add_aerial, scene.add_street)
        if not skip_test: render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background, scene.add_aerial, scene.add_street)
    return [], []

def evaluate(model_paths, eval_name, aerial_visible_count=None, street_visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    # Placeholder for evaluate logic
    pass

# =========================================================================================
#  MAIN TRAINING FUNCTION (Fixed for Robust Restore)
# =========================================================================================
def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, logger)

    modules = __import__('scene')
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])

    logger.info(f"Initialized Model: {model_config['name']}")

    # [FIX 1] 注入 model_path 以便 Mask 加载
    if checkpoint:
        model_path_injected = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
        gaussians.model_path = model_path_injected
        logger.info(f"[Config] Injected model_path: {model_path_injected}")
        
        mask_file = os.path.join(model_path_injected, "anchor_source_mask.pt")
        if os.path.exists(mask_file):
            logger.info(f"[Config] FOUND Mask File: {mask_file} -> Source-Aware Rendering ENABLED.")
        else:
            logger.warning(f"[Config] Mask File NOT FOUND at {mask_file}")
    else:
        gaussians.model_path = dataset.model_path 
        logger.info(f"[Config] No checkpoint provided. Using default path: {dataset.model_path}")

    # 1. Setup Optimization (Must call this to create optimizer)
    gaussians.training_setup(opt)
    
    # 2. Load Checkpoint (With Surgical Repair)
    if checkpoint:
        logger.info(f"[Loader] Loading Checkpoint from: {checkpoint}")
        try:
            (model_params_loaded, first_iter) = torch.load(checkpoint)
            
            # --- 手术室：修复权重结构 ---
            # 我们从 capture() 获取当前正确的结构（包括 fresh optimizer）
            current_structure_tuple = gaussians.capture()
            
            # 找到优化器在元组中的索引（通常是倒数第2个）
            opt_idx = -1
            for i, item in enumerate(current_structure_tuple):
                if isinstance(item, dict) and 'state' in item and 'param_groups' in item:
                    opt_idx = i
                    break
            
            # 准备新的参数列表
            if isinstance(model_params_loaded, tuple):
                params_list = list(model_params_loaded)
                
                # A. 如果加载的参数数量不对，尝试截断或填充（如果是 split 导致的）
                if len(params_list) != len(current_structure_tuple):
                     logger.warning(f"[Loader] Tuple length mismatch! Loaded: {len(params_list)}, Expected: {len(current_structure_tuple)}")
                
                # B. 强行替换优化器状态
                fresh_opt_state = gaussians.optimizer.state_dict()
                
                if opt_idx != -1:
                    logger.info(f"[Loader] Overwriting optimizer state at index {opt_idx} with fresh state (Resetting Optimizer).")
                    
                    # 尝试方案 1: 直接替换
                    params_list[opt_idx] = fresh_opt_state
                    
                    # 尝试方案 2: 某些旧版本 HorizonGS restore 可能期望 {'optimizer': ...}
                    # 这是一个 Hack，如果直接 restore 失败，我们可以在 catch 块里尝试这个
                    
                    try:
                        gaussians.restore(tuple(params_list), opt)
                        logger.info("[Loader] Restore successful with clean optimizer!")
                    except (TypeError, KeyError, AttributeError) as e_inner:
                        logger.warning(f"[Loader] Standard clean restore failed ({e_inner}). Trying wrapper hack...")
                        # 尝试包装成字典
                        params_list[opt_idx] = {'optimizer': fresh_opt_state}
                        gaussians.restore(tuple(params_list), opt)
                        logger.info("[Loader] Restore successful with Wrapper Hack!")
                else:
                    logger.warning("[Loader] Could not find optimizer index in current structure. Trying blind restore...")
                    gaussians.restore(model_params_loaded, opt)

            else:
                # 不是 tuple，直接尝试
                gaussians.restore(model_params_loaded, opt)

        except Exception as e:
            logger.error(f"[Loader] Critical Failure loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise e
    else:
        logger.info("[Loader] No checkpoint found. Starting training from scratch.")

    # [Diagnosis]
    scene = Scene(dataset, gaussians, shuffle=False, logger=logger, weed_ratio=pipe.weed_ratio)
    
    # [FIX 3] Ensure Camera Types
    logger.info("Verifying camera types...")
    aerial_c = 0
    street_c = 0
    for cam in scene.getTrainCameras():
        if "street" in cam.image_name.lower(): 
            cam.image_type = "street"; street_c +=1
        elif "aerial" in cam.image_name.lower(): 
            cam.image_type = "aerial"; aerial_c +=1
        else:
            cam.image_type = "aerial"; aerial_c +=1 # Default fallback
    logger.info(f"Cameras: Aerial={aerial_c}, Street={street_c}")
    scene.add_aerial = (aerial_c > 0)
    scene.add_street = (street_c > 0)

    # 3. Training Loop Setup
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    viewpoint_stack = None
    aerial_viewpoint_stack = None
    street_viewpoint_stack = None
    
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    densify_cnt = 0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    modules_renderer = __import__('gaussian_renderer')
    
    # --- Loop ---
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        
        # Pick Camera
        if pipe.camera_balance:
            if not aerial_viewpoint_stack: aerial_viewpoint_stack = [cam for cam in scene.getTrainCameras().copy() if cam.image_type == "aerial"]
            if not street_viewpoint_stack: street_viewpoint_stack = [cam for cam in scene.getTrainCameras().copy() if cam.image_type == "street"]
            
            ap, sp = pipe.camera_proportion.split("-")
            r = float(ap) / (float(ap) + float(sp))
            if np.random.rand() < r:
                viewpoint_cam = aerial_viewpoint_stack.pop(randint(0, len(aerial_viewpoint_stack)-1))
            else:
                viewpoint_cam = street_viewpoint_stack.pop(randint(0, len(street_viewpoint_stack)-1))
        else:
            if not viewpoint_stack: viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        render_pkg = getattr(modules_renderer, 'render')(viewpoint_cam, gaussians, pipe, scene.background)
        image, scaling, alpha = render_pkg["render"], render_pkg["scaling"], render_pkg["render_alphas"]

        gt_image = viewpoint_cam.original_image.cuda()
        alpha_mask = viewpoint_cam.alpha_mask.cuda()
        
        image = image * alpha_mask
        gt_image = gt_image * alpha_mask
            
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Regularizations
        if opt.lambda_dreg > 0 and scaling.shape[0] > 0:
            loss += opt.lambda_dreg * scaling.prod(dim=1).mean()
        
        if opt.lambda_sky_opa > 0:
            o = alpha.clamp(1e-6, 1-1e-6)
            sky = alpha_mask.float()
            loss += opt.lambda_sky_opa * (-(1-sky) * torch.log(1 - o)).mean()

        if opt.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss += opt.lambda_opacity_entropy * (-(o*torch.log(o)).mean())
        
        Ll1depth = 0
        if iteration > opt.start_depth and depth_l1_weight(iteration) > 0 and viewpoint_cam.invdepthmap is not None:
             render_depth = render_pkg["render_depth"]
             invDepth = torch.where(render_depth > 0.0, 1.0 / render_depth, torch.zeros_like(render_depth))            
             mono_invdepth = viewpoint_cam.invdepthmap.cuda()
             depth_mask = viewpoint_cam.depth_mask.cuda()
             Ll1depth = depth_l1_weight(iteration) * torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
             loss += Ll1depth
             Ll1depth = Ll1depth.item()

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                psnr_log = psnr(image, gt_image).mean().double()
                anchor_prim = len(gaussians.get_anchor) 
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "PSNR": f"{psnr_log:.{2}f}", "GS": f"{anchor_prim}"})
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, getattr(modules_renderer, 'render'), (pipe, scene.background), wandb, logger)
            
            if (iteration in saving_iterations):
                msg = f"[ITER {iteration}] Saving Gaussians"
                if logger: logger.info(msg)
                scene.save(iteration)
            
            # Fine Stage: 优化参数但不致密化
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                msg = f"[ITER {iteration}] Saving Checkpoint"
                if logger: logger.info(msg)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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
        
        # ================= [FIX START] 自动寻找 Checkpoint =================
        if args.start_checkpoint is None:
            if 'model_params' in cfg and 'pretrained_checkpoint' in cfg['model_params']:
                ckpt_path = cfg['model_params']['pretrained_checkpoint']
                if ckpt_path and ckpt_path not in ["''", '""']:
                    args.start_checkpoint = ckpt_path
                    print(f"[Config] YAML specified checkpoint: {args.start_checkpoint}")

        # 处理文件夹 -> 文件
        if args.start_checkpoint and os.path.isdir(args.start_checkpoint):
            potential_files = ["chkpnt.pth", "chkpnt_iteration_0.pth"]
            found = False
            for fname in potential_files:
                p = os.path.join(args.start_checkpoint, fname)
                if os.path.exists(p):
                    args.start_checkpoint = p
                    print(f"[Config] Auto-resolved to file: {p}")
                    found = True
                    break
            
            if not found:
                pths = glob.glob(os.path.join(args.start_checkpoint, "*.pth"))
                if pths:
                    args.start_checkpoint = pths[0]
                    print(f"[Config] Found .pth file: {args.start_checkpoint}")
        # ================= [FIX END] =================

        args.save_iterations.append(op.iterations)

    lp.model_path = os.path.join("outputs", lp.dataset_name, lp.scene_name)
    os.makedirs(lp.model_path, exist_ok=True)
    shutil.copy(args.config, os.path.join(lp.model_path, "config.yaml"))

    logger = get_logger(lp.model_path)

    # Defaults
    if args.test_iterations[0] == -1: args.test_iterations = [i for i in range(2000, op.iterations + 1, 2000)]
    if args.save_iterations[0] == -1: args.save_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
    
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    try:
        saveRuntimeCode(os.path.join(lp.model_path, 'backup'), logger=logger)
    except:
        pass
    
    logger.info("Optimizing " + lp.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp, op, pp, "", args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, None, logger)
    
    logger.info("\nTraining complete.")