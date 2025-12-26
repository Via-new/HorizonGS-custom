#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modified for HorizonGS Neural Opacity Conflict Analysis (v3 - Hook Version)
#

import os
import torch
import numpy as np
from random import randint
import sys
from scene import Scene
from utils.general_utils import safe_state, parse_cfg
from utils.loss_utils import l1_loss, ssim
from argparse import ArgumentParser
import yaml
from tqdm import tqdm
from utils.graphics_utils import BasicPointCloud

# 全局变量用于在 Hook 和主循环间传递梯度
HOOK_GRADS = {"val": None}

def opacity_hook(module, grad_input, grad_output):
    """
    Hook 函数：截获 MLP 输出的梯度
    grad_output[0] 是 dL / d(MLP_Output)，形状通常为 [N_anchors, N_offsets]
    """
    if grad_output[0] is not None:
        HOOK_GRADS["val"] = grad_output[0].detach()

def training_conflict_analysis(dataset, opt, pipe, dataset_name, stats_iterations=500, checkpoint=None, logger=None):
    # 1. 初始化模型
    modules = __import__('scene')
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
    
    # 2. 初始化场景
    scene = Scene(dataset, gaussians, shuffle=False, logger=logger, weed_ratio=pipe.weed_ratio)

    # 3. 相机分类
    logger.info("Classifying camera types...")
    aerial_count = 0
    street_count = 0
    for cam in scene.getTrainCameras():
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
    
    if street_count == 0:
        logger.warning("No street cameras found! Analysis may not be valid.")

    # 4. 加载模型
    gaussians.training_setup(opt)
    if checkpoint:
        if os.path.isdir(checkpoint):
            logger.info(f"Loading from point cloud folder: {checkpoint}")
            spatial_lr_scale = scene.cameras_extent
            dummy_pcd = BasicPointCloud(points=np.zeros((1, 3)), colors=np.zeros((1, 3)), normals=np.zeros((1, 3)))
            try:
                gaussians.create_from_pretrained(dummy_pcd, spatial_lr_scale, checkpoint, logger)
            except Exception as e:
                logger.error(f"Failed to load: {e}")
                return
        elif checkpoint.endswith(".pth"):
            logger.info(f"Loading from checkpoint: {checkpoint}")
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
    else:
        logger.info("Running from initialization.")

    # [Critical Fix] 注册 Hook 到 Opacity MLP
    # 我们Hook到最后一个线性层或者激活层。根据 lod_model.py:
    # self.mlp_opacity = nn.Sequential(..., nn.Linear(...), nn.Tanh())
    # 我们Hook整个Sequential模块，获取最终输出的梯度
    logger.info("Registering Hook to Opacity MLP...")
    hook_handle = gaussians.mlp_opacity.register_full_backward_hook(opacity_hook)

    # 5. 初始化梯度累加器
    num_anchors = gaussians.get_anchor.shape[0]
    
    # Opacity 梯度通常是 (N, n_offsets)
    # 我们将其求和压缩为 (N,)，代表该 Anchor 整体是想变实还是变虚
    opa_grad_air = torch.zeros(num_anchors, device="cuda")
    opa_grad_street = torch.zeros(num_anchors, device="cuda")
    
    # 活跃度统计 (梯度的绝对值)
    opa_active_air = torch.zeros(num_anchors, device="cuda")
    opa_active_street = torch.zeros(num_anchors, device="cuda")
    
    air_steps = 0
    street_steps = 0

    # 6. 统计循环
    logger.info(f"Accumulating NEURAL OPACITY Gradients for {stats_iterations} iters...")
    gaussians.train()
    opt.densification = False 

    aerial_stack = None
    street_stack = None

    for iteration in tqdm(range(1, stats_iterations + 1), desc="Training"):
        if not aerial_stack: aerial_stack = [c for c in scene.getTrainCameras() if c.image_type == "aerial"]
        if not street_stack: street_stack = [c for c in scene.getTrainCameras() if c.image_type == "street"]
        if len(aerial_stack) == 0: aerial_stack = [c for c in scene.getTrainCameras() if c.image_type == "aerial"]
        if len(street_stack) == 0: street_stack = [c for c in scene.getTrainCameras() if c.image_type == "street"]

        if iteration % 2 == 0:
            viewpoint_cam = aerial_stack.pop(randint(0, len(aerial_stack)-1))
        else:
            viewpoint_cam = street_stack.pop(randint(0, len(street_stack)-1))

        # Forward
        modules = __import__('gaussian_renderer')
        render_pkg = getattr(modules, 'render')(viewpoint_cam, gaussians, pipe, scene.background)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda() * viewpoint_cam.alpha_mask.cuda()
        image = image * viewpoint_cam.alpha_mask.cuda()

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 清空 Hook 缓存
        HOOK_GRADS["val"] = None
        
        # Backward (这将触发 Hook)
        loss.backward()

        # --- 核心逻辑: 从 Hook 中提取梯度 ---
        with torch.no_grad():
            if HOOK_GRADS["val"] is not None:
                # raw_grad shape: [N_anchors, N_offsets] (例如 [N, 10])
                raw_grad = HOOK_GRADS["val"]
                
                # Sum over offsets to get "Per-Anchor" opacity tendency
                # Sum > 0 means most offsets want to increase opacity
                anchor_opa_grad = raw_grad.sum(dim=1) 
                
                if viewpoint_cam.image_type == "aerial":
                    opa_grad_air += anchor_opa_grad
                    opa_active_air += anchor_opa_grad.abs()
                    air_steps += 1
                else:
                    opa_grad_street += anchor_opa_grad
                    opa_active_street += anchor_opa_grad.abs()
                    street_steps += 1
            
            gaussians.optimizer.zero_grad(set_to_none=True)

    # 移除 Hook
    hook_handle.remove()

    # 7. 深入分析
    logger.info("Calculating Neural Opacity Conflict...")
    
    if air_steps > 0: 
        opa_grad_air /= air_steps
        opa_active_air /= air_steps
    if street_steps > 0: 
        opa_grad_street /= street_steps
        opa_active_street /= street_steps

    # 符号积：正数表示方向一致，负数表示方向相反（冲突）
    sign_product = opa_grad_air * opa_grad_street
    
    # 筛选 Top 20% 活跃锚点
    def get_mask(tensor, percentile=0.8): 
        if tensor.numel() == 0: return tensor > 0
        threshold = torch.quantile(tensor, percentile)
        return tensor > threshold

    active_air = get_mask(opa_active_air, 0.8)
    active_street = get_mask(opa_active_street, 0.8)
    shared_active = active_air & active_street
    num_shared = shared_active.sum().item()

    if num_shared > 0:
        conflict_vals = sign_product[shared_active]
        
        num_opposite = (conflict_vals < 0).sum().item()
        num_aligned = (conflict_vals > 0).sum().item()
        
        print("\n" + "="*70)
        print(f"NEURAL OPACITY CONFLICT ANALYSIS")
        print(f"Target: Top 20% Shared Active Anchors ({num_shared} anchors)")
        print("-" * 70)
        print(f"Conflict (Opposite Signs): {num_opposite} ({num_opposite/num_shared*100:.2f}%)  <-- The 'Fog' Cause")
        print(f"Consensus (Same Signs)   : {num_aligned} ({num_aligned/num_shared*100:.2f}%)")
        print("="*70 + "\n")
        
        # Level 0 专项
        if hasattr(gaussians, "_level"):
            is_level0 = (gaussians._level == 0).squeeze()
            l0_shared_mask = shared_active & is_level0
            num_l0_shared = l0_shared_mask.sum().item()
            if num_l0_shared > 0:
                l0_conflicts = sign_product[l0_shared_mask]
                num_l0_opp = (l0_conflicts < 0).sum().item()
                
                # 冲突类型细分
                air_grad = opa_grad_air[l0_shared_mask]
                street_grad = opa_grad_street[l0_shared_mask]
                
                # Type A: 航拍想变实 (+)，街景想变虚 (-)
                type_A = ((air_grad > 0) & (street_grad < 0)).sum().item()
                # Type B: 航拍想变虚 (-)，街景想变实 (+)
                type_B = ((air_grad < 0) & (street_grad > 0)).sum().item()

                print(f"--- Level 0 Opacity Detail ---")
                print(f"Shared Active L0: {num_l0_shared}")
                print(f"True Conflict (Opposite): {num_l0_opp} ({num_l0_opp/num_l0_shared*100:.2f}%)")
                print("-" * 30)
                print(f"  > Type A (Air wants SOLID / Street wants TRANSPARENT): {type_A} ({type_A/num_l0_shared*100:.2f}%)")
                print(f"  > Type B (Air wants TRANSPARENT / Street wants SOLID): {type_B} ({type_B/num_l0_shared*100:.2f}%)")
                print("-" * 70)
    else:
        print("No shared active anchors found.")

def get_logger(path):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    if not logger.handlers:
        controlshow = logging.StreamHandler()
        controlshow.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        controlshow.setFormatter(formatter)
        logger.addHandler(controlshow)
    return logger

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--stats_iter', type=int, default=500)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)

    lp.model_path = os.path.join("outputs", "conflict_analysis_v3", lp.dataset_name, lp.scene_name)
    os.makedirs(lp.model_path, exist_ok=True)
    lp.eval = False
    
    logger = get_logger(".")
    training_conflict_analysis(lp, op, pp, "ConflictStatsV3", args.stats_iter, args.checkpoint, logger)