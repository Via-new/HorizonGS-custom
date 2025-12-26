#
# HorizonGS Anchor Conflict Analysis Tool (v5.1 - Full Coverage + Only Conflict Output)
# 
# 核心改进：
# 1. 弃用随机采样，改为全数据集遍历 (Epochs)。
# 2. 确保全场景所有能被看到的锚点都被诊断一遍。
# 3. [新增] 自动输出 conflict_ONLY.ply，方便在 CloudCompare 中定位微小的冲突区域。
#

import os
import torch
from torch import nn
import numpy as np
import random
import sys
from argparse import ArgumentParser
import yaml
from tqdm import tqdm
import logging
import shutil # 用于文件夹操作

# 引入 HorizonGS 模块
from scene import Scene
# 移除不存在的引用
from utils.general_utils import safe_state, parse_cfg 
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import BasicPointCloud

def to_cpu(tensor):
    return tensor.detach().cpu().numpy()

# =============================================================================
# 全局变量与 Hook
# =============================================================================
HOOK_GRADS = {"val": None}

def opacity_hook(module, grad_input, grad_output):
    if grad_output[0] is not None:
        HOOK_GRADS["val"] = grad_output[0].detach()

# =============================================================================
# 自定义 PLY 保存函数
# =============================================================================
def storePly(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# =============================================================================
# 主逻辑
# =============================================================================
def conflict_analysis(dataset, opt, pipe, args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)

    # 1. 初始化模型
    print("\n[1/6] Initializing Model...")
    modules = __import__('scene')
    model_config = dataset.model_config
    GaussModel = getattr(modules, model_config['name'])
    gaussians = GaussModel(**model_config['kwargs'])

    # 2. 加载 Checkpoint
    print(f"[2/6] Loading Checkpoint from: {args.checkpoint}")
    spatial_lr_scale = 10.0 
    dummy_pcd = BasicPointCloud(points=np.zeros((1, 3)), colors=np.zeros((1, 3)), normals=np.zeros((1, 3)))
    
    try:
        gaussians.create_from_pretrained(dummy_pcd, spatial_lr_scale, args.checkpoint, logger)
    except Exception as e:
        print(f"[Error] Failed to load checkpoint: {e}")
        return

    gaussians.training_setup(opt)
    gaussians.train()
    for param in gaussians.mlp_opacity.parameters():
        param.requires_grad = True

    # 3. 初始化场景
    print("[3/6] Initializing Scene (Loading Cameras)...")
    scene = Scene(dataset, gaussians, shuffle=False, logger=logger, weed_ratio=pipe.weed_ratio)

    # 4. 注册 Hook
    print("[4/6] Registering Gradient Hook...")
    hook_handle = gaussians.mlp_opacity.register_full_backward_hook(opacity_hook)

    # 5. 准备全量遍历列表
    # -------------------------------------------------------------------------
    # 区分相机类型
    aerial_indices = []
    street_indices = []
    all_cameras = scene.getTrainCameras()
    
    print("  -> Classifying cameras...")
    for idx, cam in enumerate(all_cameras):
        # 简单分类逻辑：根据文件名
        if "aerial" in cam.image_name.lower():
            aerial_indices.append(idx)
        elif "street" in cam.image_name.lower():
            street_indices.append(idx)
        else:
            # 如果文件名不含关键字，回退到高度判断 (假设 Z 轴)
            if cam.camera_center[2] > 20.0: # 假设 >20m 是航拍
                aerial_indices.append(idx)
            else:
                street_indices.append(idx)

    print(f"  -> Found {len(aerial_indices)} Aerial cameras and {len(street_indices)} Street cameras.")
    
    # 准备统计容器
    num_anchors = gaussians.get_anchor.shape[0]
    opa_grad_air = torch.zeros(num_anchors, device="cuda")
    opa_grad_street = torch.zeros(num_anchors, device="cuda")
    opa_active_air = torch.zeros(num_anchors, device="cuda")
    opa_active_street = torch.zeros(num_anchors, device="cuda")
    
    air_hits = torch.zeros(num_anchors, device="cuda") # 记录每个锚点被航拍看到的次数
    street_hits = torch.zeros(num_anchors, device="cuda") # 记录每个锚点被街景看到的次数

    # 6. 运行全量分析循环 (Epochs)
    # -------------------------------------------------------------------------
    EPOCHS = args.epochs
    print(f"\n[5/6] Running Full Coverage Analysis ({EPOCHS} Epochs)...")
    
    import gaussian_renderer
    
    total_steps = (len(aerial_indices) + len(street_indices)) * EPOCHS
    pbar = tqdm(total=total_steps, desc="Processing All Views")

    for epoch in range(EPOCHS):
        # 每个 Epoch 重新打乱顺序，模拟真实训练分布
        current_epoch_indices = aerial_indices + street_indices
        random.shuffle(current_epoch_indices)
        
        for cam_idx in current_epoch_indices:
            viewpoint_cam = all_cameras[cam_idx]
            
            # 判断当前是航拍还是街景
            is_aerial = cam_idx in aerial_indices

            # Render
            render_pkg = gaussian_renderer.render(viewpoint_cam, gaussians, pipe, scene.background)
            image = render_pkg["render"]
            visible_mask = render_pkg["visible_mask"] # [N_anchors] boolean

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask
                gt_image *= alpha_mask

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            # Reset Hook & Backward
            HOOK_GRADS["val"] = None
            loss.backward()
            
            # Accumulate
            with torch.no_grad():
                if HOOK_GRADS["val"] is not None:
                    raw_grad = HOOK_GRADS["val"]
                    anchor_grad = raw_grad.sum(dim=1) # [Visible_N]
                    
                    if is_aerial:
                        opa_grad_air[visible_mask] += anchor_grad
                        opa_active_air[visible_mask] += anchor_grad.abs()
                        air_hits[visible_mask] += 1
                    else:
                        opa_grad_street[visible_mask] += anchor_grad
                        opa_active_street[visible_mask] += anchor_grad.abs()
                        street_hits[visible_mask] += 1
                
                gaussians.optimizer.zero_grad(set_to_none=True)
                for param in gaussians.mlp_opacity.parameters():
                    if param.grad is not None: param.grad.zero_()
            
            pbar.update(1)

    pbar.close()
    hook_handle.remove()

    # 7. 计算统计结果 & 保存
    print("\n[6/6] Computing Statistics...")
    
    # 归一化：除以被看到的次数，而不是总迭代次数
    valid_air = air_hits > 0
    valid_street = street_hits > 0
    
    opa_grad_air[valid_air] /= air_hits[valid_air]
    opa_active_air[valid_air] /= air_hits[valid_air]
    
    opa_grad_street[valid_street] /= street_hits[valid_street]
    opa_active_street[valid_street] /= street_hits[valid_street]

    # 冲突检测
    sign_product = opa_grad_air * opa_grad_street
    
    # 活跃度 Mask
    def get_active_mask(tensor, quantile=0.0): 
        if tensor.max() <= 1e-6: return torch.zeros_like(tensor, dtype=torch.bool)
        if quantile <= 0: return tensor > 1e-6 # 只要有梯度就算 Active
        thresh = torch.quantile(tensor[tensor > 1e-6], quantile)
        return tensor > thresh

    mask_active_air = get_active_mask(opa_active_air, quantile=args.quantile)
    mask_active_street = get_active_mask(opa_active_street, quantile=args.quantile)
    mask_shared = mask_active_air & mask_active_street
    
    if hasattr(gaussians, "_level"):
        mask_l0 = (gaussians._level == 0).squeeze()
        target_mask = mask_shared & mask_l0
    else:
        target_mask = mask_shared

    # 真正的冲突点
    conflict_mask = target_mask & (sign_product < 0)
    conflict_indices = torch.nonzero(conflict_mask).squeeze()
    num_conflict = conflict_indices.numel() if conflict_indices.dim() > 0 else 0
    
    print("=" * 60)
    print(f"FULL COVERAGE ANALYSIS REPORT")
    print(f"  > Active Aerial Anchors: {mask_active_air.sum().item()}")
    print(f"  > Active Street Anchors: {mask_active_street.sum().item()}")
    print(f"  > Shared Active (L0)   : {target_mask.sum().item()}")
    print(f"  > CONFLICT ANCHORS     : {num_conflict}")
    print("=" * 60)

    # 8. 保存结果
    # -------------------------------------------------------------------------
    print("\n[6/6] Saving Results...")
    
    output_dir = "debug_anchor_conflict"

    # --- 文件夹管理 ---
    if os.path.exists(output_dir):
        print(f"  [Info] Directory '{output_dir}' exists. Clearing contents...")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"  [Info] Directory '{output_dir}' is ready.")
    
    # 保存索引
    save_path = os.path.join(output_dir, "conflict_indices.pt")
    torch.save(conflict_indices, save_path)
    print(f"  [Save] Conflict Indices -> {save_path}")
    
    # 保存可视化 PLY
    try:
        # 1. 提取所有锚点坐标
        all_xyz = to_cpu(gaussians.get_anchor)
        
        # 2. 构造颜色：默认为灰色 [128, 128, 128]
        all_colors = np.ones_like(all_xyz) * 128 
        
        # 3. 将冲突点染成红色 [255, 0, 0]
        mask_cpu = to_cpu(conflict_mask).astype(bool)
        if mask_cpu.sum() > 0:
            all_colors[mask_cpu] = np.array([255, 0, 0])
        
        # 颜色范围修正
        if mask_cpu.sum() > 0:
            conflict_colors_sample = all_colors[mask_cpu][:5]
            if conflict_colors_sample.max() <= 1.0:
                all_colors = all_colors * 255.0
                all_colors[mask_cpu] = np.array([255, 0, 0])

        # 保存 [全景图]
        ply_path = os.path.join(output_dir, "conflict_vis.ply")
        storePly(ply_path, all_xyz, all_colors)
        print(f"  [Save] Visualization PLY -> {ply_path}")

        # === [新增功能] 保存 [仅冲突点] ===
        # 这样在 CloudCompare 里打开时，背景是空的，这 2999 个点会非常非常显眼
        if mask_cpu.sum() > 0:
            conflict_xyz = all_xyz[mask_cpu]
            # 设为纯红
            conflict_rgb = np.zeros_like(conflict_xyz)
            conflict_rgb[:, 0] = 255 # R=255, G=0, B=0
            
            only_path = os.path.join(output_dir, "conflict_ONLY.ply")
            storePly(only_path, conflict_xyz, conflict_rgb)
            print(f"  [Save] Conflict Points Only -> {only_path}")
            print(f"  (Tip: Open '{only_path}' in CloudCompare to see the exact locations)")
        else:
            print("  [Info] No conflict points to save separately.")
        # =========================================
        
    except Exception as e:
        print(f"  [Warn] Failed to save viz ply: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=2, help="遍历全数据集的次数")
    parser.add_argument('--quantile', type=float, default=0.0, help="活跃度阈值")
    parser.add_argument("--gpu", type=str, default='0')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        
    lp.model_path = os.path.dirname(os.path.dirname(args.checkpoint))
    conflict_analysis(lp, op, pp, args)