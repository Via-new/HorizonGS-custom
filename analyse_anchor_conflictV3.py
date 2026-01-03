#
# HorizonGS Anchor Conflict Analysis Tool V3.6 (Cyan & Purple Only)
# 
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import sys
from argparse import ArgumentParser
import yaml
from tqdm import tqdm
import logging
import shutil 

from scene import Scene
from utils.general_utils import safe_state, parse_cfg 
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import BasicPointCloud

def to_cpu(tensor):
    return tensor.detach().cpu().numpy()

# =============================================================================
# 全局变量与 Hook
# =============================================================================
HOOK_GRADS = {"opacity": None}

def opacity_hook(module, grad_input, grad_output):
    if grad_output[0] is not None:
        HOOK_GRADS["opacity"] = grad_output[0].detach()

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
    print("\n[1/7] Initializing Model...")
    modules = __import__('scene')
    model_config = dataset.model_config
    GaussModel = getattr(modules, model_config['name'])
    gaussians = GaussModel(**model_config['kwargs'])

    # 2. 加载 Checkpoint
    print(f"[2/7] Loading Checkpoint from: {args.checkpoint}")
    spatial_lr_scale = 10.0 
    dummy_pcd = BasicPointCloud(points=np.zeros((1, 3)), colors=np.zeros((1, 3)), normals=np.zeros((1, 3)))
    
    try:
        gaussians.create_from_pretrained(dummy_pcd, spatial_lr_scale, args.checkpoint, logger)
    except Exception as e:
        print(f"[Error] Failed to load checkpoint: {e}")
        return

    gaussians.training_setup(opt)
    gaussians.train()
    
    # 3. 初始化场景
    print("[3/7] Initializing Scene (Loading Cameras)...")
    scene = Scene(dataset, gaussians, shuffle=False, logger=logger, weed_ratio=pipe.weed_ratio)

    # 4. 注册 Hook
    gaussians._anchor.requires_grad_(True)
    for param in gaussians.mlp_opacity.parameters():
        param.requires_grad_(True)

    print("[4/7] Registering Gradient Hook...")
    hook_handle = gaussians.mlp_opacity.register_full_backward_hook(opacity_hook)

    # 5. 准备全量遍历列表
    aerial_indices = []
    street_indices = []
    all_cameras = scene.getTrainCameras()
    
    print("  -> Classifying cameras...")
    for idx, cam in enumerate(all_cameras):
        if "aerial" in cam.image_name.lower():
            aerial_indices.append(idx)
        elif "street" in cam.image_name.lower():
            street_indices.append(idx)
        else:
            if cam.camera_center[2] > 20.0:
                aerial_indices.append(idx)
            else:
                street_indices.append(idx)

    print(f"  -> Found {len(aerial_indices)} Aerial cameras and {len(street_indices)} Street cameras.")
    
    num_anchors = gaussians.get_anchor.shape[0]
    opa_grad_air = torch.zeros(num_anchors, device="cuda")
    opa_grad_street = torch.zeros(num_anchors, device="cuda")
    xyz_grad_air = torch.zeros((num_anchors, 3), device="cuda")
    xyz_grad_street = torch.zeros((num_anchors, 3), device="cuda")
    air_hits = torch.zeros(num_anchors, device="cuda")
    street_hits = torch.zeros(num_anchors, device="cuda")

    # 6. 运行全量分析循环
    EPOCHS = args.epochs
    print(f"\n[5/7] Running Full Coverage Analysis ({EPOCHS} Epochs)...")
    
    import gaussian_renderer
    
    total_steps = (len(aerial_indices) + len(street_indices)) * EPOCHS
    pbar = tqdm(total=total_steps, desc="Collecting Gradients")

    for epoch in range(EPOCHS):
        current_epoch_indices = aerial_indices + street_indices
        random.shuffle(current_epoch_indices)
        
        for cam_idx in current_epoch_indices:
            viewpoint_cam = all_cameras[cam_idx]
            is_aerial = cam_idx in aerial_indices

            render_pkg = gaussian_renderer.render(viewpoint_cam, gaussians, pipe, scene.background)
            image = render_pkg["render"]
            visible_mask = render_pkg["visible_mask"]

            gt_image = viewpoint_cam.original_image.cuda()
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask
                gt_image *= alpha_mask

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            HOOK_GRADS["opacity"] = None
            if gaussians._anchor.grad is not None:
                gaussians._anchor.grad.zero_() 
                
            loss.backward()
            
            with torch.no_grad():
                if HOOK_GRADS["opacity"] is not None:
                    raw_opa_grad = HOOK_GRADS["opacity"]
                    anchor_opa_grad = raw_opa_grad.sum(dim=1) 
                    if is_aerial: opa_grad_air[visible_mask] += anchor_opa_grad
                    else: opa_grad_street[visible_mask] += anchor_opa_grad

                if gaussians._anchor.grad is not None:
                    full_xyz_grad = gaussians._anchor.grad
                    visible_xyz_grad = full_xyz_grad[visible_mask]
                    if is_aerial: xyz_grad_air[visible_mask] += visible_xyz_grad
                    else: xyz_grad_street[visible_mask] += visible_xyz_grad

                if is_aerial: air_hits[visible_mask] += 1
                else: street_hits[visible_mask] += 1
                
                gaussians.optimizer.zero_grad(set_to_none=True)
                if gaussians._anchor.grad is not None: gaussians._anchor.grad.zero_()
                for param in gaussians.mlp_opacity.parameters():
                    if param.grad is not None: param.grad.zero_()
            
            pbar.update(1)

    pbar.close()
    hook_handle.remove()

    # 7. 计算统计 & 分类诊断
    print("\n[6/7] Computing Statistics & Classification...")
    
    valid_air = air_hits > 0
    valid_street = street_hits > 0
    
    avg_opa_air = torch.zeros_like(opa_grad_air)
    avg_opa_street = torch.zeros_like(opa_grad_street)
    avg_opa_air[valid_air] = opa_grad_air[valid_air] / air_hits[valid_air]
    avg_opa_street[valid_street] = opa_grad_street[valid_street] / street_hits[valid_street]

    avg_xyz_air = torch.zeros_like(xyz_grad_air)
    avg_xyz_street = torch.zeros_like(xyz_grad_street)
    avg_xyz_air[valid_air] = xyz_grad_air[valid_air] / air_hits[valid_air].unsqueeze(-1)
    avg_xyz_street[valid_street] = xyz_grad_street[valid_street] / street_hits[valid_street].unsqueeze(-1)

    # --- 过滤器 (ROI Filtering) ---
    print("\n  [Filtering Statistics]")
    
    anchors_z = gaussians.get_anchor[:, 2]
    # 1. 高度上限过滤
    if args.z_limit is None:
        z_threshold = torch.quantile(anchors_z, 0.3).item()
        print(f"   > Auto Z-Max threshold: {z_threshold:.2f}")
    else:
        z_threshold = args.z_limit
    mask_height_max = anchors_z < z_threshold
    print(f"   > Z-Max Filter (< {z_threshold:.2f}) : Removed {(~mask_height_max).sum().item()} points")

    # 2. 高度下限过滤
    z_min_threshold = args.z_min
    mask_height_min = anchors_z > z_min_threshold
    print(f"   > Z-Min Filter (> {z_min_threshold:.2f}) : Removed {(~mask_height_min).sum().item()} points")

    # 3. 距离过滤
    anchors_dist = torch.norm(gaussians.get_anchor, dim=1)
    dist_threshold = args.dist_limit
    mask_dist = anchors_dist < dist_threshold
    print(f"   > Dist Filter (< {dist_threshold:.0f})   : Removed {(~mask_dist).sum().item()} points")

    # 4. 尺度过滤
    scales = gaussians.get_scaling
    max_scales = scales.max(dim=1).values
    scale_threshold = args.scale_limit
    mask_scale = max_scales < scale_threshold
    print(f"   > Scale Filter (< {scale_threshold:.1f})   : Removed {(~mask_scale).sum().item()} points")

    base_mask = mask_height_max & mask_height_min & mask_dist & mask_scale & valid_street 
    
    print("-" * 60)
    
    # --- 冲突分类 ---
    EPS = 1e-7
    sign_s = torch.sign(avg_opa_street)
    sign_a = torch.sign(avg_opa_air)
    sign_s[avg_opa_street.abs() < EPS] = 0
    sign_a[avg_opa_air.abs() < EPS] = 0

    mask_opa_conflict = base_mask & ((sign_s * sign_a) < 0)

    abs_opa_street = avg_opa_street.abs()
    opa_threshold_street = torch.quantile(abs_opa_street[valid_street], 0.6) 
    mask_street_neg = base_mask & (sign_s < 0) & (abs_opa_street > opa_threshold_street) & (sign_a == 0)

    norm_air = torch.norm(avg_xyz_air, dim=1)
    norm_street = torch.norm(avg_xyz_street, dim=1)
    
    mag_threshold_air = torch.quantile(norm_air[valid_air], 0.8) 
    mag_threshold_street = torch.quantile(norm_street[valid_street], 0.8)
    
    is_significant_move = (norm_air > mag_threshold_air) & (norm_street > mag_threshold_street)
    
    dot_product = (avg_xyz_air * avg_xyz_street).sum(dim=1)
    cosine_sim = dot_product / (norm_air * norm_street + 1e-8)
    
    mask_geom_pull = base_mask & is_significant_move & (cosine_sim < -0.5)

    ratio_street_air = norm_street / (norm_air + 1e-8)
    mask_geom_asym = base_mask & (norm_street > mag_threshold_street) & (ratio_street_air > 5.0)

    # ==========================
    # [修改] 目标筛选逻辑: 只选青色和紫色
    # ==========================
    mask_dual = (mask_opa_conflict | mask_street_neg) & (mask_geom_pull | mask_geom_asym) # PURPLE
    
    # 注意：mask_geom_asym 包含了 Purple 的一部分，这里我们取并集
    # 只要是 悬浮(Cyan) 或者 双重冲突(Purple)，都选中
    combined_mask = mask_geom_asym | mask_dual 
    
    combined_indices = torch.nonzero(combined_mask).squeeze()
    
    # 为了可视化方便，还是生成所有颜色的 map
    color_map = torch.zeros((num_anchors, 3), dtype=torch.float32, device="cuda")
    color_map[mask_street_neg] = torch.tensor([0.0, 1.0, 0.0], device="cuda") # Green
    color_map[mask_geom_asym] = torch.tensor([0.0, 1.0, 1.0], device="cuda") # Cyan
    color_map[mask_opa_conflict] = torch.tensor([1.0, 0.0, 0.0], device="cuda") # Red
    color_map[mask_geom_pull] = torch.tensor([1.0, 1.0, 0.0], device="cuda") # Yellow
    color_map[mask_dual] = torch.tensor([1.0, 0.0, 1.0], device="cuda") # Purple

    n_total = combined_mask.sum().item()
    print("=" * 60)
    print(f"GRADIENT DIAGNOSIS REPORT (TARGET: CYAN & PURPLE)")
    print("-" * 60)
    print(f"  > [CYAN]   Street-Dominant Move : {mask_geom_asym.sum().item()}")
    print(f"  > [PURPLE] Dual Conflict        : {mask_dual.sum().item()}")
    print("-" * 60)
    print(f"  > TOTAL SELECTED FOR SPLIT      : {n_total}")
    print("=" * 60)

    # 8. 保存结果
    print("\n[7/7] Saving Results...")
    output_dir = "debug_anchor_conflict"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if combined_indices.numel() > 0:
        valid_colors = color_map[combined_indices]
        save_dict = {"indices": combined_indices, "colors": valid_colors}
        torch.save(save_dict, os.path.join(output_dir, "conflict_indices.pt"))
        print(f"  [Save] Conflict Data -> {output_dir}/conflict_indices.pt")
    else:
        print("  [Info] No conflicts found.")

    # 保存PLY
    try:
        all_xyz = to_cpu(gaussians.get_anchor)
        vis_mask = combined_mask
        if vis_mask.sum() > 0:
            subset_xyz = all_xyz[to_cpu(vis_mask).astype(bool)]
            subset_colors = (to_cpu(color_map[vis_mask]) * 255).astype(np.uint8)
            storePly(os.path.join(output_dir, "diagnostic_vis.ply"), subset_xyz, subset_colors)
            print(f"  [Save] PLY Saved.")
    except Exception as e:
        print(f"  [Warn] Failed to save PLY: {e}")

    print("\nDone.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--quantile', type=float, default=0.0)
    parser.add_argument('--z_limit', type=float, default=None)
    parser.add_argument('--z_min', type=float, default=-2.0)
    parser.add_argument('--dist_limit', type=float, default=500.0)
    parser.add_argument('--scale_limit', type=float, default=3.0)
    parser.add_argument('--debug_road', action='store_true')
    parser.add_argument('--road_z', type=float, default=None)
    parser.add_argument("--gpu", type=str, default='0')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        
    lp.model_path = os.path.dirname(os.path.dirname(args.checkpoint))
    conflict_analysis(lp, op, pp, args)