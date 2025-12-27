#
# HorizonGS Anchor Conflict Analysis Tool V2.1 (Color Export Support)
# 
import os
import torch
import numpy as np
import random
import sys
from argparse import ArgumentParser
import yaml
from tqdm import tqdm
import logging
import shutil 

# 引入 HorizonGS 模块
from scene import Scene
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
    for param in gaussians.mlp_opacity.parameters():
        param.requires_grad = True

    # 3. 初始化场景
    print("[3/7] Initializing Scene (Loading Cameras)...")
    scene = Scene(dataset, gaussians, shuffle=False, logger=logger, weed_ratio=pipe.weed_ratio)

    # 4. 注册 Hook
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
    
    # 准备梯度累加器
    num_anchors = gaussians.get_anchor.shape[0]
    opa_grad_air = torch.zeros(num_anchors, device="cuda")
    opa_grad_street = torch.zeros(num_anchors, device="cuda")
    
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

            # Render
            render_pkg = gaussian_renderer.render(viewpoint_cam, gaussians, pipe, scene.background)
            image = render_pkg["render"]
            visible_mask = render_pkg["visible_mask"]

            # Loss (L1 + SSIM)
            gt_image = viewpoint_cam.original_image.cuda()
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask
                gt_image *= alpha_mask

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            # Backward
            HOOK_GRADS["val"] = None
            loss.backward()
            
            # Accumulate
            with torch.no_grad():
                if HOOK_GRADS["val"] is not None:
                    raw_grad = HOOK_GRADS["val"]
                    anchor_grad = raw_grad.sum(dim=1) # [Visible_N]
                    
                    if is_aerial:
                        opa_grad_air[visible_mask] += anchor_grad
                        air_hits[visible_mask] += 1
                    else:
                        opa_grad_street[visible_mask] += anchor_grad
                        street_hits[visible_mask] += 1
                
                # Zero Grad
                gaussians.optimizer.zero_grad(set_to_none=True)
                for param in gaussians.mlp_opacity.parameters():
                    if param.grad is not None: param.grad.zero_()
            
            pbar.update(1)

    pbar.close()
    hook_handle.remove()

    # 7. 计算统计 & 分类诊断
    print("\n[6/7] Computing Statistics & Classification...")
    
    # 归一化平均梯度
    valid_air = air_hits > 0
    valid_street = street_hits > 0
    
    avg_grad_air = torch.zeros_like(opa_grad_air)
    avg_grad_street = torch.zeros_like(opa_grad_street)
    
    avg_grad_air[valid_air] = opa_grad_air[valid_air] / air_hits[valid_air]
    avg_grad_street[valid_street] = opa_grad_street[valid_street] / street_hits[valid_street]

    # --- 高度过滤 (ROI Filtering) ---
    anchors_z = gaussians.get_anchor[:, 2]
    if args.z_limit is None:
        z_threshold = torch.quantile(anchors_z, 0.3).item()
        print(f"  [Filter] Auto-detected ground Z threshold: {z_threshold:.2f} (lower 30%)")
    else:
        z_threshold = args.z_limit
        print(f"  [Filter] Using manual Z threshold: {z_threshold:.2f}")

    mask_height = anchors_z < z_threshold
    
    # --- 梯度符号提取 ---
    EPS = 1e-6 # 忽略极小梯度
    sign_s = torch.sign(avg_grad_street)
    sign_a = torch.sign(avg_grad_air)
    
    sign_s[avg_grad_street.abs() < EPS] = 0
    sign_a[avg_grad_air.abs() < EPS] = 0

    # --- 核心分类逻辑 ---
    base_mask = mask_height & valid_street 
    
    # Type A (Red): 经典冲突 (街景-, 航拍+)
    mask_conflict = base_mask & ((sign_s * sign_a) < 0)
    
    # Type B (Blue): 双重否定 (街景-, 航拍-)
    mask_double_neg = base_mask & (sign_s < 0) & (sign_a < 0)
    
    # Type C (Green): 街景单方厌恶 (街景-, 航拍0)
    mask_street_hates = base_mask & (sign_s < 0) & (sign_a == 0)

    # 统计
    n_conflict = mask_conflict.sum().item()
    n_double_neg = mask_double_neg.sum().item()
    n_street_hates = mask_street_hates.sum().item()
    
    print("=" * 60)
    print(f"GRADIENT DIAGNOSIS REPORT (Z < {z_threshold:.2f})")
    print(f"  > [RED]   True Conflict (+/-)      : {n_conflict}")
    print(f"  > [BLUE]  Double Negative (-/-)    : {n_double_neg}")
    print(f"  > [GREEN] Street Only Negative (-/0): {n_street_hates}")
    print("=" * 60)

    # 8. 保存结果
    print("\n[7/7] Saving Results...")
    output_dir = "debug_anchor_conflict"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并 Mask
    combined_mask = mask_conflict | mask_double_neg | mask_street_hates
    combined_indices = torch.nonzero(combined_mask).squeeze()
    
    # --- [关键修改] 生成颜色 Tensor 供 Render 使用 ---
    # 默认颜色 (不会用到，因为只保存 indices 对应的)
    color_map = torch.zeros((num_anchors, 3), dtype=torch.float32, device="cuda")
    
    # RED: Conflict
    color_map[mask_conflict] = torch.tensor([1.0, 0.0, 0.0], device="cuda")
    # BLUE: Double Negative
    color_map[mask_double_neg] = torch.tensor([0.0, 0.0, 1.0], device="cuda")
    # GREEN: Street Hates
    color_map[mask_street_hates] = torch.tensor([0.0, 1.0, 0.0], device="cuda")
    
    # 提取有效点的颜色
    valid_colors = color_map[combined_indices] # [N_valid, 3]
    
    # 保存为字典
    save_dict = {
        "indices": combined_indices,
        "colors": valid_colors
    }
    
    save_path = os.path.join(output_dir, "conflict_indices.pt")
    torch.save(save_dict, save_path)
    print(f"  [Save] Conflict Data (Indices + Colors) -> {save_path}")

    # 2. 保存诊断 PLY (仅保存这些有问题的点，带颜色)
    try:
        all_xyz = to_cpu(gaussians.get_anchor)
        
        # 提取有问题的点
        problem_mask = to_cpu(combined_mask).astype(bool)
        
        if problem_mask.sum() > 0:
            subset_xyz = all_xyz[problem_mask]
            subset_colors = np.zeros_like(subset_xyz)
            
            # 使用刚才生成的颜色
            subset_colors_float = to_cpu(valid_colors)
            subset_colors = (subset_colors_float * 255).astype(np.uint8)
            
            ply_path = os.path.join(output_dir, "diagnostic_vis.ply")
            storePly(ply_path, subset_xyz, subset_colors)
            print(f"  [Save] Diagnostic Point Cloud -> {ply_path}")
        else:
            print("  [Info] No problem anchors found in ROI.")

    except Exception as e:
        print(f"  [Warn] Failed to save PLY: {e}")

    print("\nDone.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1, help="遍历全数据集的次数 (建议 1 或 2)")
    parser.add_argument('--quantile', type=float, default=0.0, help="活跃度阈值 (未使用)")
    parser.add_argument('--z_limit', type=float, default=None, help="Z轴高度限制 (例如 2.0)")
    parser.add_argument("--gpu", type=str, default='0')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        
    lp.model_path = os.path.dirname(os.path.dirname(args.checkpoint))
    conflict_analysis(lp, op, pp, args)