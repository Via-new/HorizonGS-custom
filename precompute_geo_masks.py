#
# Precompute Geometric Overlap Masks for HorizonGS
# Fixed Import and Argument Compatibility
#

import torch
import os
import numpy as np
import cv2
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import sys

# 1. 核心导入 (仿照 train_experiment1.py)
try:
    from scene import Scene
    # 注意这里的大小写: LoD (Level of Detail)
    from scene import GaussianLoDModel
except ImportError as e:
    print(f"Import Error: {e}")
    print("Trying to fix path...")
    sys.path.append(os.getcwd())
    from scene import Scene
    from scene.lod_model import GaussianLoDModel

# ================= 几何计算核心 =================
class FrustumCuller:
    """
    计算 3D 点与相机视锥的几何关系
    """
    @staticmethod
    def filter_anchors_by_frustum(anchors, view_proj_matrix, margin=0.1): 
        # anchors: (N, 3)
        # view_proj_matrix: (4, 4) World -> Clip
        
        # 补齐齐次坐标 (N, 4)
        p_hom = torch.cat([anchors, torch.ones_like(anchors[:, :1])], dim=1) 
        
        # 投影: p_clip = p_world @ ViewProj (假设矩阵是 column-major 存储的，适合右乘)
        p_clip = p_hom @ view_proj_matrix 
        
        # 剔除相机背后的点 (w > 0)
        valid_w = p_clip[:, 3] > 0.001
        
        # 透视除法 (NDC)
        denom = p_clip[:, 3] + 1e-6
        p_ndc_x = p_clip[:, 0] / denom
        p_ndc_y = p_clip[:, 1] / denom
        
        # 判断范围 (-1 ~ 1)
        limit = 1.0 + margin 
        mask_x = (p_ndc_x > -limit) & (p_ndc_x < limit)
        mask_y = (p_ndc_y > -limit) & (p_ndc_y < limit)
        
        return valid_w & mask_x & mask_y

def get_projection_matrix(cam):
    # HorizonGS 的 full_proj_transform 已经在 CUDA 上了
    return cam.full_proj_transform

def project_to_image(points3d, cam):
    """将3D点投影到2D图像平面，返回像素坐标"""
    W, H = cam.image_width, cam.image_height
    view_proj = get_projection_matrix(cam)
    
    p_hom = torch.cat([points3d, torch.ones_like(points3d[:, :1])], dim=1)
    p_clip = p_hom @ view_proj
    
    p_w = p_clip[:, 3:4] + 1e-7
    p_x = p_clip[:, 0:1] / p_w
    p_y = p_clip[:, 1:2] / p_w
    
    # NDC (-1~1) -> Pixel Coordinates (0~W, 0~H)
    u = ((p_x + 1.0) * W - 1.0) * 0.5
    v = ((p_y + 1.0) * H - 1.0) * 0.5
    
    return torch.cat([u, v], dim=1)

def main(args):
    # 路径容错处理
    if args.source_path.endswith("/images"):
        args.source_path = os.path.dirname(args.source_path)
    
    print(f"Loading Scene from: {args.source_path}")
    
    # 2. 构造 Dummy 参数以满足 Scene 的初始化需求
    # Scene 类需要很多参数 (data_format, resolution_scales 等)
    scene_args = Namespace(
        source_path=args.source_path,
        model_path=args.model_path if args.model_path else "",
        images=args.images,
        eval=args.eval,
        data_format="colmap", # 强制指定 colmap，通常不需要动
        resolution_scales=[1.0],
        white_background=args.white_background,
        random_background=False,
        add_aerial=True, # 必须开启，否则可能不加载航拍图
        add_street=True, # 必须开启
        add_mask=False,
        add_depth=False,
        llffhold=8,
        center=None, # colmap 不需要
        scale=1.0,
        ratio=1, # 降采样率
        pretrained_checkpoint="",
        global_appearance=""
    )

    # 3. 初始化高斯模型 (Mock参数)
    # 我们只需要它作为一个占位符传给 Scene，不需要它真的能训练
    # 传入必要的 dummy 参数防止 __init__ 报错
    model_kwargs = {
        "sh_degree": 0,
        "feat_dim": 16, 
        "view_dim": 3,
        "n_offsets": 1,
        "voxel_size": 0.1,
        "standard_dist": 10.0,
        "aerial_levels": 2,
        "street_levels": 2,
        "color_attr": "SH0", # 触发 active_sh_degree = 0
        "appearance_dim": 0,
        "fork": 2
    }
    
    print("Initializing GaussianLoDModel (Dummy)...")
    gaussians = GaussianLoDModel(**model_kwargs)

    # 4. 加载场景 (Camera Loading)
    print("Initializing Scene...")
    # load_iteration=None 意味着从零开始，Scene 会读取 colmap 数据
    scene = Scene(scene_args, gaussians, load_iteration=None, shuffle=False)
    
    train_cameras = scene.getTrainCameras()
    print(f"Loaded {len(train_cameras)} cameras.")

    # 手动分类相机 (根据文件名)
    aerial_cams = []
    street_cams = []
    for cam in train_cameras:
        img_name = cam.image_name.lower()
        if "street" in img_name:
            street_cams.append(cam)
        else:
            aerial_cams.append(cam)
            
    print(f"Classified: {len(aerial_cams)} Aerial, {len(street_cams)} Street cameras.")

    if len(aerial_cams) == 0:
        print("Error: No aerial cameras found. Please check dataset filenames.")
        return

    # ================= 核心算法：虚拟致密点云 =================
    
    # 5. 生成虚拟点云
    extent = scene.cameras_extent
    print(f"Scene Extent (Radius): {extent}")
    
    # 生成 100万 个点 (显存如果不够可以改小，比如 500_000)
    num_points = 1_000_000
    print(f"Generating {num_points} virtual points in scene volume...")
    
    # 在 [-extent, extent] 范围内均匀采样
    virtual_points = (torch.rand((num_points, 3), device="cuda") * 2 - 1) * extent * 1.5
    
    # 6. 计算“航拍可视区域” (Volume Carving)
    print("Step 1: Computing Aerial Visibility (Carving)...")
    
    aerial_mask_accumulator = torch.zeros(num_points, dtype=torch.bool, device="cuda")
    
    # 分批处理防止显存爆炸
    batch_size = 50 
    for i in tqdm(range(0, len(aerial_cams), batch_size), desc="Aerial Carving"):
        batch_cams = aerial_cams[i:i+batch_size]
        for cam in batch_cams:
            proj = get_projection_matrix(cam)
            mask = FrustumCuller.filter_anchors_by_frustum(virtual_points, proj, margin=0.1)
            aerial_mask_accumulator |= mask 
            
    visible_points_aerial = virtual_points[aerial_mask_accumulator]
    print(f"Points visible to Aerial views: {visible_points_aerial.shape[0]} / {num_points}")
    
    if visible_points_aerial.shape[0] == 0:
        print("Error: No points visible to aerial cameras. Check 'extent' or camera poses.")
        return

    # 7. 遍历街景相机，生成 Mask
    output_dir = os.path.join(args.source_path, "geometric_masks")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Step 2: Generating Masks -> {output_dir}")
    
    # 只需要对街景生成 Mask
    for s_cam in tqdm(street_cams, desc="Generating Masks"):
        
        proj_s = get_projection_matrix(s_cam)
        # 检查之前的“航拍可视点”里，有哪些在当前街景视野内
        mask_s = FrustumCuller.filter_anchors_by_frustum(visible_points_aerial, proj_s, margin=0.1)
        
        overlap_points = visible_points_aerial[mask_s]
        
        # 初始化全黑 Mask
        mask_img = np.zeros((s_cam.image_height, s_cam.image_width), dtype=np.uint8)
        
        if overlap_points.shape[0] > 50: # 只有当重叠点足够多时才画
            # 投影到 2D
            uv = project_to_image(overlap_points, s_cam)
            uv_cpu = uv.cpu().numpy().astype(int)
            
            # 过滤画布外的点
            H, W = s_cam.image_height, s_cam.image_width
            valid_uv = (uv_cpu[:, 0] >= 0) & (uv_cpu[:, 0] < W) & (uv_cpu[:, 1] >= 0) & (uv_cpu[:, 1] < H)
            uv_clean = uv_cpu[valid_uv]
            
            if len(uv_clean) > 0:
                # 填色 (白色 255)
                mask_img[uv_clean[:, 1], uv_clean[:, 0]] = 255
                
                # 膨胀 (Dilation) 把点连成片
                # 根据分辨率自适应核大小
                kernel_size = int(max(W, H) * 0.02) 
                if kernel_size % 2 == 0: kernel_size += 1
                if kernel_size < 3: kernel_size = 3
                
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask_img = cv2.dilate(mask_img, kernel, iterations=1)

        # 保存
        cv2.imwrite(os.path.join(output_dir, f"{s_cam.image_name}.png"), mask_img)

    print("Done! Masks generated.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Geometric Mask Generator")
    parser.add_argument('--source_path', '-s', required=True, type=str, help="path to dataset (e.g., data/fusion/train)")
    parser.add_argument('--model_path', '-m', default="", type=str)
    parser.add_argument('--images', type=str, default="images")
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--white_background', action='store_true')
    
    args = parser.parse_args()
    
    with torch.no_grad():
        main(args)