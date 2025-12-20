import os
import torch
import numpy as np
import subprocess
from argparse import ArgumentParser
import yaml
import sys
from tqdm import tqdm
import math

# 引入项目依赖
from utils.general_utils import safe_state, parse_cfg
from scene import Scene
import scene as scene_modules

class LODAnalyzer:
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.standard_dist = gaussians.standard_dist
        self.fork = float(gaussians.fork)
        self.street_levels = gaussians.street_levels
        
    def get_active_levels(self, cam, anchors, levels, extra_levels):
        """
        模拟 HorizonGS 的筛选逻辑，返回该相机视角下被激活的锚点的 Level 分布。
        """
        # 1. 视锥剔除 (Frustum Culling)
        p_hom = torch.cat([anchors, torch.ones_like(anchors[:, :1])], dim=1)
        p_clip = p_hom @ cam.full_proj_transform
        
        # 简单宽松的视锥判断
        valid_z = p_clip[:, 3] > 0.001
        denom = p_clip[:, 3] + 1e-6
        p_ndc_x = p_clip[:, 0] / denom
        p_ndc_y = p_clip[:, 1] / denom
        
        margin = 0.5
        mask_frustum = valid_z & (p_ndc_x > -1.0-margin) & (p_ndc_x < 1.0+margin) & \
                       (p_ndc_y > -1.0-margin) & (p_ndc_y < 1.0+margin)

        # 2. LOD 剔除 (LOD Culling)
        cam_center = cam.camera_center
        dist = torch.sqrt(torch.sum((anchors - cam_center)**2, dim=1)) * cam.resolution_scale
        dist = torch.clamp(dist, min=1e-6)
        
        # 计算当前视角允许的最大层级 (Int Level)
        pred_level = torch.log2(self.standard_dist / dist) / math.log2(self.fork) + extra_levels
        
        if hasattr(self.gaussians, 'map_to_int_level'):
            limit_level = self.gaussians.map_to_int_level(pred_level, self.street_levels - 1)
        else:
            limit_level = torch.floor(pred_level)
            limit_level = torch.clamp(limit_level, max=self.street_levels - 1)
            
        # 锚点自身的 level <= 限制 level 时才可见
        # 注意：这里我们不仅要 mask，还要把“当前这个点是以什么 level 被看到的”统计出来
        # 其实锚点的 level 是固定的，我们只需要筛选出 visible 的点，然后统计它们的 level
        mask_lod = (levels.squeeze() <= limit_level)
        
        # 3. 最终 Mask
        final_mask = mask_frustum & mask_lod
        
        # 4. 提取激活点的 Level
        active_levels = levels[final_mask].squeeze().cpu().numpy()
        
        return active_levels

def print_distribution(levels_array, title):
    if len(levels_array) == 0:
        print(f"\n--- {title} ---")
        print("  No anchors active.")
        return

    unique, counts = np.unique(levels_array, return_counts=True)
    total = sum(counts)
    
    print(f"\n--- {title} ---")
    print(f"  Total Active Anchors: {total}")
    for u, c in zip(unique, counts):
        print(f"  Level {int(u)}: {c:8d} ({c/total*100:.2f}%)")

def main(args):
    # 手动指定要分析的相机
    TARGET_STREET = "street_0262" 
    TARGET_AERIAL = "aerial_0165" 
    
    # GPU Setup
    try:
        cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
        os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
    except: pass
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
    lp.model_path = args.model_path
    
    print("Loading Gaussians...")
    model_config = lp.model_config
    gaussians = getattr(scene_modules, model_config['name'])(**model_config['kwargs'])
    scene = Scene(lp, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # Fix types for searching
    for cam in scene.getTrainCameras():
        if "street" in cam.image_name.lower(): cam.image_type = "street"
        else: cam.image_type = "aerial"
    
    # 获取数据
    anchors = gaussians.get_anchor.detach()
    levels = gaussians.get_level.detach() # [N, 1]
    extra_levels = gaussians.get_extra_level.detach()
    
    # 1. 全局统计 (Global Stats)
    all_levels_np = levels.squeeze().cpu().numpy()
    print_distribution(all_levels_np, "GLOBAL MODEL STATS (All Anchors)")

    # 2. 视角分析
    analyzer = LODAnalyzer(gaussians)
    
    # 查找相机
    street_cam = next((c for c in scene.getTrainCameras() if c.image_name == TARGET_STREET), None)
    aerial_cam = next((c for c in scene.getTrainCameras() if c.image_name == TARGET_AERIAL), None)
    
    if street_cam:
        active_levels_s = analyzer.get_active_levels(street_cam, anchors, levels, extra_levels)
        print_distribution(active_levels_s, f"STREET VIEW ({TARGET_STREET})")
    else:
        print(f"\nStreet cam {TARGET_STREET} not found.")

    if aerial_cam:
        active_levels_a = analyzer.get_active_levels(aerial_cam, anchors, levels, extra_levels)
        print_distribution(active_levels_a, f"AERIAL VIEW ({TARGET_AERIAL})")
    else:
        print(f"\nAerial cam {TARGET_AERIAL} not found.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--model_path', type=str, required=True, help='Path to output folder')
    parser.add_argument('--iteration', type=int, default=-1, help='Iteration to load')
    args = parser.parse_args()
    
    main(args)