#
# HorizonGS Anchor Splitter (Scheme 1 Implementation - Optimizer Fix)
# 
# 修复点：在保存 checkpoint 前调用 training_setup，确保 optimizer 存在，
# 从而让 capture() 能成功生成 chkpnt.pth
#

import os
import torch
import numpy as np
from argparse import ArgumentParser
import yaml
import shutil
import sys

# 引入 HorizonGS 模块
from scene import Scene
from utils.general_utils import parse_cfg
from utils.graphics_utils import BasicPointCloud

def split_anchors(dataset, opt, pipe, args):
    print(f"\n[1/6] Loading Model from: {args.checkpoint}")
    
    modules = __import__('scene')
    model_config = dataset.model_config
    GaussModel = getattr(modules, model_config['name'])
    gaussians = GaussModel(**model_config['kwargs'])
    
    spatial_lr_scale = 10.0 
    dummy_pcd = BasicPointCloud(points=np.zeros((1, 3)), colors=np.zeros((1, 3)), normals=np.zeros((1, 3)))
    
    # 加载权重
    try:
        gaussians.create_from_pretrained(dummy_pcd, spatial_lr_scale, args.checkpoint, None)
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return

    # 加载冲突索引
    if not os.path.exists(args.conflict_indices):
        print(f"[Error] Indices file not found: {args.conflict_indices}")
        return
        
    conflict_indices = torch.load(args.conflict_indices).to("cuda")
    n_conflict = conflict_indices.shape[0]
    n_original = gaussians.get_anchor.shape[0]
    
    print(f"  -> Original Anchors: {n_original}")
    print(f"  -> Conflict Anchors to Split: {n_conflict}")

    # ================= [核心手术] =================
    print("\n[2/6] Performing Split Operation...")
    
    with torch.no_grad():
        new_anchor = gaussians._anchor[conflict_indices].clone()
        new_offset = gaussians._offset[conflict_indices].clone()
        new_feat = gaussians._anchor_feat[conflict_indices].clone()
        new_scaling = gaussians._scaling[conflict_indices].clone()
        new_rotation = gaussians._rotation[conflict_indices].clone()
        
        new_level = gaussians._level[conflict_indices].clone()
        new_extra_level = gaussians._extra_level[conflict_indices].clone()

        # 拼接参数
        gaussians._anchor = torch.nn.Parameter(torch.cat([gaussians._anchor, new_anchor], dim=0))
        gaussians._offset = torch.nn.Parameter(torch.cat([gaussians._offset, new_offset], dim=0))
        gaussians._anchor_feat = torch.nn.Parameter(torch.cat([gaussians._anchor_feat, new_feat], dim=0))
        gaussians._scaling = torch.nn.Parameter(torch.cat([gaussians._scaling, new_scaling], dim=0))
        gaussians._rotation = torch.nn.Parameter(torch.cat([gaussians._rotation, new_rotation], dim=0))
        
        gaussians._level = torch.cat([gaussians._level, new_level], dim=0)
        gaussians._extra_level = torch.cat([gaussians._extra_level, new_extra_level], dim=0)
        
        # 更新 Mask
        if hasattr(gaussians, "_anchor_mask"):
             gaussians._anchor_mask = torch.cat([gaussians._anchor_mask, gaussians._anchor_mask[conflict_indices]], dim=0)
        if hasattr(gaussians, "_gs_mask"):
             gaussians._gs_mask = torch.cat([gaussians._gs_mask, gaussians._gs_mask[conflict_indices]], dim=0)

        n_new_total = gaussians.get_anchor.shape[0]
        print(f"  -> Split Complete. New Total: {n_new_total}")

        # ================= [生成 Mask] =================
        print("\n[3/6] Generating Source Mask...")
        source_mask = torch.full((n_new_total,), 2, dtype=torch.uint8, device="cuda")
        source_mask[conflict_indices] = 0 # Original -> Air
        new_indices = torch.arange(n_original, n_new_total, device="cuda")
        source_mask[new_indices] = 1 # Copied -> Street
        
        n_air = (source_mask == 0).sum().item()
        n_street = (source_mask == 1).sum().item()
        n_shared = (source_mask == 2).sum().item()
        print(f"  -> Mask Stats: Air-Only={n_air}, Street-Only={n_street}, Shared={n_shared}")

    # ================= [关键修复] =================
    print("\n[4/6] Initializing Optimizer for Checkpointing...")
    # 必须调用这个，否则 gaussians.capture() 会因为没有 optimizer 而报错
    # 这会为新的参数（742096个点）创建优化器状态
    gaussians.training_setup(opt)
    
    # 3. 保存结果
    output_dir = args.output_dir
    print(f"\n[5/6] Saving New Model to: {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 3.1 保存点云 PLY
    gaussians.save_ply(os.path.join(output_dir, "point_cloud.ply"))
    
    # 3.2 保存 MLP 权重
    gaussians.save_mlp_checkpoints(output_dir)
    
    # 3.3 [关键] 保存 Source Mask
    mask_path = os.path.join(output_dir, "anchor_source_mask.pt")
    torch.save(source_mask, mask_path)
    print(f"  [CRITICAL] Saved Source Mask to: {mask_path}")

    # 3.4 [新增] 保存标准的 .pth 权重文件
    try:
        save_path = os.path.join(output_dir, "chkpnt.pth")
        # 这里的 0 表示 iteration 0，告诉 train.py 这是一个新的开始（或者你可以填 60000）
        torch.save((gaussians.capture(), 0), save_path)
        print(f"  [CRITICAL] Saved PyTorch Checkpoint to: {save_path}")
    except Exception as e:
        print(f"  [Error] Failed to save .pth checkpoint: {e}")
        # 打印详细堆栈以便调试
        import traceback
        traceback.print_exc()

    # 3.5 复制配置
    try:
        if os.path.exists(os.path.join(args.checkpoint, "cfg_args")):
            shutil.copy(os.path.join(args.checkpoint, "cfg_args"), os.path.join(output_dir, "cfg_args"))
    except:
        pass

    print("\n[6/6] Done. Now train_experiment1.py should find 'chkpnt.pth'!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--conflict_indices', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--gpu", type=str, default='0')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.output_dir = os.path.abspath(args.output_dir)
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        
    lp.model_path = os.path.dirname(os.path.dirname(args.checkpoint))
    split_anchors(lp, op, pp, args)