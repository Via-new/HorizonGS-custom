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
    print(f"\n[1/5] Loading Model from: {args.checkpoint}")
    
    # 1. 加载模型结构
    modules = __import__('scene')
    model_config = dataset.model_config
    GaussModel = getattr(modules, model_config['name'])
    gaussians = GaussModel(**model_config['kwargs'])
    
    # 构造 Dummy PointCloud 以骗过加载器
    spatial_lr_scale = 10.0 
    dummy_pcd = BasicPointCloud(points=np.zeros((1, 3)), colors=np.zeros((1, 3)), normals=np.zeros((1, 3)))
    
    # 加载权重
    # 注意：这里我们不需要训练，只是处理参数，所以不需要 optim
    try:
        gaussians.create_from_pretrained(dummy_pcd, spatial_lr_scale, args.checkpoint, None)
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return

    # 2. 加载冲突索引
    if not os.path.exists(args.conflict_indices):
        print(f"[Error] Indices file not found: {args.conflict_indices}")
        return
        
    conflict_indices = torch.load(args.conflict_indices).to("cuda")
    n_conflict = conflict_indices.shape[0]
    n_original = gaussians.get_anchor.shape[0]
    
    print(f"  -> Original Anchors: {n_original}")
    print(f"  -> Conflict Anchors to Split: {n_conflict}")

    # ================= [核心手术] =================
    print("\n[2/5] Performing Split Operation...")
    
    with torch.no_grad():
        # 1. 复制所有属性
        # 注意：HorizonGS 的参数通常是 _anchor, _offset, _anchor_feat, _scaling, _rotation
        # 以及 _level, _extra_level
        
        new_anchor = gaussians._anchor[conflict_indices].clone()
        new_offset = gaussians._offset[conflict_indices].clone()
        new_feat = gaussians._anchor_feat[conflict_indices].clone()
        new_scaling = gaussians._scaling[conflict_indices].clone()
        new_rotation = gaussians._rotation[conflict_indices].clone()
        
        # 还要处理 level (新点继承原点的 level)
        new_level = gaussians._level[conflict_indices].clone()
        new_extra_level = gaussians._extra_level[conflict_indices].clone()

        # 2. 拼接到原参数后
        # 使用 torch.nn.Parameter 包装，使其成为可训练参数
        gaussians._anchor = torch.nn.Parameter(torch.cat([gaussians._anchor, new_anchor], dim=0))
        gaussians._offset = torch.nn.Parameter(torch.cat([gaussians._offset, new_offset], dim=0))
        gaussians._anchor_feat = torch.nn.Parameter(torch.cat([gaussians._anchor_feat, new_feat], dim=0))
        gaussians._scaling = torch.nn.Parameter(torch.cat([gaussians._scaling, new_scaling], dim=0))
        gaussians._rotation = torch.nn.Parameter(torch.cat([gaussians._rotation, new_rotation], dim=0))
        
        # 非 Parameter 的 buffer
        gaussians._level = torch.cat([gaussians._level, new_level], dim=0)
        gaussians._extra_level = torch.cat([gaussians._extra_level, new_extra_level], dim=0)
        
        # 更新 mask 属性 (Anchor mask 也需要延长，默认为 True)
        if hasattr(gaussians, "_anchor_mask"):
             gaussians._anchor_mask = torch.cat([gaussians._anchor_mask, gaussians._anchor_mask[conflict_indices]], dim=0)
        if hasattr(gaussians, "_gs_mask"):
             gaussians._gs_mask = torch.cat([gaussians._gs_mask, gaussians._gs_mask[conflict_indices]], dim=0)

        n_new_total = gaussians.get_anchor.shape[0]
        print(f"  -> Split Complete. New Total: {n_new_total} (Original {n_original} + Copied {n_conflict})")

        # ================= [生成 Mask] =================
        print("\n[3/5] Generating Source Mask...")
        # 2 = Shared (默认), 0 = Air, 1 = Street
        source_mask = torch.full((n_new_total,), 2, dtype=torch.uint8, device="cuda")
        
        # A. 原来的冲突点 -> 标记为 0 (Air Only)
        # 意味着：这些点只给航拍看（保留它的模糊/大方差特性）
        source_mask[conflict_indices] = 0
        
        # B. 新复制出来的点 -> 标记为 1 (Street Only)
        # 这些点追加在最后，索引范围是 [n_original, n_new_total)
        # 意味着：这些点只给街景看（街景 Loss 会重塑它们，把它压实或剔除）
        new_indices = torch.arange(n_original, n_new_total, device="cuda")
        source_mask[new_indices] = 1
        
        # 统计
        n_air = (source_mask == 0).sum().item()
        n_street = (source_mask == 1).sum().item()
        n_shared = (source_mask == 2).sum().item()
        print(f"  -> Mask Stats: Air-Only={n_air}, Street-Only={n_street}, Shared={n_shared}")

    # 3. 保存结果
    output_dir = args.output_dir
    print(f"\n[4/5] Saving New Model to: {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 3.1 保存点云 PLY
    gaussians.save_ply(os.path.join(output_dir, "point_cloud.ply"))
    
    # 3.2 保存 MLP 权重 
    # (权重本身不需要变，因为我们没有修改网络结构，只是增加了输入点的数量)
    gaussians.save_mlp_checkpoints(output_dir)
    
    # 3.3 [关键] 保存 Source Mask
    mask_path = os.path.join(output_dir, "anchor_source_mask.pt")
    torch.save(source_mask, mask_path)
    print(f"  [CRITICAL] Saved Source Mask to: {mask_path}")

    # 3.4 复制辅助文件
    try:
        if os.path.exists(os.path.join(args.checkpoint, "cfg_args")):
            shutil.copy(os.path.join(args.checkpoint, "cfg_args"), os.path.join(output_dir, "cfg_args"))
    except:
        pass

    print("\n[5/5] Done. You can now start Fine Training with this model.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--conflict_indices', type=str, required=True, help="Path to conflict_indices.pt")
    parser.add_argument('--output_dir', type=str, required=True, help="Output path for the split model")
    parser.add_argument("--gpu", type=str, default='0')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        
    lp.model_path = os.path.dirname(os.path.dirname(args.checkpoint))
    split_anchors(lp, op, pp, args)