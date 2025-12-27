#
# Split Anchors V2.5 (Fix Directory/File Path Logic & Add Debug Info)
#
import torch
import os
import numpy as np
from argparse import ArgumentParser
import yaml
import shutil
from scene import Scene
from utils.general_utils import parse_cfg
from utils.graphics_utils import BasicPointCloud
import logging

def split_anchors(dataset, opt, pipe, args):
    logger = logging.getLogger()
    
    # 1. 加载模型
    print(">>> Loading Model...")
    modules = __import__('scene')
    model_config = dataset.model_config
    GaussModel = getattr(modules, model_config['name'])
    gaussians = GaussModel(**model_config['kwargs'])
    
    spatial_lr_scale = 10.0 
    dummy_pcd = BasicPointCloud(points=np.zeros((1, 3)), colors=np.zeros((1, 3)), normals=np.zeros((1, 3)))
    gaussians.create_from_pretrained(dummy_pcd, spatial_lr_scale, args.checkpoint, logger)

    # 2. 加载冲突索引
    # 智能查找路径：尝试几个常见位置
    possible_paths = [
        args.conflict_file,
        "debug_anchor_conflict/conflict_indices.pt",
        "../debug_anchor_conflict/conflict_indices.pt",
    ]
    # 如果 args.checkpoint 是文件夹，去它父级的父级找 debug_anchor_conflict
    if os.path.isdir(args.checkpoint):
         possible_paths.append(os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "debug_anchor_conflict/conflict_indices.pt"))
    else:
         possible_paths.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.checkpoint))), "debug_anchor_conflict/conflict_indices.pt"))
    
    conflict_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            conflict_path = p
            break
    
    if conflict_path is None:
        print(f"[Error] Tried paths: {possible_paths}")
        raise FileNotFoundError("Cannot find conflict_indices.pt. Please check the path or run analyse first.")
    
    print(f">>> Loading Conflict Indices from: {conflict_path}")
    loaded_data = torch.load(conflict_path)
    
    if isinstance(loaded_data, dict):
        target_indices = loaded_data['indices'].to("cuda")
        print(f"    Format: V2 Dict. Found {target_indices.shape[0]} points.")
    else:
        target_indices = loaded_data.to("cuda")
        print(f"    Format: V1 Tensor. Found {target_indices.shape[0]} points.")

    if target_indices.shape[0] == 0:
        print("No conflict points found. Exiting.")
        return

    # 3. 执行分裂 (Cloning)
    print(">>> Splitting Anchors...")
    
    N_orig = gaussians.get_anchor.shape[0]
    
    with torch.no_grad():
        new_anchor = gaussians._anchor[target_indices].clone().detach()
        new_feat = gaussians._anchor_feat[target_indices].clone().detach()
        
        new_offset = gaussians._offset[target_indices].clone().detach() if hasattr(gaussians, "_offset") else None
        new_scaling = gaussians._scaling[target_indices].clone().detach() if hasattr(gaussians, "_scaling") else None
        new_rotation = gaussians._rotation[target_indices].clone().detach() if hasattr(gaussians, "_rotation") else None
        new_level = gaussians._level[target_indices].clone().detach() if hasattr(gaussians, "_level") else None
        new_extra_level = gaussians._extra_level[target_indices].clone().detach() if hasattr(gaussians, "_extra_level") else None
        
        # 拼接参数
        gaussians._anchor = torch.nn.Parameter(torch.cat([gaussians._anchor, new_anchor], dim=0))
        gaussians._anchor_feat = torch.nn.Parameter(torch.cat([gaussians._anchor_feat, new_feat], dim=0))
        
        if new_offset is not None:
            gaussians._offset = torch.nn.Parameter(torch.cat([gaussians._offset, new_offset], dim=0))
        if new_scaling is not None:
            gaussians._scaling = torch.nn.Parameter(torch.cat([gaussians._scaling, new_scaling], dim=0))
        if new_rotation is not None:
            gaussians._rotation = torch.nn.Parameter(torch.cat([gaussians._rotation, new_rotation], dim=0))
            
        if new_level is not None:
            if isinstance(gaussians._level, torch.nn.Parameter):
                gaussians._level = torch.nn.Parameter(torch.cat([gaussians._level, new_level], dim=0))
            else:
                gaussians._level = torch.cat([gaussians._level, new_level], dim=0)

        if new_extra_level is not None:
            if isinstance(gaussians._extra_level, torch.nn.Parameter):
                gaussians._extra_level = torch.nn.Parameter(torch.cat([gaussians._extra_level, new_extra_level], dim=0))
            else:
                gaussians._extra_level = torch.cat([gaussians._extra_level, new_extra_level], dim=0)

    N_new = gaussians.get_anchor.shape[0]
    print(f"    Original Anchors: {N_orig}")
    print(f"    New Anchors:      {N_new} (+{N_new - N_orig})")

    # 4. 生成 Mask
    print(">>> Creating Anchor Source Mask...")
    source_mask = torch.ones(N_new, dtype=torch.int8, device="cuda") * 2 
    source_mask[target_indices] = 0 
    source_mask[N_orig:] = 1 
    
    # 5. 保存结果
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(">>> Initializing Optimizer for Checkpointing...")
    gaussians.training_setup(opt)
    
    print(">>> Saving Checkpoint and Mask...")
    save_dict = gaussians.capture()
    torch.save((save_dict, 0), os.path.join(output_dir, "chkpnt.pth")) 
    torch.save(source_mask, os.path.join(output_dir, "anchor_source_mask.pt"))
    gaussians.save_ply(os.path.join(output_dir, "point_cloud.ply"))
    
    # --- [修正] 复制 MLP 权重文件逻辑 ---
    print(">>> Copying MLP checkpoints...")
    
    # 智能判断 source_dir
    if os.path.isdir(args.checkpoint):
        # 如果传入的是文件夹 (如 iteration_60000)，直接使用
        source_dir = args.checkpoint
    else:
        # 如果传入的是文件 (如 chkpnt.pth)，取其父目录
        source_dir = os.path.dirname(args.checkpoint)

    print(f"    [Debug] Search Directory: {source_dir}")

    mlp_files = ["opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"]
    
    success_count = 0
    for f in mlp_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(output_dir, f)
        
        # 打印正在尝试的完整路径，方便排查
        # print(f"    [Debug] Trying: {src}") 
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"    [OK] Copied {f}")
            success_count += 1
        else:
            print(f"    [WARN] Not found: {f}")
            print(f"           Expected at: {src}")
    
    if success_count < len(mlp_files):
        print("    [WARNING] Some MLP files were missing. This might cause training errors!")
    else:
        print("    [Success] All MLP checkpoints copied.")
    # ----------------------------------------

    try:
        cfg_args_path = os.path.join(os.path.dirname(args.checkpoint), "cfg_args")
        if os.path.exists(cfg_args_path):
            shutil.copy(cfg_args_path, os.path.join(output_dir, "cfg_args"))
    except:
        pass
    
    print(f"Done. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--conflict_file', type=str, default="")
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