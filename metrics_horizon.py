#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    
    if not os.path.exists(renders_dir) or not os.path.exists(gt_dir):
        return [], [], []

    # 强制排序，保证顺序一致
    fnames = sorted(os.listdir(renders_dir))
    
    for fname in fnames:
        if fname.endswith(".png") or fname.endswith(".jpg"):
            render = Image.open(os.path.join(renders_dir, fname))
            gt = Image.open(os.path.join(gt_dir, fname))
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)
    return renders, gts, image_names

def compute_metrics_for_list(renders, gts, desc="Evaluating"):
    """
    辅助函数：计算一组图片的指标
    """
    ssims = []
    psnrs = []
    lpipss = []
    
    if len(renders) == 0:
        return ssims, psnrs, lpipss

    for idx in tqdm(range(len(renders)), desc=desc):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
    
    return ssims, psnrs, lpipss

def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print(f"Scene: {scene_dir}")
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            # HorizonGS 可能会把结果放在 train 或 test 文件夹下
            # 根据你之前的截图，结果在 train 下 (因为 eval=False)
            # 这里做一个自动探测
            possible_dirs = ["train", "test"]
            target_base = None
            for d in possible_dirs:
                if os.path.exists(os.path.join(scene_dir, d)):
                    # 检查里面有没有 ours_xxxxx
                    sub = os.listdir(os.path.join(scene_dir, d))
                    if any("ours_" in s for s in sub):
                        target_base = Path(scene_dir) / d
                        break
            
            if target_base is None:
                print(f"Skipping {scene_dir}: Could not find 'train' or 'test' subdirectory with results.")
                continue

            print(f"Found results in: {target_base}")

            for method in os.listdir(target_base):
                if not method.startswith("ours_"): 
                    continue
                    
                print(f"Method: {method}")

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = target_base / method
                
                # --- 1. 准备数据容器 ---
                metrics_aerial = {"ssim": [], "psnr": [], "lpips": []}
                metrics_street = {"ssim": [], "psnr": [], "lpips": []}
                
                # --- 2. 处理 Aerial (如果存在) ---
                aerial_render_dir = method_dir / "aerial" / "renders"
                aerial_gt_dir = method_dir / "aerial" / "gt"
                
                renders_a, gts_a, names_a = readImages(aerial_render_dir, aerial_gt_dir)
                if len(renders_a) > 0:
                    metrics_aerial["ssim"], metrics_aerial["psnr"], metrics_aerial["lpips"] = \
                        compute_metrics_for_list(renders_a, gts_a, desc="Evaluating Aerial")

                # --- 3. 处理 Street (如果存在) ---
                street_render_dir = method_dir / "street" / "renders"
                street_gt_dir = method_dir / "street" / "gt"
                
                renders_s, gts_s, names_s = readImages(street_render_dir, street_gt_dir)
                if len(renders_s) > 0:
                    metrics_street["ssim"], metrics_street["psnr"], metrics_street["lpips"] = \
                        compute_metrics_for_list(renders_s, gts_s, desc="Evaluating Street")

                # --- 4. 汇总 (All) ---
                all_ssim = metrics_aerial["ssim"] + metrics_street["ssim"]
                all_psnr = metrics_aerial["psnr"] + metrics_street["psnr"]
                all_lpips = metrics_aerial["lpips"] + metrics_street["lpips"]
                all_names = names_a + names_s

                # --- 5. 打印结果表 ---
                def get_mean(m_list):
                    if len(m_list) == 0: return 0.0
                    return torch.tensor(m_list).mean().item()

                print("\n  --- Evaluation Results ---")
                print("  {:<10} | {:<8} | {:<8} | {:<8} | {:<6}".format("Type", "SSIM", "PSNR", "LPIPS", "Count"))
                print("  " + "-"*52)
                
                # ALL
                print("  {:<10} | {:<8.5f} | {:<8.5f} | {:<8.5f} | {:<6}".format(
                    "ALL", get_mean(all_ssim), get_mean(all_psnr), get_mean(all_lpips), len(all_ssim)))
                
                # AERIAL
                if len(renders_a) > 0:
                    print("  {:<10} | {:<8.5f} | {:<8.5f} | {:<8.5f} | {:<6}".format(
                        "AERIAL", get_mean(metrics_aerial["ssim"]), get_mean(metrics_aerial["psnr"]), get_mean(metrics_aerial["lpips"]), len(renders_a)))
                
                # STREET
                if len(renders_s) > 0:
                    print("  {:<10} | {:<8.5f} | {:<8.5f} | {:<8.5f} | {:<6}".format(
                        "STREET", get_mean(metrics_street["ssim"]), get_mean(metrics_street["psnr"]), get_mean(metrics_street["lpips"]), len(renders_s)))
                print("")

                # --- 6. 保存 JSON ---
                full_dict[scene_dir][method].update({
                    "SSIM": get_mean(all_ssim),
                    "PSNR": get_mean(all_psnr),
                    "LPIPS": get_mean(all_lpips),
                    "AERIAL_PSNR": get_mean(metrics_aerial["psnr"]),
                    "AERIAL_SSIM": get_mean(metrics_aerial["ssim"]),
                    "STREET_PSNR": get_mean(metrics_street["psnr"]),
                    "STREET_SSIM": get_mean(metrics_street["ssim"]),
                })
                
                # Per View JSON (合并两个列表)
                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: val for val, name in zip(torch.tensor(all_ssim).tolist(), all_names)},
                    "PSNR": {name: val for val, name in zip(torch.tensor(all_psnr).tolist(), all_names)},
                    "LPIPS": {name: val for val, name in zip(torch.tensor(all_lpips).tolist(), all_names)}
                })

            with open(os.path.join(scene_dir, "results.json"), 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(os.path.join(scene_dir, "per_view.json"), 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        
        except Exception as e:
            print(f"Error evaluating {scene_dir}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Evaluation script")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)