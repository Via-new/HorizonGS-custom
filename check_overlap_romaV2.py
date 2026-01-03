import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from romatch import roma_outdoor
import cv2

matplotlib.use('Agg')

# ================= 配置区域 =================
DATA_ROOT = "data/fusion/train/images"
img_path0 = os.path.join(DATA_ROOT, "street/train/street_0262.png") 
img_path1 = os.path.join(DATA_ROOT, "aerial/aerial_0165.png")
OUTPUT_NAME = "roma_overlap_final.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def analyze_pair_roma(path0, path1):
    if not os.path.exists(path0) or not os.path.exists(path1):
        print(f"错误: 找不到图片文件。")
        return

    # 1. 加载模型
    print("正在加载 RoMa 模型...")
    roma_model = roma_outdoor(device=device)

    # 2. 加载图片
    im0 = Image.open(path0).convert("RGB")
    im1 = Image.open(path1).convert("RGB")
    W0, H0 = im0.size
    W1, H1 = im1.size
    print(f"原始尺寸: Street({W0}x{H0}), Aerial({W1}x{H1})")

    # 3. 推理
    print("正在进行稠密匹配...")
    # 强制指定 upscale 分辨率，确保精度
    warp, certainty = roma_model.match(path0, path1, device=device)
    
    # 获取 RoMa 内部特征图的尺寸
    H_roma, W_roma = certainty.shape[-2:]
    print(f"RoMa 内部处理分辨率: {W_roma}x{H_roma}")

    # 4. 筛选匹配点
    threshold = 0.5
    matches_mask = certainty > threshold
    
    valid_indices = torch.nonzero(matches_mask.squeeze()) 
    total_raw_matches = len(valid_indices)
    print(f"RoMa 原始高置信度点数: {total_raw_matches}")
    
    if total_raw_matches < 8:
        print("点数太少，无法计算。")
        return

    # ================= 坐标修正 (核心修改) =================
    
    # A. 获取 Img A (Street) 的坐标
    # valid_indices 是 [y, x] 格式，基于 RoMa 分辨率
    kptsA_roma = valid_indices[:, [1, 0]].float() # (x, y)
    # 归一化到 0-1
    kptsA_norm = kptsA_roma / torch.tensor([W_roma, H_roma], device=device)
    # 映射回原始图片尺寸
    kptsA_orig = kptsA_norm * torch.tensor([W0, H0], device=device)
    kptsA_np = kptsA_orig.cpu().numpy()

    # B. 获取 Img B (Aerial) 的坐标
    # warp 的最后两维是归一化坐标 (-1 到 1)
    warp_sample = warp[0, valid_indices[:, 0], valid_indices[:, 1]] # [x_s, y_s, x_t, y_t]
    kptsB_norm = (warp_sample[:, 2:] + 1) * 0.5 # 归一化到 0-1
    # 映射回原始图片尺寸
    kptsB_orig = kptsB_norm * torch.tensor([W1, H1], device=device)
    kptsB_np = kptsB_orig.cpu().numpy()

    # ================= 几何验证 (RANSAC) =================
    print("正在进行几何验证 (RANSAC)...")
    # 使用原始图片坐标进行 RANSAC，这样才准确
    F, mask = cv2.findFundamentalMat(kptsA_np, kptsB_np, cv2.USAC_MAGSAC, 3.0, 0.999)
    
    if mask is None:
        print("RANSAC 失败。")
        return
        
    mask = mask.ravel().astype(bool)
    inliers_count = np.sum(mask)
    print(f"几何验证后剩余点数: {inliers_count} (剔除率: {1 - inliers_count/total_raw_matches:.1%})")
    
    # 提取内点
    ptsA_good = kptsA_np[mask]
    ptsB_good = kptsB_np[mask]
    
    # ================= 可视化 =================
    print("正在生成可视化...")
    
    # 随机采样显示
    num_vis = 300
    if len(ptsA_good) > num_vis:
        indices = np.random.choice(len(ptsA_good), num_vis, replace=False)
        ptsA_vis = ptsA_good[indices]
        ptsB_vis = ptsB_good[indices]
    else:
        ptsA_vis = ptsA_good
        ptsB_vis = ptsB_good

    fig = plt.figure(figsize=(12, 6))
    
    # 拼接图片
    target_h = max(H0, H1)
    scale0 = target_h / H0
    scale1 = target_h / H1
    
    im0_r = im0.resize((int(W0 * scale0), target_h))
    im1_r = im1.resize((int(W1 * scale1), target_h))
    concat_img = np.concatenate([np.array(im0_r), np.array(im1_r)], axis=1)
    
    plt.imshow(concat_img)
    plt.axis('off')
    
    offset_x = im0_r.size[0]
    
    for i in range(len(ptsA_vis)):
        # 绘图时，要把原始坐标乘以缩放比例
        pt0 = ptsA_vis[i] * scale0
        pt1 = ptsB_vis[i] * scale1
        pt1[0] += offset_x 
        
        plt.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], color='lime', linewidth=0.5, alpha=0.8)
        plt.scatter(pt0[0], pt0[1], s=5, c='red')
        plt.scatter(pt1[0], pt1[1], s=5, c='blue')

    plt.title(f"RoMa Corrected (Matches: {inliers_count})")
    plt.tight_layout()
    plt.savefig(OUTPUT_NAME, dpi=150, bbox_inches='tight')
    plt.close('all')
    
    print(f"结果已保存至: {OUTPUT_NAME}")

if __name__ == "__main__":
    analyze_pair_roma(img_path0, img_path1)