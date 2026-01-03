import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
# 关键导入
from romatch import roma_outdoor

# 设置无界面绘图，防止服务器报错
matplotlib.use('Agg')

# ================= 配置区域 =================
DATA_ROOT = "data/fusion/train/images"
# 请确保这两个路径存在
img_path0 = os.path.join(DATA_ROOT, "street/train/street_0262.png") 
img_path1 = os.path.join(DATA_ROOT, "aerial/aerial_0165.png")
OUTPUT_NAME = "roma_overlap_check.png"

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def analyze_pair_roma(path0, path1):
    if not os.path.exists(path0) or not os.path.exists(path1):
        print(f"错误: 找不到图片文件。\n{path0}\n{path1}")
        return

    # 1. 加载 RoMa 模型
    print("正在加载 RoMa 模型...")
    # 使用 roma_outdoor 预训练权重，这是目前最强的户外匹配模型
    roma_model = roma_outdoor(device=device)

    # 2. 加载图片
    im0 = Image.open(path0).convert("RGB")
    im1 = Image.open(path1).convert("RGB")
    W0, H0 = im0.size
    W1, H1 = im1.size

    # 3. 推理 (匹配)
    print("正在进行稠密匹配 (这可能需要几秒钟)...")
    # match 返回 warp (流场) 和 certainty (置信度)
    warp, certainty = roma_model.match(path0, path1, device=device)

    # 4. 筛选匹配点
    # 阈值建议：0.5 对于跨视角比较安全，如果点太少可以降到 0.35
    threshold = 0.5 
    matches_mask = certainty > threshold
    
    # 统计有效匹配像素数
    total_valid_matches = matches_mask.sum().item()
    print(f"\n=== RoMa 分析结果 ===")
    print(f"RoMa 找到的高置信度匹配像素数: {total_valid_matches}")
    
    if total_valid_matches == 0:
        print("未找到有效匹配，请尝试降低阈值。")
        return

    # 5. 可视化
    print("正在生成可视化图片...")
    
    # 从有效匹配中随机采样 500 个点用于画线（画太多看不清）
    # valid_indices 是 (N, 2) 的张量，存储 [y, x] 坐标
    valid_indices = torch.nonzero(matches_mask.squeeze()) 
    
    num_vis = 500
    if len(valid_indices) > num_vis:
        # 随机打乱取前 500 个
        perm = torch.randperm(len(valid_indices))[:num_vis]
        sample_indices = valid_indices[perm]
    else:
        sample_indices = valid_indices

    # 获取 Img A 的坐标 (x, y)
    # valid_indices 是 [y, x]，所以需要反转一下
    kptsA = sample_indices[:, [1, 0]].cpu().numpy() 
    
    # 获取 Img B 的坐标
    # warp 的形状是 (B, H, W, 4)，最后维度包含 [x_A, y_A, x_B, y_B] (归一化坐标 -1~1)
    # 我们需要取出对应的 B 图坐标并反归一化
    warp_sample = warp[0, sample_indices[:, 0], sample_indices[:, 1]] # (N, 4)
    
    # RoMa 的 warp 输出最后两维通常是 Target 图的坐标
    # 注意：RoMa match API 返回的 warp 具体格式有时随版本变化
    # 我们改用更稳健的 find_mutual 方式或者直接转换
    # 这里直接利用 warp 的后两位 (x_B, y_B) 进行反归一化
    grid_b = warp_sample[:, 2:].cpu().numpy() # (-1, 1)
    kptsB = (grid_b + 1) * 0.5 # (0, 1)
    kptsB[:, 0] *= W1
    kptsB[:, 1] *= H1

    # --- 开始绘图 ---
    fig = plt.figure(figsize=(12, 6))
    
    # 拼接图片：为了美观，把两张图高度缩放到一致
    target_h = max(H0, H1)
    scale0 = target_h / H0
    scale1 = target_h / H1
    
    im0_r = im0.resize((int(W0 * scale0), target_h))
    im1_r = im1.resize((int(W1 * scale1), target_h))
    
    # 拼成大图
    concat_img = np.concatenate([np.array(im0_r), np.array(im1_r)], axis=1)
    plt.imshow(concat_img)
    plt.axis('off')
    
    # 绘制连线
    offset_x = im0_r.size[0] # 第二张图的起始 X 坐标
    
    for i in range(len(kptsA)):
        # 坐标也要跟随缩放
        pt0 = kptsA[i] * scale0
        pt1 = kptsB[i] * scale1
        pt1[0] += offset_x 
        
        # 画线
        plt.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], color='lime', linewidth=0.3, alpha=0.7)
        # 画点
        plt.scatter(pt0[0], pt0[1], s=3, c='red', edgecolors='none')
        plt.scatter(pt1[0], pt1[1], s=3, c='red', edgecolors='none')

    plt.title(f"RoMa Dense Matching (Top {num_vis} samples from {total_valid_matches})")
    plt.tight_layout()
    
    # 保存
    plt.savefig(OUTPUT_NAME, dpi=150, bbox_inches='tight')
    plt.close('all')
    
    print(f"可视化结果已保存至: {OUTPUT_NAME}")
    print(">>> 请查看图片，重点检查：路面是否有密集的连线？ <<<")

if __name__ == "__main__":
    analyze_pair_roma(img_path0, img_path1)