import open3d as o3d
import numpy as np
import os

# ================= 配置区域 =================
# 你的原始大文件路径
INPUT_PATH = "data/fusion/train/input_origin.ply" 
# 输出路径
OUTPUT_PATH = "data/fusion/train/input.ply"

# 【核心】强制限制点数在 20万 左右
TARGET_POINTS = 500000 
# ===========================================

print(f"正在读取 {INPUT_PATH} ...")
pcd = o3d.io.read_point_cloud(INPUT_PATH)
original_count = len(pcd.points)
print(f"原始点数: {original_count}")

if original_count > TARGET_POINTS:
    print(f"点数严重超标！正在从 {original_count} 降采样到 {TARGET_POINTS} ...")
    
    # 计算采样比例
    ratio = TARGET_POINTS / original_count
    
    # 使用随机降采样（Random Downsample），速度快且显存占用低
    # 对于 DAV2 生成的均匀点云，随机采样完全能保留结构
    downpcd = pcd.random_down_sample(ratio)
    
    print(f"降采样完成。")
else:
    print("点数在合理范围内，无需处理。")
    downpcd = pcd

final_count = len(downpcd.points)
print(f"最终点数: {final_count}")
print(f"HorizonGS 将生成约 {final_count * 10} 个初始高斯球 (在显存安全范围内)。")

print(f"保存到 {OUTPUT_PATH} ...")
o3d.io.write_point_cloud(OUTPUT_PATH, downpcd)
print("\nSuccess! 请执行以下命令替换原文件：")
print(f"mv {OUTPUT_PATH} {INPUT_PATH}")