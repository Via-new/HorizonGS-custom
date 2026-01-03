import os
import torch
import numpy as np
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# ================= 配置区域 =================
DATA_ROOT = "data/fusion/train/images"
STREET_DIR = os.path.join(DATA_ROOT, "street/train")  # 你的街景文件夹路径
AERIAL_DIR = os.path.join(DATA_ROOT, "aerial")        # 你的航拍文件夹路径
RESULT_FILE = "best_pairs_lightglue.txt"              # 结果保存路径

# 显卡设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def get_image_paths(directory):
    """获取目录下所有图片路径"""
    exts = ['.png', '.jpg', '.jpeg']
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if any(f.lower().endswith(ext) for ext in exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def main():
    # 1. 初始化模型
    print("正在加载 LightGlue 模型...")
    # max_num_keypoints 可以设大一点以提高召回率，但会变慢
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    # 2. 获取文件列表
    street_paths = get_image_paths(STREET_DIR)
    aerial_paths = get_image_paths(AERIAL_DIR)
    
    print(f"找到街景图: {len(street_paths)} 张")
    print(f"找到航拍图: {len(aerial_paths)} 张")
    print(f"预计总匹配次数: {len(street_paths) * len(aerial_paths)}")

    # 3. 预提取所有航拍图特征 (缓存到 CPU 以防显存爆炸)
    aerial_feats_cache = []
    print("\n[阶段 1/2] 预提取航拍图特征...")
    
    with torch.no_grad():
        for a_path in tqdm(aerial_paths, desc="Extracting Aerial"):
            try:
                img = load_image(a_path).to(device)
                feats = extractor.extract(img)
                # 移除 batch 维度并转存到 CPU 节省显存
                feats = {k: v.cpu() for k, v in feats.items()} 
                aerial_feats_cache.append({'path': a_path, 'feats': feats})
            except Exception as e:
                print(f"Error loading {a_path}: {e}")

    # 4. 暴力匹配循环
    print("\n[阶段 2/2] 街景-航拍 全量暴力匹配...")
    results = [] # 存储格式: (street_path, best_aerial_path, match_count)

    with torch.no_grad():
        for s_path in tqdm(street_paths, desc="Processing Streets"):
            try:
                # 提取当前街景特征
                s_img = load_image(s_path).to(device)
                s_feats = extractor.extract(s_img)
                
                best_match_count = -1
                best_aerial_idx = -1
                
                # 与每一个航拍特征进行匹配
                for i, a_data in enumerate(aerial_feats_cache):
                    # 将缓存的航拍特征挪回 GPU
                    a_feats_gpu = {k: v.to(device) for k, v in a_data['feats'].items()}
                    
                    # 执行匹配
                    matches01 = matcher({'image0': s_feats, 'image1': a_feats_gpu})
                    
                    # 统计匹配点数量
                    # matches key通常包含 'matches' 或 'matches0'，取长度
                    count = len(matches01['matches'][0])
                    
                    if count > best_match_count:
                        best_match_count = count
                        best_aerial_idx = i
                
                # 记录该街景的最佳配对
                if best_aerial_idx != -1:
                    best_aerial_path = aerial_feats_cache[best_aerial_idx]['path']
                    results.append(f"{s_path},{best_aerial_path},{best_match_count}")
                    
                    # 实时打印比较好的结果，让你心里有底
                    # if best_match_count > 100:
                    #     tqdm.write(f"  High match: {os.path.basename(s_path)} -> {os.path.basename(best_aerial_path)} ({best_match_count})")

            except Exception as e:
                print(f"Error processing {s_path}: {e}")

    # 5. 保存结果到文件
    with open(RESULT_FILE, "w") as f:
        f.write("street_path,best_aerial_path,match_count\n")
        for line in results:
            f.write(line + "\n")
            
    print(f"\n计算完成！结果已保存至: {RESULT_FILE}")

if __name__ == "__main__":
    main()