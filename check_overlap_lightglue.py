import torch
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd
import matplotlib.pyplot as plt
import matplotlib
import os

# 设置无界面绘图后端，防止服务器报错
matplotlib.use('Agg')

# --- 配置部分 ---
data_root = "data/fusion/train/images"  
img_path0 = os.path.join(data_root, "street/train/street_0286.png") 
img_path1 = os.path.join(data_root, "aerial/aerial_0384.png")       

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def analyze_pair(path0, path1):
    if not os.path.exists(path0) or not os.path.exists(path1):
        print(f"错误: 找不到图片文件。\n检查路径:\n{path0}\n{path1}")
        return

    # 1. 加载模型
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    # 2. 加载图片
    image0 = load_image(path0).to(device)
    image1 = load_image(path1).to(device)
    
    # [调试信息] 打印图片形状，确认维度
    print(f"Image 0 shape: {image0.shape}") 
    print(f"Image 1 shape: {image1.shape}")

    # 3. 推理
    print("正在提取特征...")
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    
    print("正在匹配特征...")
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    
    # 移除 feature 的 batch 维度 (LightGlue 内部处理需要)
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # 4. 获取匹配结果
    matches = matches01['matches']
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]
    
    num_matches = len(matches)
    print(f"\n=== 分析结果 ===")
    print(f"街景: {os.path.basename(path0)}")
    print(f"航拍: {os.path.basename(path1)}")
    print(f"--------------------------------")
    print(f"LightGlue 找到的匹配点数量: {num_matches}")
    print(f"COLMAP 之前找到的共视点: ~57")
    if num_matches > 0:
        print(f"提升倍数: {num_matches / 57:.1f}x")

    # 5. 可视化 (鲁棒性修正版)
    # 目标: 获取 (H, W, 3) 的 Numpy 数组
    
    def tensor_to_plot_array(tensor):
        # 如果是 4D (1, 3, H, W)，去掉 batch 维度 -> (3, H, W)
        if tensor.dim() == 4:
            tensor = tensor[0]
        # 现在应该是 (3, H, W)，进行 permute -> (H, W, 3)
        return tensor.cpu().permute(1, 2, 0).numpy()

    image0_plot = tensor_to_plot_array(image0)
    image1_plot = tensor_to_plot_array(image1)
    
    # 绘图
    axes = viz2d.plot_images([image0_plot, image1_plot])
    viz2d.plot_matches(points0.cpu(), points1.cpu(), color="lime", lw=0.5)
    
    output_filename = "lightglue_overlap_check.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存为: {output_filename}")
    print("请下载并查看该图片。")

if __name__ == "__main__":
    analyze_pair(img_path0, img_path1)