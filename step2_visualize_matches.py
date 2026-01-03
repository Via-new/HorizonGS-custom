import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd

# 设置无界面绘图
matplotlib.use('Agg')

# ================= 配置区域 =================
INPUT_FILE = "best_pairs_lightglue.txt"  # 第一步生成的文件
OUTPUT_DIR = "output_lightglue_vis2"      # 图片保存文件夹

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_plot_array(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    return tensor.cpu().permute(1, 2, 0).numpy()

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}，请先运行第一步的脚本。")
        return

    # 1. 加载模型
    print("正在加载 LightGlue 模型用于可视化...")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    # 2. 读取配对列表
    pairs_to_process = []
    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()[1:] # 跳过表头
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 3:
                pairs_to_process.append(parts)

    print(f"准备生成 {len(pairs_to_process)} 张可视化图片...")

    # 3. 循环生成图片
    for s_path, a_path, count in tqdm(pairs_to_process, desc="Visualizing"):
        try:
            # 加载图片
            image0 = load_image(s_path).to(device)
            image1 = load_image(a_path).to(device)

            # 推理
            with torch.no_grad():
                feats0 = extractor.extract(image0)
                feats1 = extractor.extract(image1)
                matches01 = matcher({'image0': feats0, 'image1': feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            # 获取数据
            matches = matches01['matches']
            kpts0 = feats0['keypoints'][matches[..., 0]]
            kpts1 = feats1['keypoints'][matches[..., 1]]
            
            # 绘图数据准备
            image0_arr = tensor_to_plot_array(image0)
            image1_arr = tensor_to_plot_array(image1)
            
            # 【修改点 1】直接调用 plot_images，不要在外面包 plt.figure
            # 这会自动创建一个新的 figure
            axes = viz2d.plot_images([image0_arr, image1_arr])
            viz2d.plot_matches(kpts0.cpu(), kpts1.cpu(), color="lime", lw=0.5)
            
            # 文件名
            s_name = os.path.splitext(os.path.basename(s_path))[0]
            a_name = os.path.splitext(os.path.basename(a_path))[0]
            save_path = os.path.join(OUTPUT_DIR, f"{s_name}_vs_{a_name}_cnt{count}.png")
            
            plt.title(f"Matches: {len(matches)}")
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            
            # 【修改点 2】强制关闭所有画布，彻底释放内存
            plt.close('all')

        except Exception as e:
            print(f"Error visualizing {s_path}: {e}")
            # 出错时也要尝试清理内存
            plt.close('all')

    print(f"\n全部可视化完成！请查看文件夹: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()