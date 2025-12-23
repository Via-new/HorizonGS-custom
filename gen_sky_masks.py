import os
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm
import argparse

# === 配置区域 ===
# 默认使用的模型 (ADE20K 数据集训练，包含 'sky' 类别，ID通常为 2)
MODEL_NAME = "nvidia/segformer-b3-finetuned-ade-512-512" 

def generate_masks(source_path, batch_size=1, device='cuda'):
    # 1. 路径设置
    images_root = os.path.join(source_path, "images")
    masks_root = os.path.join(source_path, "masks")
    
    if not os.path.exists(images_root):
        print(f"Error: Images directory not found: {images_root}")
        return

    print(f"Input Root: {images_root}")
    print(f"Output Root: {masks_root}")

    # 2. 加载模型
    print(f"Loading SegFormer model: {MODEL_NAME}...")
    try:
        processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"模型加载失败，请检查网络或配置: {e}")
        return

    # ADE20K 数据集中，'sky' 的类别 ID 是 2
    SKY_LABEL_ID = 2 

    # 3. 递归遍历所有子文件夹寻找图片 (关键修改)
    image_tasks = []
    print("Scanning for images...")
    for root, dirs, files in os.walk(images_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 获取绝对路径
                abs_path = os.path.join(root, file)
                # 获取相对于 images 文件夹的路径 (例如: aerial/001.png 或 street/train/001.png)
                rel_path = os.path.relpath(abs_path, images_root)
                image_tasks.append((abs_path, rel_path))

    print(f"Found {len(image_tasks)} images.")

    if len(image_tasks) == 0:
        print("未找到任何图片，请检查路径结构。")
        return

    # 4. 批量处理
    for abs_path, rel_path in tqdm(image_tasks, desc="Generating Masks"):
        try:
            image = Image.open(abs_path).convert("RGB")
            
            # 预处理
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                outputs = model(**inputs)
                
            # 后处理：上采样回原图尺寸
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1], # (H, W)
                mode="bilinear",
                align_corners=False,
            )

            # 获取预测类别 ID
            pred_seg = upsampled_logits.argmax(dim=1)[0] # [H, W]

            # === 生成 Mask 逻辑 ===
            # 1 (白色) = 有效区域/前景，0 (黑色) = 无效区域/天空
            mask_tensor = torch.ones_like(pred_seg, dtype=torch.uint8)
            mask_tensor[pred_seg == SKY_LABEL_ID] = 0 
            
            # 转换为 PIL 图片保存
            mask_img = Image.fromarray(mask_tensor.cpu().numpy() * 255) # 0->0, 1->255
            
            # === 关键修改：保持目录结构保存 ===
            # 计算 mask 的保存路径，保持 rel_path 目录结构
            # 例如: masks/street/train/street_0000.png
            save_path = os.path.join(masks_root, rel_path)
            
            # 强制保存为 .png 格式 (mask存成jpg会有压缩噪点，影响边缘)
            save_path = os.path.splitext(save_path)[0] + ".png"
            
            # 确保父目录存在 (例如自动创建 masks/street/train/)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            mask_img.save(save_path)

        except Exception as e:
            print(f"Failed to process {rel_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help="Path to the dataset root (containing images/)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    generate_masks(args.source_path, device=args.device)