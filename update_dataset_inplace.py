import os
import json
import shutil

# ================= 配置区域 =================
# 数据集根目录
dataset_root = "data/aerial_street_fusion/test" 

# 原始 transforms.json 的路径
json_relative_path = "images/pose/transforms_test.json"
json_path = os.path.join(dataset_root, json_relative_path)

# COLMAP 转换后的 images.txt 路径
colmap_images_txt = os.path.join(dataset_root, "sparse/0/text_format/images.txt")
# ===========================================

def update_dataset():
    # 1. 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
    if not os.path.exists(colmap_images_txt):
        print(f"Error: {colmap_images_txt} not found. Did you run colmap model_converter?")
        return

    # 备份原始文件
    if not os.path.exists(json_path + ".bak"):
        shutil.copy(json_path, json_path + ".bak")
        print(f"Backed up json to {json_path}.bak")
    
    # 2. 读取 transforms.json
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # 获取 json 文件所在的目录
    json_dir = os.path.dirname(json_path)
    
    # 建立映射表
    rename_map = {} 
    
    print("\n[Step 1] Renaming files and updating JSON...")
    
    renamed_count = 0
    
    for frame in json_data['frames']:
        # file_path 例如: "../street/train/0001.png"
        rel_path = frame['file_path']
        
        # 计算物理文件的绝对路径
        abs_old_path = os.path.normpath(os.path.join(json_dir, rel_path))
        
        if not os.path.exists(abs_old_path):
            # 检查是否已重命名
            parts = os.path.split(abs_old_path)
            if "aerial_" in parts[-1] or "street_" in parts[-1]:
                continue
            print(f"Warning: File not found: {abs_old_path}")
            continue

        # 解析路径结构
        parent_dir_path, filename = os.path.split(abs_old_path)
        
        # ================= [核心修正逻辑] =================
        # 根据路径判断类型，并设定正确的 COLMAP 相对路径
        
        # 将路径转为统一格式方便判断
        norm_parent_path = parent_dir_path.replace("\\", "/")
        
        prefix = ""
        colmap_subdir = ""
        
        # if "/street/train" in norm_parent_path or "\\street\\train" in norm_parent_path:
        #     # 街景特殊结构: images/street/train/xxx.jpg
        #     prefix = "street"
        #     colmap_subdir = "street/train"
        # elif "/aerial" in norm_parent_path or "\\aerial" in norm_parent_path:
        #     # 航拍结构: images/aerial/xxx.jpg
        #     prefix = "aerial"
        #     colmap_subdir = "aerial"
        # else:
        #     # 兜底：如果路径结构很奇怪，回退到使用父文件夹名
        #     print(f"Warning: Unusual path structure for {rel_path}, using parent dir name.")
        #     prefix = os.path.basename(parent_dir_path)
        #     colmap_subdir = prefix

        if "/street/test" in norm_parent_path or "\\street\\test" in norm_parent_path:
            # 街景特殊结构: images/street/train/xxx.jpg
            prefix = "street"
            colmap_subdir = "street/test"
        elif "/aerial" in norm_parent_path or "\\aerial" in norm_parent_path:
            # 航拍结构: images/aerial/xxx.jpg
            prefix = "aerial"
            colmap_subdir = "aerial"
        else:
            # 兜底：如果路径结构很奇怪，回退到使用父文件夹名
            print(f"Warning: Unusual path structure for {rel_path}, using parent dir name.")
            prefix = os.path.basename(parent_dir_path)
            colmap_subdir = prefix

        # ==================================================
        
        # 检查是否需要重命名
        if filename.startswith(prefix + "_"):
            continue 
            
        # 构造新文件名: street_0001.png
        new_filename = f"{prefix}_{filename}"
        abs_new_path = os.path.join(parent_dir_path, new_filename)
        
        # --- 执行物理重命名 ---
        try:
            os.rename(abs_old_path, abs_new_path)
            renamed_count += 1
        except OSError as e:
            print(f"Error renaming {filename}: {e}")
            continue

        # --- 更新 JSON 对象 ---
        frame['file_path'] = rel_path.replace(filename, new_filename)
        
        # --- 准备 COLMAP 映射 (修正版) ---
        # 现在的目标是写入: street/train/street_0001.jpg
        target_colmap_name = f"{colmap_subdir}/{new_filename}"
        
        # Key 1: 纯文件名 (如果images.txt里只写了 0001.jpg)
        rename_map[filename] = target_colmap_name
        
        # Key 2: 带目录的相对路径 (如果images.txt里写了 street/train/0001.jpg)
        rename_map[f"{colmap_subdir}/{filename}"] = target_colmap_name

    # 3. 保存更新后的 JSON
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Physical rename complete. {renamed_count} files renamed.")
    print("transforms.json updated.")

    # 4. 修改 images.txt
    print("\n[Step 2] Updating COLMAP images.txt...")
    
    with open(colmap_images_txt, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    txt_mod_count = 0
    
    for line in lines:
        if line.startswith("#") or len(line.strip()) == 0:
            new_lines.append(line)
            continue
            
        parts = line.strip().split()
        if len(parts) >= 10:
            current_name = parts[-1] 
            
            if current_name.lower().endswith(('jpg', 'png', 'jpeg')):
                # 优先检查完整匹配
                if current_name in rename_map:
                    parts[-1] = rename_map[current_name]
                    new_lines.append(" ".join(parts) + "\n")
                    txt_mod_count += 1
                else:
                    # 如果找不到完全匹配，尝试只匹配文件名
                    # 例如 images.txt 里是 "0001.jpg"，但 rename_map 只有 "street/train/0001.jpg"
                    base_name = os.path.basename(current_name)
                    if base_name in rename_map:
                        parts[-1] = rename_map[base_name]
                        new_lines.append(" ".join(parts) + "\n")
                        txt_mod_count += 1
                    else:
                        new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(colmap_images_txt, 'w') as f:
        f.writelines(new_lines)
        
    print(f"images.txt updated! {txt_mod_count} entries modified.")
    print("------------------------------------------------")
    print("SUCCESS! Please execute the following command manually:")
    print(f"/home/xjw/code/colmap/build/src/colmap/exe/colmap model_converter --input_path {os.path.dirname(colmap_images_txt)} --output_path {os.path.dirname(os.path.dirname(colmap_images_txt))} --output_type BIN")

if __name__ == "__main__":
    update_dataset()