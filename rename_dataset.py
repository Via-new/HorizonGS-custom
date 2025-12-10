import os

# ================= 配置区域 =================
# 指向你的 images 文件夹所在的路径
# 例如：data/aerial_street_fusion/train/images
images_root = "data/aerial_street_fusion/train/images" 
# ===========================================

def rename_files_in_dir(target_dir, prefix):
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    files = os.listdir(target_dir)
    count = 0
    print(f"Processing {target_dir}...")

    for filename in files:
        # 过滤掉非图片文件，且防止重复添加前缀
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and not filename.startswith(prefix):
            old_path = os.path.join(target_dir, filename)
            
            # 构造新名字：前缀 + 下划线 + 原名
            # 例如: 0001.jpg -> aerial_0001.jpg
            new_filename = f"{prefix}_{filename}"
            new_path = os.path.join(target_dir, new_filename)
            
            os.rename(old_path, new_path)
            count += 1
            # print(f"Renamed: {filename} -> {new_filename}") # 如果想看详细日志取消注释

    print(f"Done. Renamed {count} files in {target_dir}.\n")

if __name__ == "__main__":
    # 1. 处理航拍
    aerial_path = os.path.join(images_root, "aerial")
    rename_files_in_dir(aerial_path, "aerial")

    # 2. 处理街景
    street_path = os.path.join(images_root, "street")
    rename_files_in_dir(street_path, "street")
    
    print("All Finished! Now you can run COLMAP.")