import os
import numpy as np
from scene.colmap_loader import read_extrinsics_text, read_points3D_text

# 新增: 手动读取 points3D.txt 以确保获取 image_ids (因为某些 loader 可能只返回坐标数组)
def load_points3D_tracks(path):
    points = {}
    print(f"正在手动解析 {path} ...")
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            elems = line.split()
            try:
                pid = int(elems[0])
                # Track starts at index 8. It contains pairs (image_id, point2d_idx)
                # We take every second element starting from 8: 8, 10, 12...
                image_ids = [int(elems[i]) for i in range(8, len(elems), 2)]
                points[pid] = set(image_ids)
            except ValueError:
                continue
    return points

def calculate_cross_view_overlap(colmap_path):
    print("正在加载 COLMAP 数据...")
    images = read_extrinsics_text(os.path.join(colmap_path, "images.txt"))
    
    # 修改: 使用自定义函数加载 points3D，避免 loader 返回 numpy 数组导致报错
    # points3D = read_points3D_text(os.path.join(colmap_path, "points3D.txt"))
    point_to_images = load_points3D_tracks(os.path.join(colmap_path, "points3D.txt"))

    # 1. 区分航拍和街景 (需根据你的文件名修改判断逻辑)
    aerial_imgs = {k: v for k, v in images.items() if "aerial" in v.name or "drone" in v.name}
    street_imgs = {k: v for k, v in images.items() if k not in aerial_imgs}
    
    print(f"航拍图: {len(aerial_imgs)} 张, 街景图: {len(street_imgs)} 张")

    # 2. 构建 3D点 -> 观测相机的反向索引 (加速查询)
    # 格式: {point3d_id: {image_id1, image_id2, ...}}
    # point_to_images 已经在 load_points3D_tracks 中构建好了，直接使用
    
    # 3. 遍历每一张街景图计算重叠
    results = []

    print("开始计算重叠率...")
    for s_id, s_img in street_imgs.items():
        # 获取该街景图所有的有效3D点
        s_p3d_ids = [p for p in s_img.point3D_ids if p != -1]
        total_s_points = len(s_p3d_ids)
        
        if total_s_points == 0:
            continue

        # 统计这张街景图与所有航拍图的共视点数
        # aerial_overlaps: {aerial_id: common_point_count}
        aerial_overlaps = {}
        
        for pid in s_p3d_ids:
            if pid in point_to_images:
                # 查看这个点被哪些航拍图看到
                linked_img_ids = point_to_images[pid]
                for linked_id in linked_img_ids:
                    if linked_id in aerial_imgs:
                        aerial_overlaps[linked_id] = aerial_overlaps.get(linked_id, 0) + 1
        
        # 4. 计算具体的重叠率并排序
        # 找出与当前街景重叠度最高的航拍图
        best_aerial_match = None
        max_common_points = 0
        
        for a_id, count in aerial_overlaps.items():
            if count > max_common_points:
                max_common_points = count
                best_aerial_match = a_id
        
        if best_aerial_match:
            a_img = aerial_imgs[best_aerial_match]
            # 获取航拍图的总点数
            total_a_points = len([p for p in a_img.point3D_ids if p != -1])
            
            # 计算两个比率
            ratio_street_view = max_common_points / total_s_points # 街景被覆盖了多少
            ratio_aerial_view = max_common_points / total_a_points # 占了航拍多大比例
            
            results.append({
                "street_name": s_img.name,
                "best_aerial_name": a_img.name,
                "common_points": max_common_points,
                "ratio_street": ratio_street_view,
                "ratio_aerial": ratio_aerial_view
            })

    # 5. 按照街景重叠率排序输出前 10 个
    results.sort(key=lambda x: x['ratio_street'], reverse=True)
    
    print("\nTop 10 重叠对 (Street -> Aerial):")
    for r in results[:10]:
        print(f"街景: {r['street_name']} <-> 航拍: {r['best_aerial_name']}")
        print(f"  共视点: {r['common_points']}")
        print(f"  街景重叠率: {r['ratio_street']:.2%}")
        print(f"  航拍重叠率: {r['ratio_aerial']:.2%}")
        print("-" * 30)

    return results

# 使用方法:
calculate_cross_view_overlap("data/fusion/train/sparse/0/text_format")