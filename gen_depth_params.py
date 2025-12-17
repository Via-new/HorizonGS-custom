import os
import json
import numpy as np
import cv2
import struct
import argparse
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import RANSACRegressor

# --- COLMAP 读取辅助函数 (保持不变) ---
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "i4d3di")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = {
                "id": image_id, "qvec": qvec, "tvec": tvec,
                "camera_id": camera_id, "name": image_name,
                "xys": xys, "point3D_ids": point3D_ids
            }
    return images

def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            points3D[point3D_id] = {"id": point3D_id, "xyz": xyz, "rgb": rgb, "error": error}
    return points3D

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def compute_depth_params(dataset_path):
    sparse_path = os.path.join(dataset_path, "sparse/0")
    depth_dir = os.path.join(dataset_path, "depths") 
    
    print(f"Loading COLMAP data from {sparse_path}...")
    try:
        images = read_images_binary(os.path.join(sparse_path, "images.bin"))
        points3D = read_points3D_binary(os.path.join(sparse_path, "points3D.bin"))
    except Exception as e:
        print(f"Error loading COLMAP data: {e}")
        return

    depth_params = {}
    
    # 统计数据
    valid_scales = []
    valid_offsets = []
    failed_keys = []
    
    # 详细计数器
    cnt_total = len(images)
    cnt_no_depth_file = 0
    cnt_not_enough_points = 0
    cnt_ransac_fail = 0
    cnt_negative_scale = 0
    cnt_success = 0
    cnt_sky_filtered = 0  # 天空过滤统计

    print(f"Start processing {cnt_total} images with RANSAC...")
    
    for _, img_data in tqdm(images.items()):
        img_name = img_data["name"]
        img_id = img_data["id"]
        dict_key = str(img_id)

        # --- 1. 寻找深度图 ---
        base_name_no_ext = os.path.splitext(os.path.basename(img_name))[0]
        rel_path_no_ext = os.path.splitext(img_name)[0]
        
        candidates = [
            os.path.join(depth_dir, rel_path_no_ext + ".npy"),
            os.path.join(depth_dir, rel_path_no_ext + ".png"),
            os.path.join(depth_dir, "aerial", base_name_no_ext + ".npy"),
            os.path.join(depth_dir, "street", "train", base_name_no_ext + ".npy"),
            os.path.join(depth_dir, base_name_no_ext + ".npy"),
            os.path.join(depth_dir, img_name.replace(".jpg", ".png").replace(".png", ".png"))
        ]
        
        mono_depth = None
        for path in candidates:
            if os.path.exists(path):
                try:
                    if path.endswith(".npy"): 
                        # .npy 文件：直接加载浮点数深度（DAV2原始输出）
                        mono_depth = np.load(path)
                    elif path.endswith(".png"): 
                        # .png 文件：16位归一化深度，需要反归一化
                        # DAV2保存的PNG是归一化到0-65535的，但这是相对深度
                        # 注意：PNG格式的深度图丢失了绝对尺度信息，只能用于可视化
                        # 建议优先使用.npy文件
                        temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                        if temp is not None:
                            if temp.ndim == 3:
                                temp = temp[..., 0]  # 如果是3通道，取第一个通道
                            # 16位PNG：值域0-65535，需要反归一化
                            # 但注意：由于保存时做了min-max归一化，这里无法恢复绝对尺度
                            # 只能得到相对深度（0-1范围）
                            if temp.max() > 255:  # 16位深度图
                                mono_depth = temp.astype(np.float32) / 65535.0
                            else:  # 8位深度图
                                mono_depth = temp.astype(np.float32) / 255.0
                    if mono_depth is not None and mono_depth.size > 0: 
                        break
                except Exception as e:
                    continue
        
        if mono_depth is None:
            cnt_no_depth_file += 1
            failed_keys.append(dict_key)
            continue

        # --- 2. 提取匹配点对 ---
        p3d_ids = img_data["point3D_ids"]
        xys = img_data["xys"]
        valid_mask = p3d_ids != -1
        
        valid_p3d_ids = p3d_ids[valid_mask]
        valid_xys = xys[valid_mask]
        
        R = qvec2rotmat(img_data["qvec"])
        t = img_data["tvec"]
        
        colmap_inv_depths = []
        mono_depths = []
        h, w = mono_depth.shape[:2]
        
        # 天空过滤参数（针对街景图）
        is_street_view = "street" in img_name.lower()
        sky_filter_enabled = is_street_view  # 只对街景图启用天空过滤
        
        # 计算单目深度图的统计信息（用于过滤异常值）
        mono_depth_valid = mono_depth[mono_depth > 0]
        if len(mono_depth_valid) > 0:
            mono_depth_median = np.median(mono_depth_valid)
            mono_depth_p95 = np.percentile(mono_depth_valid, 95)  # 95分位数
            mono_depth_max_reasonable = mono_depth_p95 * 3  # 允许的最大深度值（用于宽松过滤）
        else:
            mono_depth_p95 = 1000.0
            mono_depth_max_reasonable = 1000.0
        
        # 统计过滤前的点数
        points_before_sky_filter = 0
        
        # 收集有效点
        for i, pid in enumerate(valid_p3d_ids):
            if pid not in points3D: continue
            xyz_world = points3D[pid]["xyz"]
            # COLMAP 坐标变换：R 是从世界到相机的旋转，t 是相机在世界坐标系中的位置
            # 标准变换：xyz_cam = R @ xyz_world + t (COLMAP 的 tvec 已经是 -R @ t)
            xyz_cam = R @ xyz_world + t
            metric_depth = xyz_cam[2] # 物理 Z
            
            u, v = int(round(valid_xys[i, 0])), int(round(valid_xys[i, 1]))
            
            # 基础过滤：Z > 0.5m (太近不稳定), Z < 1000m (太远不准)
            if not (0 <= u < w and 0 <= v < h and metric_depth > 0.5 and metric_depth < 1000):
                continue
            
            m_depth = mono_depth[v, u]
            if m_depth <= 0:
                continue
            
            points_before_sky_filter += 1
            
            # === 天空过滤（针对街景图）- 智能策略 ===
            if sky_filter_enabled:
                # 先收集所有候选点，然后根据点数决定过滤严格程度
                # 1. 过滤图像上半部分（天空通常在图像上方）
                # 只过滤掉图像上方20%的区域（降低阈值，避免过滤太多有效点）
                if v < h * 0.2:
                    continue
                
                # 2. 过滤单目深度图中过大的值（可能是天空的错误深度）
                # 使用更宽松的阈值：95分位数的3倍（而不是2倍）
                if m_depth > mono_depth_p95 * 3:
                    continue
                
                # 3. 过滤深度差异过大的点（单目深度和COLMAP深度差异过大，可能是天空）
                # 放宽阈值：从10倍放宽到20倍，避免过滤掉太多有效点
                depth_ratio = m_depth / metric_depth
                if depth_ratio > 20.0 or depth_ratio < 0.05:  # 差异超过20倍，可能是错误匹配
                    continue
            
            colmap_inv_depths.append(1.0 / metric_depth) # Target: 1/Z
            mono_depths.append(m_depth)              # Source: 单目深度图
        
        # 统计天空过滤效果
        if sky_filter_enabled and points_before_sky_filter > 0:
            filtered_count = points_before_sky_filter - len(colmap_inv_depths)
            if filtered_count > 0:
                cnt_sky_filtered += 1
        
        # 点数检查
        if len(colmap_inv_depths) < 15: # 提高门槛到15个点
            cnt_not_enough_points += 1
            failed_keys.append(dict_key)
            # tqdm.write(f"[Skip] {img_name}: Only {len(colmap_inv_depths)} overlapping points.")
            continue
            
        X = np.array(mono_depths).reshape(-1, 1)
        y = np.array(colmap_inv_depths)
        
        try:
            # === 诊断信息：检查数据分布 ===
            mono_depths_arr = np.array(mono_depths)
            colmap_depths_arr = 1.0 / np.array(colmap_inv_depths)  # 转换为深度用于诊断
            correlation = np.corrcoef(mono_depths_arr, colmap_depths_arr)[0, 1]
            
            # === RANSAC 拟合 ===
            # residual_threshold: 允许的误差范围 (逆深度单位)
            ransac = RANSACRegressor(min_samples=10, residual_threshold=0.05, max_trials=100)
            ransac.fit(X, y)
            
            scale = ransac.estimator_.coef_[0]
            offset = ransac.estimator_.intercept_
            
            # 统计内点 (Inliers)
            inlier_mask = ransac.inlier_mask_
            num_inliers = np.sum(inlier_mask)
            ratio = num_inliers / len(y)
            
            # === 尝试反向拟合（如果单目深度图是逆深度） ===
            # [FIXED] 的含义：
            # 当检测到负scale时，说明单目深度图和COLMAP深度可能是负相关
            # 这可能是因为单目深度图存储的是"逆深度"（1/depth）而不是"深度"（depth）
            # 反向拟合：将单目深度值取倒数（1/mono_depth），然后重新拟合
            # 如果反向拟合得到正scale且内点比例更高，说明单目深度图确实是逆深度格式
            if scale < 0:
                # 尝试假设单目深度图是逆深度
                X_inv = 1.0 / (mono_depths_arr + 1e-6)  # 将深度转换为逆深度
                X_inv = X_inv.reshape(-1, 1)
                try:
                    ransac_inv = RANSACRegressor(min_samples=10, residual_threshold=0.05, max_trials=100)
                    ransac_inv.fit(X_inv, y)  # 拟合：1/Z_colmap = scale * (1/D_mono) + offset
                    scale_inv = ransac_inv.estimator_.coef_[0]
                    offset_inv = ransac_inv.estimator_.intercept_
                    inlier_mask_inv = ransac_inv.inlier_mask_
                    num_inliers_inv = np.sum(inlier_mask_inv)
                    ratio_inv = num_inliers_inv / len(y)
                    
                    # 如果反向拟合更好，使用它
                    if scale_inv > 0 and ratio_inv > ratio:
                        scale = scale_inv
                        offset = offset_inv
                        num_inliers = num_inliers_inv
                        ratio = ratio_inv
                        tqdm.write(f"[FIXED] {img_name}: Reversed fit (mono depth is inverse depth). Scale={scale:.5f}, Inliers={num_inliers}/{len(y)} ({ratio:.2f})")
                except:
                    pass
            
            # --- 详细日志判断 ---
            if scale > 0:
                depth_params[dict_key] = {"scale": float(scale), "offset": float(offset)}
                valid_scales.append(scale)
                valid_offsets.append(offset)
                cnt_success += 1
                
                # 可选：打印高质量的拟合
                # if ratio > 0.8: 
                #     tqdm.write(f"[Good] {img_name}: Scale={scale:.4f}, Inliers={num_inliers}/{len(y)} ({ratio:.2f})")
            else:
                # 负 Scale 报警，输出诊断信息
                cnt_negative_scale += 1
                failed_keys.append(dict_key)
                tqdm.write(f"[BAD]  {img_name}: Negative Scale={scale:.5f}. Inliers={num_inliers}/{len(y)} ({ratio:.2f})")
                tqdm.write(f"       Correlation={correlation:.3f}, Mono range=[{mono_depths_arr.min():.2f}, {mono_depths_arr.max():.2f}], "
                          f"COLMAP range=[{colmap_depths_arr.min():.2f}, {colmap_depths_arr.max():.2f}]")
                
        except Exception as e:
            cnt_ransac_fail += 1
            failed_keys.append(dict_key)
            tqdm.write(f"[Error] RANSAC failed for {img_name}: {e}")

    # --- 后处理：中位数填充 ---
    print("\n" + "="*50)
    print("Alignment Statistics:")
    print(f"  Total Images       : {cnt_total}")
    print(f"  No Depth File      : {cnt_no_depth_file}")
    print(f"  Not Enough Points  : {cnt_not_enough_points}")
    print(f"  RANSAC Error       : {cnt_ransac_fail}")
    print(f"  Negative Scale     : {cnt_negative_scale} (Problematic images)")
    print(f"  Success            : {cnt_success}")
    print(f"  Sky Filter Applied  : {cnt_sky_filtered} street view images")
    
    if len(valid_scales) > 0:
        median_scale = float(np.median(valid_scales))
        median_offset = float(np.median(valid_offsets))
        print("-" * 30)
        print(f"  Global Median Scale : {median_scale:.6f}")
        print(f"  Global Median Offset: {median_offset:.6f}")
        print(f"  Fixing {len(failed_keys)} failed images with Median values...")
        
        for k in failed_keys:
            depth_params[k] = {"scale": median_scale, "offset": median_offset}
    else:
        print("-" * 30)
        print("  [CRITICAL] No valid scales found! Defaulting to 1.0/0.0")
        for k in failed_keys:
            depth_params[k] = {"scale": 1.0, "offset": 0.0}

    # 保存
    out_path = os.path.join(sparse_path, "depth_params.json")
    with open(out_path, 'w') as f:
        json.dump(depth_params, f, indent=4)
    print("="*50)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="data/depthTest/park", help="Path to the dataset root")
    args = parser.parse_args()
    
    compute_depth_params(args.source_path)