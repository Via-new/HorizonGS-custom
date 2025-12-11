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
                    if path.endswith(".npy"): mono_depth = np.load(path)
                    elif path.endswith(".png"): 
                        temp = Image.open(path)
                        mono_depth = np.array(temp).astype(np.float32)
                    if mono_depth is not None: break
                except: continue
        
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
        
        # 收集有效点
        for i, pid in enumerate(valid_p3d_ids):
            if pid not in points3D: continue
            xyz_world = points3D[pid]["xyz"]
            xyz_cam = np.dot(R, xyz_world) + t
            metric_depth = xyz_cam[2] # 物理 Z
            
            u, v = int(round(valid_xys[i, 0])), int(round(valid_xys[i, 1]))
            
            # 过滤：Z > 0.5m (太近不稳定), Z < 1000m (太远不准)
            if 0 <= u < w and 0 <= v < h and metric_depth > 0.5 and metric_depth < 1000:
                m_depth = mono_depth[v, u]
                if m_depth > 0:
                    colmap_inv_depths.append(1.0 / metric_depth) # Target: 1/Z
                    mono_depths.append(m_depth)              # Source: DVA2
        
        # 点数检查
        if len(colmap_inv_depths) < 15: # 提高门槛到15个点
            cnt_not_enough_points += 1
            failed_keys.append(dict_key)
            # tqdm.write(f"[Skip] {img_name}: Only {len(colmap_inv_depths)} overlapping points.")
            continue
            
        X = np.array(mono_depths).reshape(-1, 1)
        y = np.array(colmap_inv_depths)
        
        try:
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
                # 负 Scale 报警
                cnt_negative_scale += 1
                failed_keys.append(dict_key)
                tqdm.write(f"[BAD]  {img_name}: Negative Scale={scale:.5f}. Inliers={num_inliers}/{len(y)}")
                
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
    parser.add_argument("--source_path", type=str, default="data/fusion/train", help="Path to the dataset root")
    args = parser.parse_args()
    
    compute_depth_params(args.source_path)