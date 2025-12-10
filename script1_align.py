import os
import struct
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "data/aerial_street_fusion/train"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
DEPTHS_DIR = os.path.join(DATA_ROOT, "depths")
COLMAP_DIR = os.path.join(DATA_ROOT, "sparse/0")
OUTPUT_DIR = os.path.join(DATA_ROOT, "aligned_depths")
DEPTH_EXT = ".npy" 
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- COLMAP 读取工具函数 ---
def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            params = struct.unpack("<iiQQ", fid.read(24))
            camera_id = params[0]
            model_id = params[1]
            width = params[2]
            height = params[3]
            num_params = 4 if model_id == 1 else (3 if model_id == 0 else 0)
            if num_params == 0: 
                 pass 
            params = struct.unpack("<" + "d" * num_params, fid.read(8 * num_params))
            cameras[camera_id] = {"w": width, "h": height, "params": params}
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("<IdddddddI", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
            
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            points2D = []
            point3D_ids = []
            for _ in range(num_points2D):
                x, y, p3d_id = struct.unpack("<ddQ", fid.read(24))
                if p3d_id != 18446744073709551615:
                    points2D.append([x, y])
                    point3D_ids.append(p3d_id)
            
            images[image_id] = {
                "name": image_name, "cam_id": camera_id, "q": qvec, "t": tvec,
                "p2d": np.array(points2D), "p3d_ids": np.array(point3D_ids)
            }
    return images

def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            binary_point_properties = struct.unpack("<Qddd", fid.read(32))
            point3D_id = binary_point_properties[0]
            xyz = np.array(binary_point_properties[1:4])
            _ = fid.read(3) 
            _ = fid.read(8) # Skip Error
            track_len = struct.unpack("<Q", fid.read(8))[0]
            _ = fid.read(8 * track_len) 
            points3D[point3D_id] = xyz
    return points3D

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

# --- 主逻辑 ---
if __name__ == "__main__":
    print("正在读取 COLMAP 数据...")
    cameras = read_cameras_binary(os.path.join(COLMAP_DIR, "cameras.bin"))
    images = read_images_binary(os.path.join(COLMAP_DIR, "images.bin"))
    points3D = read_points3D_binary(os.path.join(COLMAP_DIR, "points3D.bin"))
    print(f"读取完毕: {len(images)} 张图片, {len(points3D)} 个 3D 点")

    for img_id, img_data in tqdm(images.items(), desc="对齐深度图"):
        basename = os.path.splitext(img_data["name"])[0]
        # 这里 basename 可能是 "aerial/aerial_0003"
        depth_path = os.path.join(DEPTHS_DIR, basename + ".npy")
        
        # 兼容性检查：如果 depth_path 不存在，尝试 flattened 结构（比如 depths/aerial_0003.npy）
        if not os.path.exists(depth_path):
             flat_name = os.path.basename(basename) # 只取 aerial_0003
             flat_path = os.path.join(DEPTHS_DIR, flat_name + ".npy")
             if os.path.exists(flat_path):
                 depth_path = flat_path
             else:
                 # 再尝试找 png
                 depth_path_png = os.path.join(DEPTHS_DIR, basename + ".png")
                 if os.path.exists(depth_path_png):
                     depth_path = depth_path_png
                 else:
                     continue

        # 读取
        if depth_path.endswith(".npy"):
            d_pred = np.load(depth_path)
        else:
            d_pred = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if d_pred.dtype == np.uint16:
                d_pred = d_pred.astype(np.float32) / 65535.0
            elif d_pred.dtype == np.uint8:
                d_pred = d_pred.astype(np.float32) / 255.0
        
        cam = cameras[img_data["cam_id"]]
        W, H = int(cam['w']), int(cam['h'])
        
        if d_pred.shape[:2] != (H, W):
            d_pred = cv2.resize(d_pred, (W, H), interpolation=cv2.INTER_NEAREST)

        valid_p3d_ids = img_data["p3d_ids"]
        valid_p2d = img_data["p2d"]
        gt_depths = []
        pred_depths = []
        R = qvec2rotmat(img_data["q"])
        t = img_data["t"]

        for i, p3d_id in enumerate(valid_p3d_ids):
            if p3d_id not in points3D: continue
            xyz_world = points3D[p3d_id]
            xyz_cam = np.dot(R, xyz_world) + t
            z_depth = xyz_cam[2]
            if z_depth <= 0.1: continue 

            u, v = int(round(valid_p2d[i][0])), int(round(valid_p2d[i][1]))
            if 0 <= u < W and 0 <= v < H:
                val = d_pred[v, u]
                gt_depths.append(z_depth)
                pred_depths.append(val)

        gt_depths = np.array(gt_depths).reshape(-1, 1)
        pred_depths = np.array(pred_depths).reshape(-1, 1)

        if len(gt_depths) < 10: continue

        try:
            ransac = RANSACRegressor(min_samples=5, residual_threshold=1.0) 
            ransac.fit(pred_depths, gt_depths)
            scale = ransac.estimator_.coef_[0][0]
            shift = ransac.estimator_.intercept_[0]
            if scale < 0: scale = abs(scale)
        except Exception as e:
            print(f"RANSAC 失败 ({img_data['name']}): {e}")
            continue

        d_aligned = d_pred * scale + shift
        d_aligned[d_aligned < 0] = 0 
        
        # === 修复: 自动创建子文件夹 ===
        save_path = os.path.join(OUTPUT_DIR, basename + ".npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # 关键：确保 subfolder 存在
        # ============================
        
        np.save(save_path, d_aligned)

    print("深度对齐完成！结果保存在:", OUTPUT_DIR)