import os
import struct
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "data/aerial_street_fusion/train"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
ALIGNED_DEPTH_DIR = os.path.join(DATA_ROOT, "aligned_depths")
COLMAP_DIR = os.path.join(DATA_ROOT, "sparse/0")
OUTPUT_PLY = os.path.join(DATA_ROOT, "init_point_cloud.ply")

# 采样参数
PIXEL_STRIDE = 10   # 采样步长：4 表示每 4x4 个像素取 1 个点 (数值越大点越少，建议 4 或 8)
MAX_DEPTH = 150.0  # 深度阈值：超过 150 米的点通常是天空，扔掉
# ===========================================

# --- 1. 修复后的 COLMAP 读取函数 (与 script1 保持一致) ---
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
            # 修复：使用 correct format <IdddddddI
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

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def get_intrinsic_matrix(cam_params):
    # 处理 SIMPLE_PINHOLE 和 PINHOLE
    if len(cam_params) == 3: # SIMPLE_PINHOLE: f, cx, cy
        fx = fy = cam_params[0]
        cx, cy = cam_params[1], cam_params[2]
    elif len(cam_params) == 4: # PINHOLE: fx, fy, cx, cy
        fx, fy = cam_params[0], cam_params[1]
        cx, cy = cam_params[2], cam_params[3]
    else: 
        # 简单回退：假设前两个是f，后两个是c
        fx, fy = cam_params[0], cam_params[1]
        cx, cy = cam_params[2], cam_params[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# --- 主逻辑 ---
print("正在加载 COLMAP 数据...")
cameras = read_cameras_binary(os.path.join(COLMAP_DIR, "cameras.bin"))
images = read_images_binary(os.path.join(COLMAP_DIR, "images.bin"))
# 注意：生成点云不需要 points3D.bin，所以不需要读取它，避免报错

all_xyz = []
all_rgb = []

print(f"开始反投影生成点云 (采样步长: {PIXEL_STRIDE})...")

for img_id, img_data in tqdm(images.items()):
    # img_data["name"] 可能是 "aerial/aerial_0003.jpg"
    basename = os.path.splitext(img_data["name"])[0] # "aerial/aerial_0003"
    
    # 寻找对齐后的深度图 (.npy)
    depth_path = os.path.join(ALIGNED_DEPTH_DIR, basename + ".npy")
    image_path = os.path.join(IMAGES_DIR, img_data["name"])
    
    if not os.path.exists(depth_path): 
        # 尝试去掉子文件夹名 (以防万一 script1 是 flatten 保存的)
        flat_name = os.path.basename(basename)
        flat_path = os.path.join(ALIGNED_DEPTH_DIR, flat_name + ".npy")
        if os.path.exists(flat_path):
            depth_path = flat_path
        else:
            continue # 找不到深度图，跳过

    if not os.path.exists(image_path): continue

    # 1. 加载数据
    try:
        depth = np.load(depth_path) # (H, W) Metric Depth
        rgb = cv2.imread(image_path) # (H, W, 3) BGR
    except:
        continue
        
    if rgb is None: continue
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) # 转 RGB

    # 确保尺寸一致
    H, W = depth.shape
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    cam = cameras[img_data["cam_id"]]
    K = get_intrinsic_matrix(cam['params'])
    
    # 2. 构建像素网格 (u, v)
    u_vals = np.arange(0, W, PIXEL_STRIDE)
    v_vals = np.arange(0, H, PIXEL_STRIDE)
    uu, vv = np.meshgrid(u_vals, v_vals)
    
    # 获取对应点的深度和颜色
    d_samples = depth[vv, uu]
    color_samples = rgb[vv, uu] / 255.0 # 归一化颜色
    
    # 3. 过滤 (Masking)
    # 过滤掉极近点(可能是噪声)和极远点(天空)
    mask = (d_samples > 0.5) & (d_samples < MAX_DEPTH) 
    d_samples = d_samples[mask]
    color_samples = color_samples[mask]
    uu = uu[mask]
    vv = vv[mask]
    
    if len(d_samples) == 0: continue

    # 4. 反投影: Pixel -> Camera Coordinate
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_cam = (uu - cx) * d_samples / fx
    y_cam = (vv - cy) * d_samples / fy
    z_cam = d_samples
    
    p_cam = np.stack([x_cam, y_cam, z_cam], axis=-1) # (N, 3)

    # 5. 相机坐标 -> 世界坐标
    # COLMAP World-to-Cam: P_cam = R * P_world + t
    # Inverse (Cam-to-World): P_world = R.T * (P_cam - t)
    R = qvec2rotmat(img_data["q"])
    t = img_data["t"]
    
    # 向量化计算: (P_cam - t) @ R 
    # 原理: (R.T @ (P_cam - t).T).T  == (P_cam - t) @ R
    p_world = np.dot(p_cam - t, R) 
    
    all_xyz.append(p_world)
    all_rgb.append(color_samples)

# 合并所有点
print("正在合并点云...")
if len(all_xyz) == 0:
    print("错误: 没有生成任何点！请检查路径是否正确。")
    exit()

all_xyz = np.concatenate(all_xyz, axis=0)
all_rgb = np.concatenate(all_rgb, axis=0)

print(f"原始稠密点数: {len(all_xyz)}")

# --- Open3D 处理 ---
print("正在使用 Open3D 进行去噪 (这可能需要几分钟)...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_xyz)
pcd.colors = o3d.utility.Vector3dVector(all_rgb)

# 统计离群点移除 (Statistical Outlier Removal)
# 这步很关键，因为 DAV2 边缘会有飞点
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_clean = pcd.select_by_index(ind)

print(f"去噪后点数: {len(pcd_clean.points)}")

# 保存
o3d.io.write_point_cloud(OUTPUT_PLY, pcd_clean)
print(f"成功！已保存至: {OUTPUT_PLY}")
print("下一步：在训练 3DGS 时，将此文件作为 input.ply")