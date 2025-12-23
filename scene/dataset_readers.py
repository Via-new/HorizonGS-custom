#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import glob
import sys
import re
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
try:
    import laspy
except:
    print("No laspy")
from utils.graphics_utils import BasicPointCloud
import concurrent.futures

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    CX: np.array
    CY: np.array
    image: np.array
    mask: np.array
    depth: np.array
    depth_params: dict
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def read_las_file(path):
    las = laspy.read(path)
    positions = np.vstack((las.x, las.y, las.z)).transpose()
    try:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.random.rand(positions.shape[0], positions.shape[1])

    return positions, colors, normals

def read_multiple_las_files(paths, ply_path):
    all_positions = []
    all_colors = []
    all_normals = []

    for path in paths:
        positions, colors, normals = read_las_file(path)
        all_positions.append(positions)
        all_colors.append(colors)
        all_normals.append(normals)

    all_positions = np.vstack(all_positions)
    all_colors = np.vstack(all_colors)
    all_normals = np.vstack(all_normals)

    print("Saving point cloud to .ply file...")
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(all_positions.shape[0], dtype=dtype)
    attributes = np.concatenate((all_positions, all_normals, all_colors), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

    return BasicPointCloud(points=all_positions, colors=all_colors, normals=all_normals)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, masks_folder, depths_folder, add_aerial, add_street):
#     cam_infos = []
    
#     def process_frame(idx, key):
#         # key 就是 COLMAP 的 Image ID (Integer)
#         extr = cam_extrinsics[key]
#         intr = cam_intrinsics[extr.camera_id]
#         height = intr.height
#         width = intr.width

#         uid = intr.id
#         R = np.transpose(qvec2rotmat(extr.qvec))
#         T = np.array(extr.tvec)

#         if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
#             focal_length_x = intr.params[0]
#             CX, CY = intr.params[1], intr.params[2]
#             FovY = focal2fov(focal_length_x, height)
#             FovX = focal2fov(focal_length_x, width)
#         elif intr.model=="PINHOLE":
#             focal_length_x = intr.params[0]
#             focal_length_y = intr.params[1]
#             CX, CY = intr.params[2], intr.params[3]
#             FovY = focal2fov(focal_length_y, height)
#             FovX = focal2fov(focal_length_x, width)
#         else:
#             assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
#         image_path = os.path.join(images_folder, extr.name)
#         if not os.path.exists(image_path):
#             return None
#         image_name = os.path.basename(image_path).split(".")[0]
#         image = Image.open(image_path)

#         # ================= [FIX START] 鲁棒的参数查找 =================
#         depth_params = None
#         if depths_params is not None:
#             # 构造所有可能的 Key，挨个试
#             keys_to_try = [
#                 key,                        # 1. 整数 ID (最推荐)
#                 str(key),                   # 2. 字符串 ID
#                 extr.name,                  # 3. 相对路径 (aerial/001.png)
#                 os.path.basename(extr.name),# 4. 文件名 (001.png)
#                 extr.name.split(".")[0]     # 5. 旧逻辑 (aerial/001)
#             ]
            
#             for k in keys_to_try:
#                 if k in depths_params:
#                     depth_params = depths_params[k]
#                     break
            
#             # 如果试了一圈还是没找到，打印详细报错
#             if depth_params is None:
#                 # 为了防止多线程打印混乱，这里简单 print，或者你可以用 tqdm.write
#                 print(f"\n[WARN] ID {key} ({extr.name}) not found in depths_params!")
#         # ================= [FIX END] =================

#         if masks_folder is not None:
#             mask_path = os.path.join(masks_folder, extr.name)
#             if os.path.exists(mask_path):
#                 mask = Image.open(mask_path)
#             else:
#                 mask = None
#         else:
#             mask = None

#         # ================= [FIX START] 鲁棒的深度图路径查找 =================
#         depth = None
#         if depths_folder is not None:
#             # 尝试多种深度图命名/路径格式
#             # 1. 直接替换后缀 (最常见)
#             p1 = os.path.join(depths_folder, extr.name.replace(".JPG", ".png").replace(".jpg", ".png"))
#             # 2. 也是替换后缀为 .npy
#             p2 = os.path.join(depths_folder, extr.name.replace(".JPG", ".npy").replace(".jpg", ".npy"))
#             # 3. 只有文件名 (不带 aerial/street 目录)
#             p3 = os.path.join(depths_folder, os.path.basename(extr.name).replace(".JPG", ".png").replace(".jpg", ".png"))
#             p4 = os.path.join(depths_folder, os.path.basename(extr.name).replace(".JPG", ".npy").replace(".jpg", ".npy"))

#             candidate_paths = [p1, p2, p3, p4]
            
#             for d_path in candidate_paths:
#                 if os.path.exists(d_path):
#                     if d_path.endswith(".npy"):
#                         depth = np.load(d_path)
#                     else:
#                         # 假设 png 存的是 16bit 深度，需要除以 65535 或者 1000 (视具体数据而定)
#                         # HorizonGS 默认行为是 / 2**16
#                         depth = cv2.imread(d_path, -1)
#                         if depth is not None:
#                             depth = depth.astype(np.float32) / float(2**16)
                    
#                     if depth is not None:
#                         break # 找到了就退出循环
#         # ================= [FIX END] =================

#         return CameraInfo(
#             uid=uid, 
#             R=R, 
#             T=T, 
#             FovY=FovY,
#             FovX=FovX, 
#             CX=CX,
#             CY=CY,
#             image=image,
#             mask=mask,
#             depth=depth,
#             depth_params=depth_params,
#             image_path=image_path, 
#             image_name=image_name, 
#             width=width, 
#             height=height
#         )

#     ct = 0
#     # 注意：这里把 cam_extrinsics 转为 list 可能会更稳定，因为它是 dict
#     progress_bar = tqdm(cam_extrinsics, desc="Loading dataset")

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_frame, idx, key) for idx, key in enumerate(cam_extrinsics)]

#         for future in concurrent.futures.as_completed(futures):
#             cam_info = future.result()
#             if cam_info is None:
#                 continue
            
#             # 这里的路径判断也建议加个 lower() 增加鲁棒性
#             path_lower = cam_info.image_path.lower()
            
#             if "aerial" in path_lower:
#                 if add_aerial:
#                     cam_infos.append(cam_info)
#             elif "street" in path_lower: 
#                 if add_street:
#                     cam_infos.append(cam_info)
#             else:
#                 # 默认行为
#                 cam_infos.append(cam_info)
            
#             ct+=1
#             if ct % 10 == 0:
#                 progress_bar.set_postfix({"num": f"{ct}/{len(cam_extrinsics)}"})
#                 progress_bar.update(10)

#         progress_bar.close()

#     cam_infos = sorted(cam_infos, key = lambda x : x.image_path)
#     return cam_infos

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, masks_folder, depths_folder, add_aerial, add_street):
    cam_infos = []
    
    def process_frame(idx, key):
        # key 就是 COLMAP 的 Image ID (Integer)
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            CX, CY = intr.params[1], intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            CX, CY = intr.params[2], intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        image_path = os.path.join(images_folder, extr.name)
        if not os.path.exists(image_path):
            return None
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # ---------------- 查找深度参数 ----------------
        depth_params = None
        if depths_params is not None:
            # 尝试多种 Key 匹配，增加鲁棒性
            keys_to_try = [
                key, str(key), extr.name, 
                os.path.basename(extr.name), extr.name.split(".")[0]
            ]
            for k in keys_to_try:
                if k in depths_params:
                    depth_params = depths_params[k]
                    break
        
        # ---------------- 查找 Mask (并区分是否存在) ----------------
        mask = None
        # 这里只负责“尝试读取”，如果不存在就保持 None
        if masks_folder is not None:
            mask_path = os.path.join(masks_folder, extr.name)
            # 尝试原名
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
            else:
                # 尝试替换后缀 (例如 jpg -> png)
                mask_path_png = os.path.splitext(mask_path)[0] + ".png"
                if os.path.exists(mask_path_png):
                    mask = Image.open(mask_path_png)

        # ---------------- 查找深度图 ----------------
        depth = None
        if depths_folder is not None:
            # 尝试常见深度图后缀
            candidates = [
                extr.name.replace(".JPG", ".png").replace(".jpg", ".png"),
                extr.name.replace(".JPG", ".npy").replace(".jpg", ".npy"),
                os.path.basename(extr.name).replace(".JPG", ".png").replace(".jpg", ".png")
            ]
            for c_name in candidates:
                d_path = os.path.join(depths_folder, c_name)
                if os.path.exists(d_path):
                    if d_path.endswith(".npy"):
                        depth = np.load(d_path)
                    else:
                        depth = cv2.imread(d_path, -1)
                        if depth is not None:
                            depth = depth.astype(np.float32) / float(2**16)
                    break

        return CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, CX=CX, CY=CY,
            image=image, mask=mask, depth=depth, depth_params=depth_params,
            image_path=image_path, image_name=image_name, width=width, height=height
        )

    ct = 0
    # [新增] 真实 Mask 计数器
    real_mask_count = 0
    
    progress_bar = tqdm(cam_extrinsics, desc="Loading dataset")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, idx, key) for idx, key in enumerate(cam_extrinsics)]

        for future in concurrent.futures.as_completed(futures):
            cam_info = future.result()
            if cam_info is None:
                continue
            
            # 过滤逻辑
            path_lower = cam_info.image_path.lower()
            keep = False
            
            if "aerial" in path_lower:
                if add_aerial: keep = True
            elif "street" in path_lower: 
                if add_street: keep = True
            else:
                keep = True # 默认保留其他图片
            
            if keep:
                cam_infos.append(cam_info)
                # [新增] 如果读到了有效的 mask 对象，计数加 1
                if cam_info.mask is not None:
                    real_mask_count += 1
            
            ct+=1
            if ct % 10 == 0:
                progress_bar.set_postfix({"num": f"{ct}/{len(cam_extrinsics)}"})
                progress_bar.update(10)

        progress_bar.close()

    # [新增] 打印真实的 Mask 加载统计信息
    total_cams = len(cam_infos)
    print("\n" + "="*50)
    print(f" [Dataset Reader Report]")
    print(f" Total Cameras Loaded : {total_cams}")
    print(f" Masks Found on Disk  : {real_mask_count}")
    
    if real_mask_count == 0:
        print(f" {Fore.RED}[WARNING] No masks found! Training will use default FULL-WHITE masks.{Style.RESET_ALL}")
    elif real_mask_count < total_cams:
        print(f" {Fore.YELLOW}[WARNING] Missing {total_cams - real_mask_count} masks. Missing ones will default to FULL-WHITE.{Style.RESET_ALL}")
    else:
        print(f" {Fore.GREEN}[OK] All {total_cams} cameras have corresponding mask files.{Style.RESET_ALL}")
    print("="*50 + "\n")

    cam_infos = sorted(cam_infos, key = lambda x : x.image_path)
    return cam_infos


def readUCGSCameras(cam_extrinsics, cam_intrinsics, images_folder, add_aerial, add_street):
    cam_infos = []
    
    def process_frame(idx, key):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            CX, CY = intr.params[1], intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            CX, CY = intr.params[2], intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        if not os.path.exists(image_path):
            return None
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        depth_params = None
        mask = None
        depth = None

        return CameraInfo(
            uid=uid, 
            R=R, 
            T=T, 
            FovY=FovY,
            FovX=FovX, 
            CX=CX,
            CY=CY,
            image=image,
            mask=mask,
            depth=depth,
            depth_params=depth_params,
            image_path=image_path, 
            image_name=image_name, 
            width=width, 
            height=height
        )

    ct = 0
    progress_bar = tqdm(cam_extrinsics, desc="Loading dataset")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, idx, key) for idx, key in enumerate(cam_extrinsics)]

        for future in concurrent.futures.as_completed(futures):
            cam_info = future.result()
            if cam_info is None:
                continue
            if "train" in cam_info.image_name and int(re.findall(r'\d+',cam_info.image_name)[0]) >= 352:
                if add_aerial:
                    cam_infos.append(cam_info)
            else:
                if add_street:
                    cam_infos.append(cam_info)
            
            ct+=1
            if ct % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(cam_extrinsics)}"+Style.RESET_ALL})
                progress_bar.update(10)

        progress_bar.close()

    cam_infos = sorted(cam_infos, key = lambda x : x.image_path)
    return cam_infos

def readCamerasFromTransforms(path, transformsfile, add_mask, add_depth, add_aerial, add_street, center, scale):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]

        def process_frame(idx, frame):
            # Process each frame and append cam_info to cam_infos list
            cam_name = frame["file_path"]
            image_path = os.path.join(path, cam_name)
            if not os.path.exists(image_path):
                raise ValueError(f"Image {image_path} does not exist!")
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])

            c2w[:3, 3] -= center
            c2w[:3, 3] /= scale

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if "small_city" in path:
                c2w[-1, -1] = 1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if (
                "k1" in frame
                and "k2" in frame
                and "p1" in frame
                and "p2" in frame
                and "k3" in frame
            ):
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist = np.array(
                    [frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]],
                    dtype=np.float32,
                )
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx
                CX = image.size[0] / 2
                CY = image.size[1] / 2
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])
                CX = frame["cx"]
                CY = frame["cy"]

            mask = None
            if add_mask:
                mask_path = os.path.join(path, frame["mask_path"])
                mask = Image.open(mask_path)

            depth = None
            if add_depth and "depth_path" in frame:
                depth_name = os.path.join(path, frame["depth_path"])
                depth_scale = 10000.0 * scale
                depth = cv2.imread(depth_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if len(depth.shape) == 3:
                    depth = depth[..., 0]
                depth /= depth_scale
            
            return CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                CX=CX,
                CY=CY,
                image=image,
                mask=mask,
                depth=depth,
                depth_params={"scale":(6 / scale)},
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
        ct = 0
        progress_bar = tqdm(frames, desc="Loading dataset")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_frame, idx, frame) for idx, frame in enumerate(frames)]

            for future in concurrent.futures.as_completed(futures):
                cam_info = future.result()
                if "aerial" in cam_info.image_path:
                    if add_aerial:
                        cam_infos.append(cam_info)
                elif "street" in cam_info.image_path: 
                    if add_street:
                        cam_infos.append(cam_info)
                else:
                    raise ValueError(f"Unknown image type: {cam_info.image_path}")
                
                ct+=1
                if ct % 10 == 0:
                    progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                    progress_bar.update(10)
            
            progress_bar.close()
    
    cam_infos = sorted(cam_infos, key = lambda x : x.image_path)
    return cam_infos

def readColmapSceneInfo(path, eval, images, add_mask, add_depth, add_aerial, add_street, llffhold=32):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    depths_params = None
    if add_depth:
        try:
            print(f"[LOADER] Loading depth params from {depth_params_file}")
            with open(depth_params_file, "r") as f:
                raw_params = json.load(f)
            
            # === [FIX START] 全覆盖填充策略 ===
            depths_params = {}
            
            # 1. 建立基础索引: String ID -> Params
            # raw_params 的 key 是 String ID ("1", "2"...)
            
            # 2. 利用 cam_extrinsics 建立映射表
            # 我们需要把 ID, Name, Basename 全部指向同一个参数对象
            for img_id, img_info in cam_extrinsics.items():
                # img_id 是整数 (e.g., 1)
                # img_info.name 是文件名 (e.g., "aerial/aerial_0001.png")
                
                str_id = str(img_id)
                
                # 尝试从原始 JSON 中获取参数
                if str_id in raw_params:
                    param_val = raw_params[str_id]
                elif img_id in raw_params: # 虽然不太可能，但也检查下整数
                    param_val = raw_params[img_id]
                else:
                    # 如果找不到，说明 JSON 里缺数据
                    # print(f"Warning: ID {img_id} not found in json")
                    continue
                
                # 填充所有可能的键
                depths_params[img_id] = param_val           # Key: 1 (Int)
                depths_params[str_id] = param_val           # Key: "1" (Str)
                depths_params[img_info.name] = param_val    # Key: "aerial/aerial_0001.png"
                
                # 甚至加上纯文件名 (不带路径)
                basename = os.path.basename(img_info.name)
                depths_params[basename] = param_val         # Key: "aerial_0001.png"
                
                # 甚至加上无后缀名
                name_no_ext = os.path.splitext(basename)[0]
                depths_params[name_no_ext] = param_val      # Key: "aerial_0001"

            print(f"[LOADER] Depth params populated with {len(depths_params)} keys (Multi-index).")
            # === [FIX END] ===

            # 计算中位数 Scale
            # 注意：raw_params 里的才是原始独立数据，计算中位数用它
            all_scales = np.array([v["scale"] for v in raw_params.values()])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            
            # 更新到所有条目 (因为是引用，改一个即可同步)
            for key in raw_params:
                raw_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = os.path.join(path, images)
    mask_dir = os.path.join(path, "masks") if add_mask else None
    depth_dir = os.path.join(path, "depths") if add_depth else None

    # # ================= [DEBUG START] 案发现场取证 =================
    # print("\n" + "="*20 + " INTERNAL DEBUG " + "="*20)
    
    # # 1. 检查传进去的 depths_params 到底有些什么 Key
    # if depths_params is not None:
    #     dp_keys = list(depths_params.keys())
    #     print(f"[DEBUG] depths_params 长度: {len(dp_keys)}")
    #     print(f"[DEBUG] depths_params 前5个 Key: {dp_keys[:5]}")
    #     print(f"[DEBUG] depths_params Key 类型: {type(dp_keys[0])}")
        
    #     # 2. 针对性测试：是否存在整数 1 和字符串 '1'
    #     print(f"[DEBUG] 整数 1 在 depths_params 吗? -> {1 in depths_params}")
    #     print(f"[DEBUG] 字符串 '1' 在 depths_params 吗? -> {'1' in depths_params}")
    # else:
    #     print("[DEBUG] depths_params is None!")

    # # 3. 检查 cam_extrinsics (这是 readColmapCameras 用来遍历的源头)
    # ce_keys = list(cam_extrinsics.keys())
    # print(f"[DEBUG] cam_extrinsics 长度: {len(ce_keys)}")
    # print(f"[DEBUG] cam_extrinsics 前5个 Key: {ce_keys[:5]}")
    # print(f"[DEBUG] cam_extrinsics Key 类型: {type(ce_keys[0])}")
    
    # # 4. 模拟 readColmapCameras 的查找过程
    # test_id = ce_keys[0] # 拿第一个相机ID做测试
    # print(f"[DEBUG] 模拟查找 ID: {test_id} (Type: {type(test_id)})")
    # if depths_params is not None:
    #     if test_id in depths_params:
    #         print(f"[DEBUG] -> 查找成功! Value: {depths_params[test_id]}")
    #     else:
    #         print(f"[DEBUG] -> 查找失败! (这就是报错的原因)")
    
    # print("="*56 + "\n")
    # # ================= [DEBUG END] =================

    cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, reading_dir, mask_dir, depth_dir, add_aerial, add_street)  
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    # try:
    print(f'start fetching data from ply file')
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# def readColmapSceneInfo(path, eval, images, add_mask, add_depth, add_aerial, add_street, llffhold=32):
#     try:
#         cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
#         cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
#         cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#     except:
#         cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
#         cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
#         cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

#     depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
#     depths_params = None
#     if add_depth:
#         try:
#             with open(depth_params_file, "r") as f:
#                 depths_params = json.load(f)
#             all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
#             if (all_scales > 0).sum():
#                 med_scale = np.median(all_scales[all_scales > 0])
#             else:
#                 med_scale = 0
#             for key in depths_params:
#                 depths_params[key]["med_scale"] = med_scale

#         except FileNotFoundError:
#             print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
#             sys.exit(1)
#         except Exception as e:
#             print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
#             sys.exit(1)

#     reading_dir = os.path.join(path, images)
#     mask_dir = os.path.join(path, "masks") if add_mask else None
#     depth_dir = os.path.join(path, "depths") if add_depth else None
#     cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, reading_dir, mask_dir, depth_dir, add_aerial, add_street)
    
#     if eval:
#         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
#         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     ply_path = os.path.join(path, "sparse/0/points3D.ply")
#     bin_path = os.path.join(path, "sparse/0/points3D.bin")
#     txt_path = os.path.join(path, "sparse/0/points3D.txt")
#     if not os.path.exists(ply_path):
#         print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
#         try:
#             xyz, rgb, _ = read_points3D_binary(bin_path)
#         except:
#             xyz, rgb, _ = read_points3D_text(txt_path)
#         storePly(ply_path, xyz, rgb)
#     # try:
#     print(f'start fetching data from ply file')
#     pcd = fetchPly(ply_path)

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info

def readNerfSyntheticInfo(path, eval, add_mask, add_depth, add_aerial, add_street, center, scale):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", add_mask, add_depth, add_aerial, add_street, center, scale)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", add_mask, add_depth, add_aerial, add_street, center, scale)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_paths = glob.glob(os.path.join(path, "*.ply"))
    if len(ply_paths)==0:
        ply_path = os.path.join(path, "points3d.ply")
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        colors = np.random.random((num_pts, 3))
        normals=np.zeros((num_pts, 3))
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

        storePly(ply_path, xyz, colors*255)
    else:
        ply_path = ply_paths[0]
        pcd = fetchPly(ply_path)
    
    pcd.points[:, :] -= center
    pcd.points[:, :] /= scale  # mainly adapt to params

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCityInfo(path, eval, add_mask, add_depth, add_aerial, add_street, center, scale, llffhold=32):
    
    json_path = glob.glob(os.path.join(path, f"transforms.json"))[0].split('/')[-1]
    print("Reading Training Transforms from {}".format(json_path))
    
    # load ply
    ply_path = glob.glob(os.path.join(path, "*.ply"))[0]
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except:
            raise ValueError("must have tiepoints!")
    else:
        las_paths = glob.glob(os.path.join(path, "LAS/*.las"))
        las_path = las_paths[0]
        print(f'las_path: {las_path}')
        try:
            pcd = read_multiple_las_files(las_paths, ply_path)
        except:
            raise ValueError("Load LAS failed!")
    
    # recenter poses and points clouds
    pcd.points[:,:] -= center
    pcd.points[:,:] /=scale # mainly adapt to params
    
    # load camera
    cam_infos = readCamerasFromTransforms(path, json_path, add_mask, add_depth, add_aerial, add_street, center, scale)
    
    print("Load Cameras: ", len(cam_infos))
    train_cam_infos = []
    test_cam_infos = []
    
    if not eval:
        train_cam_infos.extend(cam_infos)
        test_cam_infos = []
    else:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readUCGSInfo(path, images, add_aerial, add_street):
    print(path)
    if "NYC" in path:
        train_path = os.path.join(path, "traina20g1.5")
        test_pos_path = os.path.join(path, "test1.6")
        test_pos_rot_path = os.path.join(path, "test1.6d5")
    elif "SF" in path:
        train_path = os.path.join(path, "traina20g1.8")
        test_pos_path = os.path.join(path, "test1.9")
        test_pos_rot_path = os.path.join(path, "test1.9d5")
    
    def load_cam_infos(cam_path):
        
        print(cam_path)
        try:
            cameras_extrinsic_file = os.path.join(cam_path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(cam_path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(cam_path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(cam_path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        reading_dir = os.path.join(cam_path, images)
        cam_infos = readUCGSCameras(cam_extrinsics, cam_intrinsics, reading_dir, add_aerial, add_street)
        
        return cam_infos
    
    cam_20_infos = load_cam_infos(train_path)
    cam_view1_infos = load_cam_infos(test_pos_path)
    cam_view1d5_infos = load_cam_infos(test_pos_rot_path)
    
    train_cam_infos = [c for idx, c in enumerate(cam_20_infos) if "eval" not in c.image_name] # 315 ground images and 351 aerial images
    test_cam_20_infos = [c for idx, c in enumerate(cam_20_infos) if "eval" in c.image_name] # 36
    test_cam_view1_infos = [c for idx, c in enumerate(cam_view1_infos) if "eval" in c.image_name] # 36
    test_cam_view1d5_infos = [c for idx, c in enumerate(cam_view1d5_infos) if "eval" in c.image_name] # 36

    test_cam_infos = test_cam_20_infos + test_cam_view1_infos + test_cam_view1d5_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(train_path, "sparse/0/points3D.ply")
    bin_path = os.path.join(train_path, "sparse/0/points3D.bin")
    txt_path = os.path.join(train_path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    # try:
    print(f'start fetching data from ply file')
    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
    
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "City": readCityInfo,
    "UCGS": readUCGSInfo,
}