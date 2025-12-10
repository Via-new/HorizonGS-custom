import argparse
import copy
import json
from types import SimpleNamespace
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN, KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from plyfile import PlyData, PlyElement
from typing import NamedTuple
import sys
import yaml
import torch
import os
import random
import matplotlib.pyplot as plt

# 将当前目录及上级目录添加到系统路径，以便导入自定义模块
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

# 导入自定义的预处理和配置生成模块
from preprocess.generate_chunks_config import generate_chunks_config
from preprocess.generate_config import generate_config
from preprocess.depth2pc import depth2pc, depth2pc_partition
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from utils.camera_utils import cameraList_from_camInfos
from utils.partition_utils import *
from utils.general_utils import *
from types import SimpleNamespace

# 定义保存点云为PLY格式的函数
def storePly(path, xyz, rgb):
    # 定义结构化数组的数据类型：坐标(x,y,z)，法线(nx,ny,nz)，颜色(r,g,b)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    # 初始化法线为0
    normals = np.zeros_like(xyz)

    # 创建空数组用于存储元素
    elements = np.empty(xyz.shape[0], dtype=dtype)
    # 拼接坐标、法线和颜色数据
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    # 将属性映射到元组列表并赋值
    elements[:] = list(map(tuple, attributes))

    # 创建PlyData对象并写入文件
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# 加载场景数据的函数
def load_data(args):
    center = [0, 0, 0]
    scale = 1.0
    # 根据参数选择不同的数据格式加载方式
    if args.data_format == 'blender':
        print("Use Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](
            args.source_path, args.eval, args.add_mask, args.add_depth, 
            args.add_aerial, args.add_street, center, scale
        )
    elif args.data_format == 'colmap':
        print("Use Colmap data set!")
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            args.source_path, args.eval, args.images, args.add_mask, args.add_depth, \
            args.add_aerial, args.add_street, args.llffhold
        )
    elif args.data_format == 'city':
        print("Use City data set!")
        scene_info = sceneLoadTypeCallbacks["City"](
            args.source_path, args.eval, args.add_mask, args.add_depth, \
            args.add_aerial, args.add_street, center, scale, args.llffhold
        )
    elif args.data_format == 'ucgs':
        print("Use UCGS data set!")
        scene_info = sceneLoadTypeCallbacks["UCGS"](
            args.source_path, args.images, args.add_aerial, args.add_street
        )
    else:
        assert False, "Could not recognize scene type!"
        
    return scene_info

# 基于相机位置进行区域划分的函数
def camera_position_based_region_division(pcd, train_cameras, m_region, n_region, plane_index):
    print("############ camera_position_based_region_division ############")
    m, n = m_region, n_region # m和n分别代表两个维度的划分数量
    points = pcd.points
    
    # 获取所有训练相机的中心坐标
    cameras = np.array([camera.camera_center.cpu().numpy() for camera in train_cameras])

    # 步骤 1: 将点和相机投影到指定的平面上（通常是x/y或x/z平面，由plane_index决定）
    x_points = points[:, plane_index[0]]
    y_points = points[:, plane_index[1]]
    x_cameras = cameras[:, plane_index[0]]
    
    # 步骤 2: 确定包围盒（Bounding Box）
    x_min, x_max = np.min(x_points), np.max(x_points)
    y_min, y_max = np.min(y_points), np.max(y_points)

    # 步骤 3: 统计相机总数
    V = len(train_cameras)

    # 步骤 4: 沿第一个轴（例如x轴）将包围盒分成m段
    segment_size_x = V / m # 每段理想的相机数量
    x_segments = []
    x_segments_cameras = []
    
    # 对相机在第一个轴上的坐标进行排序，以便确定分割边界
    sorted_x_cameras = np.sort(x_cameras)

    for i in range(m):
        # 计算当前段的起始和结束索引
        start_index = int(i * segment_size_x) if i == 0 else int(i * segment_size_x) + 1
        end_index = int((i + 1) * segment_size_x) if i < m - 1 else V
        # 确定下界和上界
        lower_bound = x_min if i == 0 else sorted_x_cameras[start_index]
        upper_bound = x_max if i == m - 1 else sorted_x_cameras[end_index - 1]

        # 确保段之间连接正常（处理边界重叠或间隙）
        if i > 0 and lower_bound > x_segments[-1][1]:
            lower_bound = (x_segments[-1][1] + lower_bound) / 2
            x_segments[-1] = (x_segments[-1][0], lower_bound)

        x_segments.append((lower_bound, upper_bound))

    # 根据X轴分段筛选相机
    for x_segment in x_segments:
        mask_x = (x_cameras >= x_segment[0]) & (x_cameras <= x_segment[1])
        indices = np.where(mask_x)[0]
        tmp_cameras = [train_cameras[i] for i in indices]
        x_segments_cameras.append(tmp_cameras)

    partitions = {}
    # 步骤 5: 将每个X轴段沿Y轴（或第二个轴）进一步细分为n段
    m = 0
    partition_total_points = 0
    partition_total_cams = 0
    for x_segment, x_segments_camera in zip(x_segments, x_segments_cameras):
        segment_size_y = len(x_segments_camera) / n
        y_segments = []
        y_cameras = np.array([camera.camera_center.cpu().numpy() for camera in x_segments_camera])[:, plane_index[1]]
        y_cameras_num = len(y_cameras)
        sorted_y_cameras = np.sort(y_cameras)
        for i in range(n):
            start_index = int(i * segment_size_y) if i == 0 else int(i * segment_size_y) + 1
            end_index = int((i + 1) * segment_size_y) if i < n - 1 else y_cameras_num
            lower_bound = y_min if i == 0 else sorted_y_cameras[start_index]
            upper_bound = y_max if i == n - 1 else sorted_y_cameras[end_index - 1]

            # 确保段之间连接正常
            if i > 0 and lower_bound > y_segments[-1][1]:
                lower_bound = (y_segments[-1][1] + lower_bound) / 2
                y_segments[-1] = (y_segments[-1][0], lower_bound)

            y_segments.append((lower_bound, upper_bound))

        n = 0
        for y_segment in y_segments:
            # 提取该区域内的点云
            partition_ply = extract_point_cloud_from_bound(pcd, x_segment, y_segment, plane_index)
            # 提取该区域内的相机及其索引
            indices, partition_cams = extract_cams_from_bound(train_cameras, x_segment, y_segment, plane_index)
            # 存储分区信息
            partitions[f"{m}_{n}"] = {
                "bounds": (x_segment, y_segment),
                "pcd": partition_ply,
                "cameras": partition_cams,
                "indices": indices,
            }
            partition_points = partition_ply.points.shape[0]
            partition_cams = len(partition_cams)
            partition_total_points += partition_points
            partition_total_cams += partition_cams
            print(f"{m}_{n}: point num: {partition_points}")
            print(f"{m}_{n}: cameras num: {partition_cams}")
            n += 1
        m += 1

    # 打印统计信息
    origin_points = pcd.points.shape[0]
    origin_cams = len(train_cameras)
    print(f"{origin_points=}")
    print(f"{origin_cams=}")
    print(f"{partition_total_points=}")
    print(f"{partition_total_cams=}")
    print("###############################################################")

    return partitions

# 基于位置的数据选择与扩展函数（添加重叠区域）
def position_based_data_selection(partitions, pcd, train_cameras, threshold, plane_index):
    print("############### position_based_data_selection #################")
    points = pcd.points

    x_points = points[:, plane_index[0]]
    y_points = points[:, plane_index[1]]

    x_min, x_max = np.min(x_points), np.max(x_points)
    y_min, y_max = np.min(y_points), np.max(y_points)

    partition_total_points = 0
    partition_total_cams = 0
    for partition_id, partition in partitions.items():
        partition_x_bounds, partition_y_bounds = partition["bounds"]
        # 获取当前分区内相机的坐标范围
        partition_cams = np.array([camera.camera_center.cpu().numpy() for camera in partition["cameras"]])
        partition_x_cameras = partition_cams[:, plane_index[0]]
        partition_y_cameras = partition_cams[:, plane_index[1]]
        partition_x_min, partition_x_max = np.min(partition_x_cameras), np.max(partition_x_cameras)
        partition_y_min, partition_y_max = np.min(partition_y_cameras), np.max(partition_y_cameras)
        
        partition_x_width = partition_x_max - partition_x_min
        partition_y_height = partition_y_max - partition_y_min
        
        # 根据阈值（overlap_area）扩展边界框
        new_x_bounds = [
            min(partition_x_bounds[0], partition_x_min - threshold * partition_x_width),
            max(partition_x_bounds[1], partition_x_max + threshold * partition_x_width),
        ]
        new_y_bounds = [
            min(partition_y_bounds[0], partition_y_min - threshold * partition_y_height),
            max(partition_y_bounds[1], partition_y_max + threshold * partition_y_height),
        ]

        # 确保扩展后的边界不超出全局范围
        new_x_bounds[0] = max(new_x_bounds[0], x_min)
        new_x_bounds[1] = min(new_x_bounds[1], x_max)
        new_y_bounds[0] = max(new_y_bounds[0], y_min)
        new_y_bounds[1] = min(new_y_bounds[1], y_max)

        # 基于新边界重新提取点云和相机
        partition_ply = extract_point_cloud_from_bound(pcd, new_x_bounds, new_y_bounds, plane_index)
        indices, partition_cams = extract_cams_from_bound(train_cameras, new_x_bounds, new_y_bounds, plane_index)
        
        # 分离航拍相机和街景相机
        aerial_cams = [camera for camera in partition_cams if camera.image_type == "aerial"]
        street_cams = [camera for camera in partition_cams if camera.image_type != "aerial"]
        aerial_indices = np.array([indices[i] for i, camera in enumerate(partition_cams) if camera.image_type == "aerial"])
        street_indices = np.array([indices[i] for i, camera in enumerate(partition_cams) if camera.image_type != "aerial"])
        
        # 校验索引一致性
        assert aerial_cams == [train_cameras[idx] for idx in aerial_indices]
        # 更新分区数据
        partitions[partition_id] = {
            "true_bounds": partition["bounds"], # 原始边界
            "bounds": (new_x_bounds, new_y_bounds), # 扩展后的边界
            "pcd": partition_ply,
            "cameras": partition_cams,
            "aerial_cams": aerial_cams,
            "street_cams": street_cams,
            "indices": indices,
            "aerial_indices": aerial_indices,
            "street_indices": street_indices
        }
        partition_points_num = partition_ply.points.shape[0]
        partition_cams_num = len(partition_cams)
        partition_total_points += partition_points_num
        partition_total_cams += partition_cams_num
        print(f"{partition_id}: point num: {partition_points_num}")
        print(f"{partition_id}: cameras num: {partition_cams_num}")
    
    # 打印统计信息
    origin_points = pcd.points.shape[0]
    origin_cams = len(train_cameras)
    print(f"{origin_points=}")
    print(f"{origin_cams=}")
    print(f"{partition_total_points=}")
    print(f"{partition_total_cams=}")
    print("###############################################################")
    return partitions

# 基于可见性的相机选择和基于覆盖率的点云选择函数
def visibility_based_camera_and_coverage_based_point_selection(partitions, visible_rate):
    print("## visibility_based_camera_and_coverage_based_point_selection ##")
    new_partitions = {}
    partition_total_points = 0
    partition_total_cams = 0
    # 遍历每个分区作为目标分区 j
    for j_partition_id, j_partition in partitions.items():
        # 获取分区 j 点云的8个角点
        extent_8_corner_points = get_8_corner_points(j_partition["pcd"])
        total_partition_camera_count = 0
        j_collect_names = [camera.image_path for camera in j_partition["cameras"]]
        j_copy_cameras = copy.copy(j_partition["cameras"])
        j_indices = j_partition["indices"].tolist()
        new_points = []
        new_colors = []
        new_normals = []

        # 遍历其他分区 i
        for i_partition_id, i_partition in partitions.items():
            if i_partition_id == j_partition_id:
                continue
            pcd_j = i_partition["pcd"]
            append_camera_count = 0
            
            # 遍历分区 i 中的相机
            for idx, camera in enumerate(i_partition["cameras"]):
                proj_8_corner_points = {}

                # 基于可见性的相机选择：检查相机是否能看到分区 j 的角点
                # 空域感知可见性
                for key, point in extent_8_corner_points.items():
                    points_in_image, _, _ = point_in_image(camera, np.array([point]))
                    if len(points_in_image) == 0:
                        continue
                    proj_8_corner_points[key] = points_in_image[0]

                # 基于覆盖率的点云选择：如果可见角点少于3个（无法构成有效平面），跳过
                if len(list(proj_8_corner_points.values())) <= 3:
                    continue
                # 使用葛立恒扫描法（Graham scan）计算凸包，评估投影面积/覆盖率
                pkg = run_graham_scan(list(proj_8_corner_points.values()), camera.image_width, camera.image_height)

                # 如果相交率（intersection_rate）大于阈值
                if pkg["intersection_rate"] >= visible_rate:
                    if camera.image_path in j_collect_names:
                        continue
                    append_camera_count += 1
                    j_collect_names.append(camera.image_path)
                    j_copy_cameras.append(camera)
                    j_indices.append(i_partition["indices"][idx])

                    # 将该相机视野内的点云添加到分区 j
                    _, _, mask = point_in_image(camera, pcd_j.points)
                    updated_points, updated_colors, updated_normals = (
                        pcd_j.points[mask],
                        pcd_j.colors[mask],
                        pcd_j.normals[mask],
                    )
                    new_points.append(updated_points)
                    new_colors.append(updated_colors)
                    new_normals.append(updated_normals)
            total_partition_camera_count += append_camera_count

        # 合并原始点云和新添加的点云
        point_cloud = j_partition["pcd"]
        new_points.append(point_cloud.points)
        new_colors.append(point_cloud.colors)
        new_normals.append(point_cloud.normals)
        new_points = np.concatenate(new_points, axis=0)
        new_colors = np.concatenate(new_colors, axis=0)
        new_normals = np.concatenate(new_normals, axis=0)

        # 去重
        new_points, mask = np.unique(new_points, return_index=True, axis=0)
        new_colors = new_colors[mask]
        new_normals = new_normals[mask]
        
        # 统计并保存
        partition_points_num = new_points.shape[0]
        partition_cams_num = len(j_copy_cameras)
        partition_total_points += partition_points_num
        partition_total_cams += partition_cams_num
        print(f"{j_partition_id}: point num: {partition_points_num}")
        print(f"{j_partition_id}: cameras num: {partition_cams_num}")
        new_partitions[j_partition_id] = {
            "true_bounds": j_partition["true_bounds"],
            "bounds": j_partition["bounds"],
            "pcd": BasicPointCloud(points=new_points, colors=new_colors, normals=new_normals),
            "cameras": j_copy_cameras,
            "indices": j_indices,
        }
    print(f"{partition_total_points=}")
    print(f"{partition_total_cams=}")
    print("################################################################")
    return new_partitions

# 专门针对航拍和街景混合场景的可见性选择函数
def visibility_based_camera_and_coverage_based_point_selection_aerial_street(partitions, pcd, visible_rate):
    print("## visibility_based_camera_and_coverage_based_point_selection ##")
    new_partitions = {}
    partition_total_points = 0
    partition_total_cams = 0
    # 遍历每个分区作为目标分区 j
    for j_partition_id, j_partition in partitions.items():
        extent_8_corner_points = get_8_corner_points(j_partition["pcd"])
        total_partition_camera_count = 0
        j_collect_names = [camera.image_path for camera in j_partition["cameras"]]
        j_copy_cameras = copy.copy(j_partition["cameras"])
        j_indices = j_partition["indices"].tolist()
        new_points = []
        new_colors = []
        new_normals = []

        # 遍历其他分区 i
        for i_partition_id, i_partition in partitions.items():
            if i_partition_id == j_partition_id:
                continue
            pcd_j = i_partition["pcd"]
            append_camera_count = 0
            # 仅检查分区 i 中的航拍相机 (aerial_cams)
            for idx, camera in enumerate(i_partition["aerial_cams"]):
                proj_8_corner_points = {}

                # 检查可见性
                for key, point in extent_8_corner_points.items():
                    points_in_image, _, _ = point_in_image(camera, np.array([point]))
                    if len(points_in_image) == 0:
                        continue
                    proj_8_corner_points[key] = points_in_image[0]

                if len(list(proj_8_corner_points.values())) <= 3:
                    continue
                # 葛立恒扫描法计算覆盖率
                pkg = run_graham_scan(list(proj_8_corner_points.values()), camera.image_width, camera.image_height)

                if pkg["intersection_rate"] >= visible_rate:
                    if camera.image_path in j_collect_names:
                        continue
                    append_camera_count += 1
                    j_collect_names.append(camera.image_path)
                    j_copy_cameras.append(camera)
                    j_indices.append(i_partition["aerial_indices"][idx])

                    # 收集该相机视野下的点
                    _, _, mask = point_in_image(camera, pcd_j.points)
                    updated_points, updated_colors, updated_normals = (
                        pcd_j.points[mask],
                        pcd_j.colors[mask],
                        pcd_j.normals[mask],
                    )
                    new_points.append(updated_points)
                    new_colors.append(updated_colors)
                    new_normals.append(updated_normals)
                
            total_partition_camera_count += append_camera_count

        ## TODO: 添加街景点云 (如果存在街景相机)
        if len(j_partition["street_cams"]) > 0:
            updated_points, updated_colors = depth2pc_partition(j_partition["street_cams"])
            updated_normals = np.zeros_like(updated_points)
            
            new_points.append(updated_points)
            new_colors.append(updated_colors)
            new_normals.append(updated_normals)
        
        # 合并所有点云数据
        point_cloud = j_partition["pcd"]
        new_points.append(point_cloud.points)
        new_colors.append(point_cloud.colors)
        new_normals.append(point_cloud.normals)
        new_points = np.concatenate(new_points, axis=0)
        new_colors = np.concatenate(new_colors, axis=0)
        new_normals = np.concatenate(new_normals, axis=0)

        # 去重
        new_points, mask = np.unique(new_points, return_index=True, axis=0)
        new_colors = new_colors[mask]
        new_normals = new_normals[mask]
        
        # 统计并更新
        partition_points_num = new_points.shape[0]
        partition_cams_num = len(j_copy_cameras)
        partition_total_points += partition_points_num
        partition_total_cams += partition_cams_num
        print(f"{j_partition_id}: point num: {partition_points_num}")
        print(f"{j_partition_id}: cameras num: {partition_cams_num}")
        new_partitions[j_partition_id] = {
            "true_bounds": j_partition["true_bounds"],
            "bounds": j_partition["bounds"],
            "pcd": BasicPointCloud(points=new_points, colors=new_colors, normals=new_normals),
            "cameras": j_copy_cameras,
            "indices": j_indices,
        }
    print(f"{partition_total_points=}")
    print(f"{partition_total_cams=}")
    print("################################################################")
    return new_partitions

# 保存分区数据（目前主要针对Blender格式）
def save_partition_data(partitions, ckpt_path, logfolder, m_region, n_region, frames):
    
    for m in range(m_region):
        for n in range(n_region):
            # 创建分区文件夹
            partition_path = os.path.join(logfolder, f"{m}_{n}")
            os.makedirs(partition_path, exist_ok=True)
            partition_id = f"{m}_{n}"
            partition = partitions[partition_id]
            # 保存点云PLY文件，颜色限制在0-255并转为uint8
            pcd_colors = np.clip(partition["pcd"].colors*255., 0, 255).astype(np.uint8)
            storePly(os.path.join(partition_path, f"points3d.ply"), partition["pcd"].points, pcd_colors)
            
            # 保存相机参数transforms.json
            select_frames = [frames[idx] for idx in partition["indices"]]
            # 校验路径一致性
            assert [cam.image_path for cam in partition["cameras"]] == [os.path.join(dp.source_path, frame['file_path']) for frame in select_frames]
            save_frames = copy.deepcopy(select_frames)
            # 转换为绝对路径
            for i, frame in enumerate(save_frames):
                save_frames[i]["file_path"] = os.path.abspath(os.path.join(dp.source_path, frame["file_path"]))
                save_frames[i]["depth_path"] = os.path.abspath(os.path.join(dp.source_path, frame["depth_path"]))
            save_json(os.path.join(partition_path, "transforms.json"), save_frames) 
    
    # 清理大对象以便保存轻量级checkpoint
    for key in partitions.keys():
        partitions[key].pop("cameras")
        partitions[key].pop("pcd")
        partitions[key].pop("indices")
    torch.save(partitions, ckpt_path)

# 执行渐进式分区的函数：划分 -> 选择/扩展 -> 可见性筛选 -> 保存
def run_progressive_partition(dp, pcd, train_cameras, m_region, n_region, plane_index, labels, logfolder, frames):
    ckpt_path = os.path.join(
            logfolder, f"init_ply_coverage_{m_region*n_region}parts_{dp.visible_rate}.th"
        )
    print("try to create new partitions.")
    # 1. 初始划分
    partitions = camera_position_based_region_division(pcd, train_cameras, m_region, n_region, plane_index)
    draw_partitions(partitions, "camera_position_based_region_division",labels, plane_index, logfolder)
    # 2. 基于位置的重叠/扩展
    partitions = position_based_data_selection(partitions, pcd, train_cameras, dp.overlap_area, plane_index)
    draw_partitions(partitions, "position_based_data_selection",labels, plane_index, logfolder)
    # 3. 基于可见性的筛选
    partitions = visibility_based_camera_and_coverage_based_point_selection_aerial_street(partitions, pcd, dp.visible_rate)
    draw_each_partition(partitions, "visibility_based_camera_and_coverage_based_point_selection", plane_index, logfolder)
    # 4. 保存结果
    save_partition_data(partitions, ckpt_path, logfolder, m_region, n_region, frames)
    print(f"save partitions in {ckpt_path}")
    
# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="cluster poses and pcd")

    parser.add_argument('--config', type=str, help='partition config file path') # 分区配置文件路径
    parser.add_argument('--chunk_size', default=2, type=float,help='1 means 100 meters in matrixicty') # 块大小
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # 加载YAML配置文件
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        dp = parse_cfg_dp(cfg) # 解析数据处理配置
        # 创建命名空间对象lp，用于存储加载参数
        lp = SimpleNamespace()
        lp.model_config = dp.model_config
        lp.pretrained_checkpoint = ""
        lp.global_appearance = ""
        lp.dataset_name = dp.dataset_name
        lp.scene_name = dp.scene_name
        lp.images = dp.images
        lp.resolution = dp.resolution
        lp.white_background = dp.white_background
        lp.random_background = dp.random_background
        lp.resolution_scales = dp.resolution_scales
        lp.data_device = dp.data_device
        lp.eval = dp.eval
        lp.ratio = dp.ratio
        lp.data_format = dp.data_format
        lp.add_mask = dp.add_mask
        lp.add_depth = dp.add_depth
        lp.add_aerial = dp.add_aerial
        lp.add_street = dp.add_street
        # 针对Colmap或City数据的特殊处理
        if lp.data_format == "colmap" or lp.data_format == "city":
            lp.llffhold = dp.llffhold
        if lp.data_format == "blender" or lp.data_format == "city":
            lp.scale = dp.scale
            lp.center = dp.center

        dp.config_path = args.config
    
    dp.add_mask = False
    # 加载场景信息
    scene_info = load_data(dp)    
    # 生成训练相机列表
    training_cams = cameraList_from_camInfos(scene_info.train_cameras, 1, dp, torch.zeros(3, dtype=torch.float32, device="cpu"))
    training_cams = sorted(training_cams, key = lambda x : x.image_path)
    pcd = scene_info.point_cloud
    # 如果设置了ratio，进行点云降采样
    if dp.ratio > 1:
        pcd = pcd._replace(points=pcd.points[::dp.ratio])
        pcd = pcd._replace(colors=pcd.colors[::dp.ratio])
        pcd = pcd._replace(normals=pcd.normals[::dp.ratio])

    # 如果启用了分区
    if dp.partition: 
        logfolder = os.path.join(dp.source_path, "chunks")
        os.makedirs(logfolder, exist_ok=True)
        
        # 读取原始相机参数
        json_file_path = os.path.join(dp.source_path, "transforms_train.json")
        _, _, frames = read_camera_parameters(json_file_path)
        frames = sorted(frames, key = lambda x : x['file_path'])
        # 校验相机列表一致性
        assert [cam.image_path for cam in training_cams] == [os.path.join(dp.source_path, frame['file_path']) for frame in frames]
        
        # 确定用于划分的平面轴索引（例如 x,y 轴）
        plane_index = [index for index, value in enumerate(dp.xyz_plane) if value == 1]
        assert len(plane_index) == 2

        labels = ["X-axis", "Y-axis", "Z-axis"]
        print(f"plane is constructed by {labels[plane_index[0]]} and {labels[plane_index[1]]}")
        
        # 确定分区数量 m * n
        if dp.partition_type == 'num':
            m_region = dp.n_width
            n_region = dp.n_height
        elif dp.partition_type == 'size':
            # 如果是按尺寸划分，计算全局包围盒并根据 chunk_size 计算 m, n
            cam_centers = np.array([camera.camera_center.cpu().numpy() for camera in training_cams])
            global_bbox = np.stack([cam_centers.min(axis=0), cam_centers.max(axis=0)])
            # 扩展边界
            global_bbox[0, :2] -= args.overlap_area * args.chunk_size
            global_bbox[1, :2] += args.overlap_area * args.chunk_size
            extent = global_bbox[1] - global_bbox[0]
            padd = np.array([args.chunk_size - extent[0] % args.chunk_size, args.chunk_size - extent[1] % args.chunk_size])
            global_bbox[0, :2] -= padd / 2
            global_bbox[1, :2] += padd / 2
            
            # 设置Z轴无限大，因为通常只在XY平面划分
            global_bbox[0, 2] = -1e12
            global_bbox[1, 2] = 1e12

            excluded_chunks = []
            chunks_pcd = {}

            extent = global_bbox[1] - global_bbox[0]
            n_width = round(extent[0] / args.chunk_size)
            n_height = round(extent[1] / args.chunk_size)
        else:
            raise ValueError(f"Unknown partition type: {args.partition_type}")

        # 运行分区逻辑
        run_progressive_partition(dp, pcd, training_cams, m_region, n_region, plane_index, labels, logfolder, frames)
        print("partition successfully")
    
    # 针对 GaussianLoDModel 模型的 LoD（细节层次）计算
    if lp.model_config["name"] == "GaussianLoDModel":
        points = torch.tensor(pcd.points).float().cuda()
        # 处理不同格式的中心和缩放
        if dp.data_format != "colmap" and dp.data_format != "ucgs":
            center = torch.tensor(dp.center).float().cuda()
            scale = dp.scale
        else:
            center = torch.tensor([0,0,0]).float().cuda()
            scale = 1.0
        # 归一化点云
        points = (points-center)/scale

        aerial_dist = torch.tensor([]).cuda()
        street_dist = torch.tensor([]).cuda()
        dist_ratio = dp.dist_ratio
        dist_ratio = 0.9
        fork = lp.model_config["kwargs"]["fork"]
        
        print("Calculating distance statistics...")
        
        debug_aerial_count = 0
        debug_street_count = 0

        for cam in training_cams:
            cam_center = (cam.camera_center-center)/scale
            dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
            
            dist_max = torch.quantile(dist, dist_ratio)
            dist_min = torch.quantile(dist, 1 - dist_ratio)
            new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
            
            name_lower = cam.image_name.lower()
            
            # [修复点 1] 关键词修正
            if "aerial" in name_lower:
                aerial_dist = torch.cat((aerial_dist, new_dist), dim=0)
                debug_aerial_count += 1
            elif "street" in name_lower: 
                street_dist = torch.cat((street_dist, new_dist), dim=0)
                debug_street_count += 1
            else:
                # 兜底：既没有aerial也没有street，归为aerial
                # print(f"Unclassified: {name_lower}")
                aerial_dist = torch.cat((aerial_dist, new_dist), dim=0)
                debug_aerial_count += 1
        print(f"Stats: Found {debug_aerial_count} aerials, {debug_street_count} streets.")
        # 计算全局的距离范围
        aerial_dist_max = torch.quantile(aerial_dist, dist_ratio)
        aerial_dist_min = torch.quantile(aerial_dist, 1 - dist_ratio)
        street_dist_max = torch.quantile(street_dist, dist_ratio)
        street_dist_min = torch.quantile(street_dist, 1 - dist_ratio)
        # breakpoint() # 调试断点

        # 根据配置设置 LoD 层级
        if dp.aerial_lod == "single":
            lp.model_config["kwargs"]["standard_dist"] = aerial_dist_min.item()
            lp.model_config["kwargs"]["aerial_levels"] = 1
            if dp.street_lod == "single":
                lp.model_config["kwargs"]["street_levels"] = 2
            else:
                # 计算街景所需的层级数 (log2 based)
                lp.model_config["kwargs"]["street_levels"] = torch.floor(torch.log2(aerial_dist_min/street_dist_min)/math.log2(fork)).int().item() + 1 
        else:
            lp.model_config["kwargs"]["standard_dist"] = aerial_dist_max.item()
            # 计算航拍和街景所需的层级数
            lp.model_config["kwargs"]["aerial_levels"] = torch.floor(torch.log2(aerial_dist_max/aerial_dist_min)/math.log2(fork)).int().item() + 1
            lp.model_config["kwargs"]["street_levels"] = torch.floor(torch.log2(aerial_dist_max/street_dist_min)/math.log2(fork)).int().item() + 1 

    # 生成最终的配置文件
    if dp.partition:
        generate_chunks_config(dp, lp)
    else:
        generate_config(dp, lp)