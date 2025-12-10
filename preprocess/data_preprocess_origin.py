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
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from preprocess.generate_chunks_config import generate_chunks_config
from preprocess.generate_config import generate_config
from preprocess.depth2pc import depth2pc, depth2pc_partition
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from utils.camera_utils import cameraList_from_camInfos
from utils.partition_utils import *
from utils.general_utils import *
from types import SimpleNamespace

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

def load_data(args):
    center = [0, 0, 0]
    scale = 1.0
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

def camera_position_based_region_division(pcd, train_cameras, m_region, n_region, plane_index):
    print("############ camera_position_based_region_division ############")
    m, n = m_region, n_region
    points = pcd.points
    
    cameras = np.array([camera.camera_center.cpu().numpy() for camera in train_cameras])

    # Step 1: Project Points and Cameras onto x/z Plane
    x_points = points[:, plane_index[0]]
    y_points = points[:, plane_index[1]]
    x_cameras = cameras[:, plane_index[0]]
    
    # Step 2: Determine the Bounding Box
    x_min, x_max = np.min(x_points), np.max(x_points)
    y_min, y_max = np.min(y_points), np.max(y_points)

    # Step 3: Count total cameras
    V = len(train_cameras)

    # Step 4: Divide the bbox into m segments along the x-axis
    segment_size_x = V / m
    x_segments = []
    x_segments_cameras = []
    
    # Sort cameras to determine boundaries
    sorted_x_cameras = np.sort(x_cameras)

    for i in range(m):
        start_index = int(i * segment_size_x) if i == 0 else int(i * segment_size_x) + 1
        end_index = int((i + 1) * segment_size_x) if i < m - 1 else V
        lower_bound = x_min if i == 0 else sorted_x_cameras[start_index]
        upper_bound = x_max if i == m - 1 else sorted_x_cameras[end_index - 1]

        # Ensure the segments connect properly
        if i > 0 and lower_bound > x_segments[-1][1]:
            lower_bound = (x_segments[-1][1] + lower_bound) / 2
            x_segments[-1] = (x_segments[-1][0], lower_bound)

        x_segments.append((lower_bound, upper_bound))

    for x_segment in x_segments:
        mask_x = (x_cameras >= x_segment[0]) & (x_cameras <= x_segment[1])
        indices = np.where(mask_x)[0]
        tmp_cameras = [train_cameras[i] for i in indices]
        x_segments_cameras.append(tmp_cameras)

    partitions = {}
    # Step 5: Each x segment is further subdivided into n segments along the y-axis
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

            # Ensure the segments connect properly
            if i > 0 and lower_bound > y_segments[-1][1]:
                lower_bound = (y_segments[-1][1] + lower_bound) / 2
                y_segments[-1] = (y_segments[-1][0], lower_bound)

            y_segments.append((lower_bound, upper_bound))

        n = 0
        for y_segment in y_segments:
            partition_ply = extract_point_cloud_from_bound(pcd, x_segment, y_segment, plane_index)
            indices, partition_cams = extract_cams_from_bound(train_cameras, x_segment, y_segment, plane_index)
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

    origin_points = pcd.points.shape[0]
    origin_cams = len(train_cameras)
    print(f"{origin_points=}")
    print(f"{origin_cams=}")
    print(f"{partition_total_points=}")
    print(f"{partition_total_cams=}")
    print("###############################################################")

    return partitions

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
        partition_cams = np.array([camera.camera_center.cpu().numpy() for camera in partition["cameras"]])
        partition_x_cameras = partition_cams[:, plane_index[0]]
        partition_y_cameras = partition_cams[:, plane_index[1]]
        partition_x_min, partition_x_max = np.min(partition_x_cameras), np.max(partition_x_cameras)
        partition_y_min, partition_y_max = np.min(partition_y_cameras), np.max(partition_y_cameras)
        
        partition_x_width = partition_x_max - partition_x_min
        partition_y_height = partition_y_max - partition_y_min
        
        new_x_bounds = [
            min(partition_x_bounds[0], partition_x_min - threshold * partition_x_width),
            max(partition_x_bounds[1], partition_x_max + threshold * partition_x_width),
        ]
        new_y_bounds = [
            min(partition_y_bounds[0], partition_y_min - threshold * partition_y_height),
            max(partition_y_bounds[1], partition_y_max + threshold * partition_y_height),
        ]

        new_x_bounds[0] = max(new_x_bounds[0], x_min)
        new_x_bounds[1] = min(new_x_bounds[1], x_max)
        new_y_bounds[0] = max(new_y_bounds[0], y_min)
        new_y_bounds[1] = min(new_y_bounds[1], y_max)

        partition_ply = extract_point_cloud_from_bound(pcd, new_x_bounds, new_y_bounds, plane_index)
        indices, partition_cams = extract_cams_from_bound(train_cameras, new_x_bounds, new_y_bounds, plane_index)
        aerial_cams = [camera for camera in partition_cams if camera.image_type == "aerial"]
        street_cams = [camera for camera in partition_cams if camera.image_type != "aerial"]
        aerial_indices = np.array([indices[i] for i, camera in enumerate(partition_cams) if camera.image_type == "aerial"])
        street_indices = np.array([indices[i] for i, camera in enumerate(partition_cams) if camera.image_type != "aerial"])
        
        assert aerial_cams == [train_cameras[idx] for idx in aerial_indices]
        partitions[partition_id] = {
            "true_bounds": partition["bounds"],
            "bounds": (new_x_bounds, new_y_bounds),
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
    origin_points = pcd.points.shape[0]
    origin_cams = len(train_cameras)
    print(f"{origin_points=}")
    print(f"{origin_cams=}")
    print(f"{partition_total_points=}")
    print(f"{partition_total_cams=}")
    print("###############################################################")
    return partitions

def visibility_based_camera_and_coverage_based_point_selection(partitions, visible_rate):
    print("## visibility_based_camera_and_coverage_based_point_selection ##")
    new_partitions = {}
    partition_total_points = 0
    partition_total_cams = 0
    for j_partition_id, j_partition in partitions.items():
        extent_8_corner_points = get_8_corner_points(j_partition["pcd"])
        total_partition_camera_count = 0
        j_collect_names = [camera.image_path for camera in j_partition["cameras"]]
        j_copy_cameras = copy.copy(j_partition["cameras"])
        j_indices = j_partition["indices"].tolist()
        new_points = []
        new_colors = []
        new_normals = []

        for i_partition_id, i_partition in partitions.items():
            if i_partition_id == j_partition_id:
                continue
            pcd_j = i_partition["pcd"]
            append_camera_count = 0
            # Visibility_based_camera_selection
            for idx, camera in enumerate(i_partition["cameras"]):
                proj_8_corner_points = {}

                # Visibility_based_camera_selection
                # airspace-aware visibility
                for key, point in extent_8_corner_points.items():
                    points_in_image, _, _ = point_in_image(camera, np.array([point]))
                    if len(points_in_image) == 0:
                        continue
                    proj_8_corner_points[key] = points_in_image[0]

                # coverage-based point selection
                if len(list(proj_8_corner_points.values())) <= 3:
                    continue
                pkg = run_graham_scan(list(proj_8_corner_points.values()), camera.image_width, camera.image_height)

                # pkg = run_graham_scan(points_in_image, camera.image_width, camera.image_height)
                if pkg["intersection_rate"] >= visible_rate:
                    if camera.image_path in j_collect_names:
                        continue
                    append_camera_count += 1
                    j_collect_names.append(camera.image_path)
                    j_copy_cameras.append(camera)
                    j_indices.append(i_partition["indices"][idx])

                    # Coverage-based point selection
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

        point_cloud = j_partition["pcd"]
        new_points.append(point_cloud.points)
        new_colors.append(point_cloud.colors)
        new_normals.append(point_cloud.normals)
        new_points = np.concatenate(new_points, axis=0)
        new_colors = np.concatenate(new_colors, axis=0)
        new_normals = np.concatenate(new_normals, axis=0)

        new_points, mask = np.unique(new_points, return_index=True, axis=0)
        new_colors = new_colors[mask]
        new_normals = new_normals[mask]
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

def visibility_based_camera_and_coverage_based_point_selection_aerial_street(partitions, pcd, visible_rate):
    print("## visibility_based_camera_and_coverage_based_point_selection ##")
    new_partitions = {}
    partition_total_points = 0
    partition_total_cams = 0
    for j_partition_id, j_partition in partitions.items():
        extent_8_corner_points = get_8_corner_points(j_partition["pcd"])
        total_partition_camera_count = 0
        j_collect_names = [camera.image_path for camera in j_partition["cameras"]]
        j_copy_cameras = copy.copy(j_partition["cameras"])
        j_indices = j_partition["indices"].tolist()
        new_points = []
        new_colors = []
        new_normals = []

        for i_partition_id, i_partition in partitions.items():
            if i_partition_id == j_partition_id:
                continue
            pcd_j = i_partition["pcd"]
            append_camera_count = 0
            # Visibility_based_camera_selection
            # for idx, camera in enumerate(i_partition["cameras"]):
            for idx, camera in enumerate(i_partition["aerial_cams"]):
                proj_8_corner_points = {}

                # Visibility_based_camera_selection
                for key, point in extent_8_corner_points.items():
                    points_in_image, _, _ = point_in_image(camera, np.array([point]))
                    if len(points_in_image) == 0:
                        continue
                    proj_8_corner_points[key] = points_in_image[0]

                if len(list(proj_8_corner_points.values())) <= 3:
                    continue
                pkg = run_graham_scan(list(proj_8_corner_points.values()), camera.image_width, camera.image_height)

                # pkg = run_graham_scan(points_in_image, camera.image_width, camera.image_height)
                if pkg["intersection_rate"] >= visible_rate:
                    if camera.image_path in j_collect_names:
                        # print("skip")
                        continue
                    append_camera_count += 1
                    j_collect_names.append(camera.image_path)
                    j_copy_cameras.append(camera)
                    j_indices.append(i_partition["aerial_indices"][idx])

                    # Coverage-based point selection
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

        ## TODO: add street point cloud
        if len(j_partition["street_cams"]) > 0:
            updated_points, updated_colors = depth2pc_partition(j_partition["street_cams"])
            updated_normals = np.zeros_like(updated_points)
            
            new_points.append(updated_points)
            new_colors.append(updated_colors)
            new_normals.append(updated_normals)
        
        point_cloud = j_partition["pcd"]
        new_points.append(point_cloud.points)
        new_colors.append(point_cloud.colors)
        new_normals.append(point_cloud.normals)
        new_points = np.concatenate(new_points, axis=0)
        new_colors = np.concatenate(new_colors, axis=0)
        new_normals = np.concatenate(new_normals, axis=0)

        new_points, mask = np.unique(new_points, return_index=True, axis=0)
        new_colors = new_colors[mask]
        new_normals = new_normals[mask]
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

# only for blender format now
def save_partition_data(partitions, ckpt_path, logfolder, m_region, n_region, frames):
    
    for m in range(m_region):
        for n in range(n_region):
            partition_path = os.path.join(logfolder, f"{m}_{n}")
            os.makedirs(partition_path, exist_ok=True)
            partition_id = f"{m}_{n}"
            partition = partitions[partition_id]
            # save ply
            pcd_colors = np.clip(partition["pcd"].colors*255., 0, 255).astype(np.uint8)
            storePly(os.path.join(partition_path, f"points3d.ply"), partition["pcd"].points, pcd_colors)
            
            # save cameras
            select_frames = [frames[idx] for idx in partition["indices"]]
            assert [cam.image_path for cam in partition["cameras"]] == [os.path.join(dp.source_path, frame['file_path']) for frame in select_frames]
            save_frames = copy.deepcopy(select_frames)
            for i, frame in enumerate(save_frames):
                save_frames[i]["file_path"] = os.path.abspath(os.path.join(dp.source_path, frame["file_path"]))
                save_frames[i]["depth_path"] = os.path.abspath(os.path.join(dp.source_path, frame["depth_path"]))
            save_json(os.path.join(partition_path, "transforms.json"), save_frames) 
    
    for key in partitions.keys():
        partitions[key].pop("cameras")
        partitions[key].pop("pcd")
        partitions[key].pop("indices")
    torch.save(partitions, ckpt_path)

def run_progressive_partition(dp, pcd, train_cameras, m_region, n_region, plane_index, labels, logfolder, frames):
    ckpt_path = os.path.join(
            logfolder, f"init_ply_coverage_{m_region*n_region}parts_{dp.visible_rate}.th"
        )
    print("try to create new partitions.")
    partitions = camera_position_based_region_division(pcd, train_cameras, m_region, n_region, plane_index)
    draw_partitions(partitions, "camera_position_based_region_division",labels, plane_index, logfolder)
    partitions = position_based_data_selection(partitions, pcd, train_cameras, dp.overlap_area, plane_index)
    draw_partitions(partitions, "position_based_data_selection",labels, plane_index, logfolder)
    partitions = visibility_based_camera_and_coverage_based_point_selection_aerial_street(partitions, pcd, dp.visible_rate)
    draw_each_partition(partitions, "visibility_based_camera_and_coverage_based_point_selection", plane_index, logfolder)
    save_partition_data(partitions, ckpt_path, logfolder, m_region, n_region, frames)
    print(f"save partitions in {ckpt_path}")
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="cluster poses and pcd")

    parser.add_argument('--config', type=str, help='partition config file path')
    parser.add_argument('--chunk_size', default=2, type=float,help='1 means 100 meters in matrixicty') 
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        dp = parse_cfg_dp(cfg)
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
        if lp.data_format == "colmap" or lp.data_format == "city":
            lp.llffhold = dp.llffhold
        if lp.data_format == "blender" or lp.data_format == "city":
            lp.scale = dp.scale
            lp.center = dp.center

        dp.config_path = args.config
    
    dp.add_mask = False
    scene_info = load_data(dp)    
    training_cams = cameraList_from_camInfos(scene_info.train_cameras, 1, dp, torch.zeros(3, dtype=torch.float32, device="cpu"))
    training_cams = sorted(training_cams, key = lambda x : x.image_path)
    pcd = scene_info.point_cloud
    if dp.ratio > 1:
        pcd = pcd._replace(points=pcd.points[::dp.ratio])
        pcd = pcd._replace(colors=pcd.colors[::dp.ratio])
        pcd = pcd._replace(normals=pcd.normals[::dp.ratio])

    if dp.partition: 
        logfolder = os.path.join(dp.source_path, "chunks")
        os.makedirs(logfolder, exist_ok=True)
        
        json_file_path = os.path.join(dp.source_path, "transforms_train.json")
        _, _, frames = read_camera_parameters(json_file_path)
        frames = sorted(frames, key = lambda x : x['file_path'])
        assert [cam.image_path for cam in training_cams] == [os.path.join(dp.source_path, frame['file_path']) for frame in frames]
        
        plane_index = [index for index, value in enumerate(dp.xyz_plane) if value == 1]
        assert len(plane_index) == 2

        labels = ["X-axis", "Y-axis", "Z-axis"]
        print(f"plane is constructed by {labels[plane_index[0]]} and {labels[plane_index[1]]}")
        
        if dp.partition_type == 'num':
            m_region = dp.n_width
            n_region = dp.n_height
        elif dp.partition_type == 'size':
            cam_centers = np.array([camera.camera_center.cpu().numpy() for camera in training_cams])
            global_bbox = np.stack([cam_centers.min(axis=0), cam_centers.max(axis=0)])
            global_bbox[0, :2] -= args.overlap_area * args.chunk_size
            global_bbox[1, :2] += args.overlap_area * args.chunk_size
            extent = global_bbox[1] - global_bbox[0]
            padd = np.array([args.chunk_size - extent[0] % args.chunk_size, args.chunk_size - extent[1] % args.chunk_size])
            global_bbox[0, :2] -= padd / 2
            global_bbox[1, :2] += padd / 2
            
            global_bbox[0, 2] = -1e12
            global_bbox[1, 2] = 1e12

            excluded_chunks = []
            chunks_pcd = {}

            extent = global_bbox[1] - global_bbox[0]
            n_width = round(extent[0] / args.chunk_size)
            n_height = round(extent[1] / args.chunk_size)
        else:
            raise ValueError(f"Unknown partition type: {args.partition_type}")

        run_progressive_partition(dp, pcd, training_cams, m_region, n_region, plane_index, labels, logfolder, frames)
        print("partition successfully")
    
    if lp.model_config["name"] == "GaussianLoDModel":
        points = torch.tensor(pcd.points).float().cuda()
        if dp.data_format != "colmap" and dp.data_format != "ucgs":
            center = torch.tensor(dp.center).float().cuda()
            scale = dp.scale
        else:
            center = torch.tensor([0,0,0]).float().cuda()
            scale = 1.0
        points = (points-center)/scale

        aerial_dist = torch.tensor([]).cuda()
        street_dist = torch.tensor([]).cuda()
        dist_ratio = dp.dist_ratio
        dist_ratio = 0.9
        fork = lp.model_config["kwargs"]["fork"]
        for cam in training_cams:
            cam_center = (cam.camera_center-center)/scale
            
            dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
            # breakpoint()
            dist_max = torch.quantile(dist, dist_ratio)
            dist_min = torch.quantile(dist, 1 - dist_ratio)
            new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
            if cam.image_type == "aerial":
                aerial_dist = torch.cat((aerial_dist, new_dist), dim=0)
            elif cam.image_type == "street":
                street_dist = torch.cat((street_dist, new_dist), dim=0)
        aerial_dist_max = torch.quantile(aerial_dist, dist_ratio)
        aerial_dist_min = torch.quantile(aerial_dist, 1 - dist_ratio)
        street_dist_max = torch.quantile(street_dist, dist_ratio)
        street_dist_min = torch.quantile(street_dist, 1 - dist_ratio)
        breakpoint()
        if dp.aerial_lod == "single":
            lp.model_config["kwargs"]["standard_dist"] = aerial_dist_min.item()
            lp.model_config["kwargs"]["aerial_levels"] = 1
            if dp.street_lod == "single":
                lp.model_config["kwargs"]["street_levels"] = 2
            else:
                lp.model_config["kwargs"]["street_levels"] = torch.floor(torch.log2(aerial_dist_min/street_dist_min)/math.log2(fork)).int().item() + 1 
        else:
            lp.model_config["kwargs"]["standard_dist"] = aerial_dist_max.item()
            lp.model_config["kwargs"]["aerial_levels"] = torch.floor(torch.log2(aerial_dist_max/aerial_dist_min)/math.log2(fork)).int().item() + 1
            lp.model_config["kwargs"]["street_levels"] = torch.floor(torch.log2(aerial_dist_max/street_dist_min)/math.log2(fork)).int().item() + 1 

    if dp.partition:
        generate_chunks_config(dp, lp)
    else:
        generate_config(dp, lp)