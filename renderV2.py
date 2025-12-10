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
import sys
import imageio
import yaml
from os import makedirs
import torch
import numpy as np

import subprocess
# 构建查询GPU显存使用情况的命令：查询显存信息，筛选GPU行，筛选Used行
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# 执行命令并获取输出结果，按换行符分割
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# 找到显存使用量最小的GPU ID，并设置为当前可见的CUDA设备
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# 打印当前使用的GPU编号
os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state, parse_cfg, visualize_depth, visualize_normal
from utils.image_utils import save_rgba
from argparse import ArgumentParser

def render_set(model_path, name, iteration, views, gaussians, pipe, background, add_aerial, add_street):
    """
    渲染指定的数据集（如train或test集）。
    参数:
        model_path: 模型保存路径
        name: 数据集名称 (如 "train", "test")
        iteration: 当前迭代次数
        views: 相机视角列表
        gaussians: 高斯模型对象
        pipe: 渲染管线参数
        background: 背景颜色
        add_aerial: 是否包含航拍数据
        add_street: 是否包含街景数据
    """
    vis_normal=False
    vis_depth=False
    # 如果高斯属性是2D（通常指2D Gaussian或Surfel变体），则开启法线和深度可视化
    if gaussians.gs_attr == "2D":
        vis_normal=True
        vis_depth=True
        
    # 如果包含航拍数据，设置相关输出路径
    if add_aerial:
        aerial_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "renders")
        aerial_error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "errors")
        aerial_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "gt")
        # 创建目录，如果存在则忽略
        makedirs(aerial_render_path, exist_ok=True)
        makedirs(aerial_error_path, exist_ok=True)
        makedirs(aerial_gts_path, exist_ok=True)
        
        # 如果需要可视化法线，创建对应目录
        if vis_normal:
            aerial_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "normal")
            makedirs(aerial_normal_path, exist_ok=True)
        # 如果需要可视化深度，创建对应目录
        if vis_depth:
            aerial_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "depth")
            makedirs(aerial_depth_path, exist_ok=True)
    
    # 如果包含街景数据，设置相关输出路径（逻辑同上）
    if add_street:
        street_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "renders")
        street_error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "errors")
        street_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "gt")
        makedirs(street_render_path, exist_ok=True)
        makedirs(street_error_path, exist_ok=True)
        makedirs(street_gts_path, exist_ok=True)
        
        if vis_normal:
            street_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "normal")
            makedirs(street_normal_path, exist_ok=True)
        if vis_depth:
            street_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "depth")
            makedirs(street_depth_path, exist_ok=True)

    # 动态导入 gaussian_renderer 模块
    modules = __import__('gaussian_renderer')

    # -------- 渲染街景视图 --------
    street_t_list = []
    street_visible_count_list = []
    street_per_view_dict = {}
    # 筛选出类型为 "street" 的视角
    street_views = [view for view in views if view.image_type=="street"]
    # 遍历所有街景视角，显示进度条
    for idx, view in enumerate(tqdm(street_views, desc="Street rendering progress")):
        
        # 同步CUDA并记录渲染开始时间
        torch.cuda.synchronize();t_start = time.time()
        # 调用渲染函数
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        # 同步CUDA并记录渲染结束时间
        torch.cuda.synchronize();t_end = time.time()
        
        # 记录耗时
        street_t_list.append(t_end - t_start)

        # renders: 获取渲染图像，并限制数值范围在 [0, 1]
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        # 获取参与渲染的可见高斯点数量
        visible_count = render_pkg["visibility_filter"].sum()

        # gts: 获取原始真值图像并移动到GPU
        gt = view.original_image.cuda()
        # 获取Alpha掩码并移动到GPU
        alpha_mask = view.alpha_mask.cuda()
        # 将Alpha通道拼接到渲染结果和真值图像上 (RGB -> RGBA)
        rendering = torch.cat([rendering, alpha_mask], dim=0)
        gt = torch.cat([gt, alpha_mask], dim=0)
        
        # error maps: 计算L1误差图
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()
        
        # 如果需要可视化法线
        if vis_normal == True:
            normal_map = render_pkg['render_normals'][0].detach()
            # 转换法线图为可视化格式
            vis_normal_map = visualize_normal(normal_map, view)
            # 处理Alpha掩码以匹配格式
            vis_alpha_mask = ((alpha_mask * 255).byte()).permute(1, 2, 0).cpu().numpy()
            # 将法线图与Alpha掩码拼接
            vis_normal_map = np.concatenate((vis_normal_map,vis_alpha_mask),axis=2)
            # 保存法线图
            imageio.imwrite(os.path.join(street_normal_path, '{0:05d}'.format(idx) + ".png"), vis_normal_map)
        
        # 如果需要可视化深度
        if vis_depth == True:
            depth_map = render_pkg["render_depth"]
            # 转换深度图为可视化格式
            vis_depth_map = visualize_depth(depth_map) 
            # 拼接Alpha通道
            vis_depth_map = torch.concat([vis_depth_map,alpha_mask],dim=0)
            # 保存深度图
            torchvision.utils.save_image(vis_depth_map, os.path.join(street_depth_path, '{0:05d}'.format(idx) + ".png"))

        # 保存渲染结果、误差图和真值图
        save_rgba(rendering, os.path.join(street_render_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(errormap, os.path.join(street_error_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(gt, os.path.join(street_gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # 记录统计信息
        street_visible_count_list.append(visible_count)
        street_per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    # 如果有街景数据，保存每张图的可见点统计
    if len(street_views) > 0:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "street", "per_view_count.json"), 'w') as fp:
            json.dump(street_per_view_dict, fp, indent=True)
    
    # -------- 渲染航拍视图 (逻辑与街景部分基本一致) --------
    aerial_t_list = []
    aerial_visible_count_list = []
    aerial_per_view_dict = {}
    # 筛选出类型为 "aerial" 的视角
    aerial_views = [view for view in views if view.image_type=="aerial"]
    for idx, view in enumerate(tqdm(aerial_views, desc="Aerial rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        torch.cuda.synchronize();t_end = time.time()

        aerial_t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()

        # gts
        gt = view.original_image.cuda()
        alpha_mask = view.alpha_mask.cuda()
        rendering = torch.cat([rendering, alpha_mask], dim=0)
        gt = torch.cat([gt, alpha_mask], dim=0)
        
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()
        
        if vis_normal == True:
            normal_map = render_pkg['render_normals'][0] 
            vis_normal_map = visualize_normal(normal_map, view)
            vis_alpha_mask = ((alpha_mask * 255).byte()).permute(1, 2, 0).cpu().numpy()
            vis_normal_map = np.concatenate((vis_normal_map,vis_alpha_mask),axis=2)
            imageio.imwrite(os.path.join(aerial_normal_path, '{0:05d}'.format(idx) + ".png"), vis_normal_map)
        
        if vis_depth == True:
            depth_map = render_pkg["render_depth"]
            vis_depth_map = visualize_depth(depth_map) 
            vis_depth_map = torch.concat([vis_depth_map,alpha_mask],dim=0)
            torchvision.utils.save_image(vis_depth_map, os.path.join(aerial_depth_path, '{0:05d}'.format(idx) + ".png"))

        save_rgba(rendering, os.path.join(aerial_render_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(errormap, os.path.join(aerial_error_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(gt, os.path.join(aerial_gts_path, '{0:05d}'.format(idx) + ".png"))
        aerial_visible_count_list.append(visible_count)
        aerial_per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    if len(aerial_views) > 0:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "per_view_count.json"), 'w') as fp:
            json.dump(aerial_per_view_dict, fp, indent=True)

    # 下面这行被注释掉了，似乎是用于计算FPS的
    # print((len(aerial_t_list)-5+len(street_t_list)-5)/( sum(street_t_list[5:]) + sum(aerial_t_list[5:])))

    
def render_sets(dataset, opt, pipe, iteration, skip_train, skip_test, ape_code, explicit):
    """
    主渲染逻辑：初始化模型和场景，并调用render_set进行渲染。
    """
    with torch.no_grad(): # 禁用梯度计算，节省显存并加速
        if pipe.no_prefilter_step > 0:
            pipe.add_prefilter = False
        else:
            pipe.add_prefilter = True
        
        modules = __import__('scene')
        model_config = dataset.model_config
        model_config['kwargs']['ape_code'] = ape_code
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        
        # 初始化场景
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit)

        # ================= [新增] 手动修正相机类型 (与 train.py 保持一致) =================
        print("Fixing camera types for rendering based on filenames...")
        aerial_count = 0
        street_count = 0
        
        # 因为 eval=False，测试集图片被加载到了 TrainCameras 列表里
        # 我们遍历所有相机进行修正
        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        
        for cam in all_cameras:
            img_name = cam.image_name.lower()
            if "street" in img_name:
                cam.image_type = "street"
                street_count += 1
            elif "aerial" in img_name:
                cam.image_type = "aerial"
                aerial_count += 1
            else:
                cam.image_type = "aerial" # 默认兜底
                aerial_count += 1
                
        print(f"Render Set Classification: Aerial={aerial_count}, Street={street_count}")
        # ==============================================================================

        gaussians.eval()

        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        # 这里的 "train" 实际上对应的是你的测试集 (因为 eval=False)
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, dataset.add_aerial, dataset.add_street)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background, dataset.add_aerial, dataset.add_street)

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('-m', '--model_path', type=str, required=True) # 模型路径 (必须)
    parser.add_argument("--iteration", default=-1, type=int) # 指定加载的迭代次数，-1通常表示最新
    parser.add_argument("--ape", default=-1, type=int) # 外观编码索引
    parser.add_argument("--skip_train", action="store_true") # 是否跳过训练集渲染
    parser.add_argument("--skip_test", action="store_true") # 是否跳过测试集渲染
    parser.add_argument("--quiet", action="store_true") # 是否静默模式（减少日志）
    parser.add_argument("--explicit", action="store_true") # 是否显式模式（具体含义取决于Scene类实现）
    args = parser.parse_args(sys.argv[1:]) # 解析参数

    # 读取配置文件 config.yaml
    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        # 解析配置参数：lp(数据加载), op(优化), pp(管线)
        lp, op, pp = parse_cfg(cfg)
        # 覆盖配置中的模型路径为命令行参数提供的路径
        lp.model_path = args.model_path
    print("Rendering " + args.model_path)

    # 初始化系统状态（如随机种子），保证可复现性
    safe_state(args.quiet)

    # 开始渲染流程
    render_sets(lp, op, pp, args.iteration, args.skip_train, args.skip_test, args.ape, args.explicit)