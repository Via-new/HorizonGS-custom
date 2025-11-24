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
import shutil
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import torch
import torchvision
import json
import wandb
import time
from datetime import datetime
from os import makedirs
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from gaussian_renderer import network_gui
from scene import Scene
from utils.general_utils import get_expon_lr_func, safe_state, parse_cfg, visualize_depth
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, save_rgba
from argparse import ArgumentParser, Namespace
import yaml
import torch.nn.functional as F
import warnings

# 1105
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
warnings.filterwarnings('ignore')

lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    assert os.path.exists(os.path.join(ROOT, '.gitignore'))
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    modules = __import__('scene')
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
    scene = Scene(dataset, gaussians, shuffle=False, logger=logger, weed_ratio=pipe.weed_ratio)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    if pipe.camera_balance:
        aerial_viewpoint_stack = None
        street_viewpoint_stack = None
    else:
        viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    densify_cnt = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    modules = __import__('gaussian_renderer')
    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in Horizon-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.add_prefilter, keep_alive = network_gui.receive()
                if custom_cam != None:
                    net_image = getattr(modules, 'render')(custom_cam, gaussians, pipe, scene.background)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        
        # Pick a random Camera
        if pipe.camera_balance:
            if not aerial_viewpoint_stack:
                aerial_viewpoint_stack = [cam for cam in scene.getTrainCameras().copy() if cam.image_type == "aerial"]
            if not street_viewpoint_stack:
                street_viewpoint_stack = [cam for cam in scene.getTrainCameras().copy() if cam.image_type == "street"]
            aerial_proportion, street_proportion = pipe.camera_proportion.split("-")
            r = float(aerial_proportion) / ( float(aerial_proportion) + float(street_proportion) )
            if np.random.rand() < r:
                viewpoint_cam = aerial_viewpoint_stack.pop(randint(0, len(aerial_viewpoint_stack)-1))
            else:
                viewpoint_cam = street_viewpoint_stack.pop(randint(0, len(street_viewpoint_stack)-1))
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = getattr(modules, 'render')(viewpoint_cam, gaussians, pipe, scene.background)
        image, scaling, alpha = render_pkg["render"], render_pkg["scaling"], render_pkg["render_alphas"]

        gt_image = viewpoint_cam.original_image.cuda()
        alpha_mask = viewpoint_cam.alpha_mask.cuda()
        image = image * alpha_mask
        gt_image = gt_image * alpha_mask
            
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
       
        if opt.lambda_dreg > 0:
            if scaling.shape[0] > 0:
                scaling_reg = scaling.prod(dim=1).mean()
            else:
                scaling_reg = torch.tensor(0.0, device="cuda")
            loss += opt.lambda_dreg * scaling_reg
        
        if opt.lambda_sky_opa > 0:
            o = alpha.clamp(1e-6, 1-1e-6)
            sky = alpha_mask.float()
            loss_sky_opa = (-(1-sky) * torch.log(1 - o)).mean()
            loss = loss + opt.lambda_sky_opa * loss_sky_opa

        if opt.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o*torch.log(o)).mean()
            loss = loss + opt.lambda_opacity_entropy * loss_opacity_entropy

        if opt.lambda_normal > 0 and iteration > opt.normal_start_iter:
            assert gaussians.render_mode=="RGB+ED" or gaussians.render_mode=="RGB+D"
            normals = render_pkg["render_normals"].squeeze(0).permute((2, 0, 1))
            normals_from_depth = render_pkg["render_normals_from_depth"] * render_pkg["render_alphas"].permute((1, 2, 0)).detach()
            if len(normals_from_depth.shape) == 4:
                normals_from_depth = normals_from_depth.squeeze(0)
            normals_from_depth = normals_from_depth.permute((2, 0, 1))
            normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
            loss += opt.lambda_normal * (normal_error * alpha_mask).mean()

        if opt.lambda_dist and iteration > opt.dist_start_iter:
            loss += opt.lambda_dist * (render_pkg["render_distort"].squeeze(3) * alpha_mask).mean()
        
        if iteration > opt.start_depth and depth_l1_weight(iteration) > 0 and viewpoint_cam.invdepthmap is not None:
            assert gaussians.render_mode=="RGB+ED" or gaussians.render_mode=="RGB+D"
            render_depth = render_pkg["render_depth"]
            invDepth = torch.where(render_depth > 0.0, 1.0 / render_depth, torch.zeros_like(render_depth))            
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
    
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                psnr_log = psnr(image, gt_image).mean().double()
                anchor_prim = len(gaussians.get_anchor)
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}","Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}","psnr":f"{psnr_log:.{3}f}","GS_num":f"{anchor_prim}","prefilter":f"{pipe.add_prefilter}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, getattr(modules, 'render'), (pipe, scene.background), wandb, logger)
            
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % pipe.vis_step == 0 or iteration == 1:
                # render_img/gt_img/render_depth/gt_depth/render_alphas/masks
                other_img = []
                resolution = (int(viewpoint_cam.image_width/5.0), int(viewpoint_cam.image_height/5.0))
                vis_img = F.interpolate(image.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                vis_gt_img = F.interpolate(gt_image.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                vis_alpha = F.interpolate(alpha.repeat(3, 1, 1).unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]

                if iteration > opt.start_depth and viewpoint_cam.invdepthmap is not None:
                    vis_depth = visualize_depth(invDepth) 
                    gt_depth = visualize_depth(mono_invdepth)
                    vis_depth = F.interpolate(vis_depth.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                    vis_gt_depth = F.interpolate(gt_depth.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                    other_img.append(vis_depth)
                    other_img.append(vis_gt_depth)
                
                grid = torchvision.utils.make_grid([
                    vis_img, 
                    vis_gt_img, 
                    vis_alpha,
                ] + other_img, nrow=3)

                vis_path = os.path.join(scene.model_path, "vis")
                os.makedirs(vis_path, exist_ok=True)
                torchvision.utils.save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                if  (viewpoint_cam.image_type == "aerial" and pipe.aerial_densify) \
                    or (viewpoint_cam.image_type == "street" and pipe.street_densify) :
                    gaussians.training_statis(opt, render_pkg, image.shape[2], image.shape[1])
                    densify_cnt += 1 

                # densification
                if opt.densification and iteration > opt.update_from and densify_cnt > 0 and densify_cnt % opt.update_interval == 0:
                    if dataset.pretrained_checkpoint != "":
                        gaussians.roll_back()
                    gaussians.run_densify(opt, iteration)
            
            elif iteration == opt.update_until:
                if dataset.pretrained_checkpoint != "":
                    gaussians.roll_back()
                gaussians.clean()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration >= opt.iterations - pipe.no_prefilter_step:
                pipe.add_prefilter = False

            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in range(0, len(scene.getTrainCameras()), 100)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test_aerial = 0.0
                psnr_test_aerial = 0.0
                aerial_cnt = 0
                l1_test_street = 0.0
                psnr_test_street = 0.0
                street_cnt = 0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    alpha_mask = viewpoint.alpha_mask.cuda()
                    image = image * alpha_mask
                    gt_image = gt_image * alpha_mask
                    
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        if wandb:
                            gt_image_list.append(gt_image[None])
                
                    if viewpoint.image_type == "aerial":
                        l1_test_aerial += l1_loss(image, gt_image).mean().double()
                        psnr_test_aerial += psnr(image, gt_image).mean().double()
                        aerial_cnt += 1 
                    else:
                        l1_test_street += l1_loss(image, gt_image).mean().double()
                        psnr_test_street += psnr(image, gt_image).mean().double()
                        street_cnt += 1 
                
                if scene.add_aerial and aerial_cnt > 0:
                    l1_test_aerial /= aerial_cnt
                    psnr_test_aerial /= aerial_cnt    
                    logger.info("\n[ITER {}] Evaluating {} Aerial: L1 {} PSNR {}".format(iteration, config['name'], l1_test_aerial, psnr_test_aerial))
                if scene.add_street and street_cnt > 0:       
                    l1_test_street /= street_cnt
                    psnr_test_street /= street_cnt       
                    logger.info("\n[ITER {}] Evaluating {} Street: L1 {} PSNR {}".format(iteration, config['name'], l1_test_street, psnr_test_street))
                
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', len(scene.gaussians.get_anchor), iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipe, background, add_aerial, add_street):
    if add_aerial:
        aerial_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "renders")
        aerial_error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "errors")
        aerial_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "gt")
        makedirs(aerial_render_path, exist_ok=True)
        makedirs(aerial_error_path, exist_ok=True)
        makedirs(aerial_gts_path, exist_ok=True)
    if add_street:
        street_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "renders")
        street_error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "errors")
        street_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "gt")
        makedirs(street_render_path, exist_ok=True)
        makedirs(street_error_path, exist_ok=True)
        makedirs(street_gts_path, exist_ok=True)

    modules = __import__('gaussian_renderer')

    street_t_list = []
    street_visible_count_list = []
    street_per_view_dict = {}
    street_views = [view for view in views if view.image_type=="street"]
    for idx, view in enumerate(tqdm(street_views, desc="Street rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        torch.cuda.synchronize();t_end = time.time()

        street_t_list.append(t_end - t_start)

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

        save_rgba(rendering, os.path.join(street_render_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(errormap, os.path.join(street_error_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(gt, os.path.join(street_gts_path, '{0:05d}'.format(idx) + ".png"))
        street_visible_count_list.append(visible_count)
        street_per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    if len(street_views) > 0:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "street", "per_view_count.json"), 'w') as fp:
            json.dump(street_per_view_dict, fp, indent=True)
    
    aerial_t_list = []
    aerial_visible_count_list = []
    aerial_per_view_dict = {}
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

        save_rgba(rendering, os.path.join(aerial_render_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(errormap, os.path.join(aerial_error_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(gt, os.path.join(aerial_gts_path, '{0:05d}'.format(idx) + ".png"))
        aerial_visible_count_list.append(visible_count)
        aerial_per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    if len(aerial_views) > 0:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "per_view_count.json"), 'w') as fp:
            json.dump(aerial_per_view_dict, fp, indent=True)

    return aerial_visible_count_list, street_visible_count_list

def render_sets(dataset, opt, pipe, iteration, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        if pipe.no_prefilter_step > 0:
            pipe.add_prefilter = False
        else:
            pipe.add_prefilter = True
        modules = __import__('scene')
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, logger=logger)
        gaussians.eval()

        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            aerial_visible_count, street_visible_count = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, scene.add_aerial, scene.add_street)

        if not skip_test:
            aerial_visible_count, street_visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background, scene.add_aerial, scene.add_street)

    return aerial_visible_count, street_visible_count

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        render_image = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        render_mask = tf.to_tensor(render).unsqueeze(0)[:, 3:4, :, :].cuda()
        render_image = render_image * render_mask
        gt_image = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        gt_mask = tf.to_tensor(gt).unsqueeze(0)[:, 3:4, :, :].cuda()
        gt_image = gt_image * gt_mask
        renders.append(render_image)
        gts.append(gt_image)
        image_names.append(fname)
    return renders, gts, image_names
def evaluate(model_paths, eval_name, aerial_visible_count=None, street_visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        if "ucgs" in model_paths:

            base_method_dir = test_dir / method
            method_dir = base_method_dir / "street" 
            if os.path.exists(method_dir):
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
                logger.info("  Held-out STREET_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs[72:-1]).mean(), ".5"))
                logger.info("  Held-out STREET_SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims[72:-1]).mean(), ".5"))
                logger.info("  Held-out STREET_LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss[72:-1]).mean(), ".5"))
                logger.info("  Held-out STREET_GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(street_visible_count[72:-1]).float().mean(), ".5"))
                
                logger.info("  View(+0.1m) STREET_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs[:36]).mean(), ".5"))
                logger.info("  View(+0.1m) STREET_SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims[:36]).mean(), ".5"))
                logger.info("  View(+0.1m) STREET_LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss[:36]).mean(), ".5"))
                logger.info("  View(+0.1m) STREET_GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(street_visible_count[:36]).float().mean(), ".5"))
                
                logger.info("  View(+0.1m 5°down) STREET_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs[36:72]).mean(), ".5"))
                logger.info("  View(+0.1m 5°down) STREET_SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims[36:72]).mean(), ".5"))
                logger.info("  View(+0.1m 5°down) STREET_LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss[36:72]).mean(), ".5"))
                logger.info("  View(+0.1m 5°down) STREET_GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(street_visible_count[36:72]).float().mean(), ".5"))
                
                full_dict[scene_dir][method].update({
                "Held-out STREET_PSNR": torch.tensor(psnrs[72:-1]).mean().item(),
                "Held-out STREET_SSIM": torch.tensor(ssims[72:-1]).mean().item(),
                "Held-out STREET_LPIPS": torch.tensor(lpipss[72:-1]).mean().item(),
                "Held-out STREET_GS_NUMS": torch.tensor(street_visible_count[72:-1]).float().mean().item(),
                
                "View(+0.1m) STREET_PSNR": torch.tensor(psnrs[:36]).mean().item(),
                "View(+0.1m) STREET_SSIM": torch.tensor(ssims[:36]).mean().item(),
                "View(+0.1m) STREET_LPIPS": torch.tensor(lpipss[:36]).mean().item(),
                "View(+0.1m) STREET_GS_NUMS": torch.tensor(street_visible_count[:36]).float().mean().item(),
                
                "View(+0.1m 5°down) STREET_PSNR": torch.tensor(psnrs[36:72]).mean().item(),
                "View(+0.1m 5°down) STREET_SSIM": torch.tensor(ssims[36:72]).mean().item(),
                "View(+0.1m 5°down) STREET_LPIPS": torch.tensor(lpipss[36:72]).mean().item(),
                "View(+0.1m 5°down) STREET_GS_NUMS": torch.tensor(street_visible_count[36:72]).float().mean().item(),
                })
        else:
            base_method_dir = test_dir / method
            method_dir = base_method_dir / "aerial" 
            if os.path.exists(method_dir):
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
                logger.info("  AERIAL_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
                logger.info("  AERIAL_SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
                logger.info("  AERIAL_LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
                logger.info("  AERIAL_GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(aerial_visible_count).float().mean(), ".5"))
                print("")
                
                full_dict[scene_dir][method].update({
                    "AERIAL_PSNR": torch.tensor(psnrs).mean().item(),
                    "AERIAL_SSIM": torch.tensor(ssims).mean().item(),
                    "AERIAL_LPIPS": torch.tensor(lpipss).mean().item(),
                    "AERIAL_GS_NUMS": torch.tensor(aerial_visible_count).float().mean().item(),
                    })

                per_view_dict[scene_dir][method].update({
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(aerial_visible_count).tolist(), image_names)}
                    })

            method_dir = base_method_dir / "street" 
            if os.path.exists(method_dir):
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
                logger.info("  STREET_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
                logger.info("  STREET_SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
                logger.info("  STREET_LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
                logger.info("  STREET_GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(street_visible_count).float().mean(), ".5"))
                print("")
                
                full_dict[scene_dir][method].update({
                    "STREET_PSNR": torch.tensor(psnrs).mean().item(),
                    "STREET_SSIM": torch.tensor(ssims).mean().item(),
                    "STREET_LPIPS": torch.tensor(lpipss).mean().item(),
                    "STREET_GS_NUMS": torch.tensor(street_visible_count).float().mean().item(),
                    })

                per_view_dict[scene_dir][method].update({
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(street_visible_count).tolist(), image_names)}
                    })

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # 1. 解析命令行参数（新增local_rank支持DDP）
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default='-1')
    # 新增：DDP本地进程编号（由torchrun自动传递）
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP local rank')
    args = parser.parse_args(sys.argv[1:])

    # 2. 读取配置文件
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        args.save_iterations.append(op.iterations)

    # 3. DDP初始化（核心步骤）
    if args.local_rank != -1:
        # 初始化分布式进程组（NCCL后端适合GPU通信）
        dist.init_process_group(
            backend='nccl',
            init_method='env://',  # 由torchrun自动设置环境变量
            world_size=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0))
        )
        # 设置当前进程的设备
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # 主进程标识（rank=0的进程负责日志和保存）
        is_main_process = (dist.get_rank() == 0)
    else:
        # 单卡模式
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    # 4. 输出目录与日志（仅主进程创建，避免冲突）
    if is_main_process:
        lp.model_path = os.path.join("outputs", lp.dataset_name, lp.scene_name)
        os.makedirs(lp.model_path, exist_ok=True)
        shutil.copy(args.config, os.path.join(lp.model_path, "config.yaml"))
        logger = get_logger(lp.model_path)
    else:
        logger = None  # 非主进程不输出日志

    # 5. 处理测试/保存迭代参数（仅主进程打印）
    if is_main_process:
        if args.test_iterations[0] == -1:
            args.test_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
        if len(args.test_iterations) == 0 or args.test_iterations[-1] != op.iterations:
            args.test_iterations.append(op.iterations)
        if args.save_iterations[0] == -1:
            args.save_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
        if len(args.save_iterations) == 0 or args.save_iterations[-1] != op.iterations:
            args.save_iterations.append(op.iterations)

    # 6. GPU设备设置（覆盖原GPU参数，由DDP管理）
    if args.gpu != '-1' and is_main_process:
        logger.info(f'忽略--gpu参数，DDP自动管理设备：{args.local_rank}')

    # 7. 代码备份（仅主进程执行）
    if is_main_process:
        try:
            saveRuntimeCode(os.path.join(lp.model_path, 'backup'))
        except:
            logger.info(f'save code failed~')

    # 8. Wandb初始化（仅主进程启动）
    wandb_run = None
    if args.use_wandb and is_main_process:
        exp_name = lp.scene_name if lp.dataset_name=="" else lp.dataset_name+"_"+lp.scene_name
        wandb.login()
        wandb_run = wandb.init(
            project=f"Horizon-GS",
            name=exp_name,
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )

    # 9. 初始化随机种子（确保多进程同步）
    safe_state(args.quiet)
    torch.manual_seed(42 + dist.get_rank() if args.local_rank != -1 else 42)

    # 10. 启动训练（传递设备和主进程标识）
    if is_main_process and logger:
        logger.info("Optimizing " + lp.model_path)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 调用training函数时传入DDP相关参数
    training(
        lp, op, pp, exp_name, 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
        args.start_checkpoint, wandb_run, logger,
        device=device,  # 新增：当前进程的设备
        is_main_process=is_main_process,  # 新增：是否为主进程
        local_rank=args.local_rank  # 新增：本地进程编号
    )

    # 11. 训练结束后清理（仅主进程执行后续步骤）
    if is_main_process:
        logger.info("\nTraining complete.")

        # 渲染（仅主进程执行）
        logger.info(f'\nStarting Rendering~')
        if lp.eval:
            aerial_visible_count, street_visible_count = render_sets(
                lp, op, pp, -1, skip_train=True, skip_test=False, 
                wandb=wandb_run, logger=logger
            )
        else:
            aerial_visible_count, street_visible_count = render_sets(
                lp, op, pp, -1, skip_train=False, skip_test=True, 
                wandb=wandb_run, logger=logger
            )
        logger.info("\nRendering complete.")

        # 评估（仅主进程执行）
        logger.info("\n Starting evaluation...")
        eval_name = 'test' if lp.eval else 'train'
        evaluate(
            lp.model_path, eval_name, 
            aerial_visible_count=aerial_visible_count, 
            street_visible_count=street_visible_count, 
            wandb=wandb_run, logger=logger
        )
        logger.info("\nEvaluating complete.")

    # 12. 销毁分布式进程组
    if args.local_rank != -1:
        dist.destroy_process_group()