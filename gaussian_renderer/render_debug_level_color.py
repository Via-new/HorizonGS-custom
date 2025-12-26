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
import torch
import math
import gsplat
from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs

def render(viewpoint_camera, pc, pipe, bg_color):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if pc.explicit_gs:
        pc.set_gs_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        visible_mask = pc._gs_mask
        xyz, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_explicit_gaussians(visible_mask)
    else:
        pc.set_anchor_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        visible_mask = prefilter_voxel(viewpoint_camera, pc).squeeze() if pipe.add_prefilter else pc._anchor_mask    
        xyz, offset, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_neural_gaussians(viewpoint_camera, visible_mask)

    # ================= [实验观测功能：按 Level 染色 (增强版)] =================
    if getattr(pc, "debug_level_color", False):
        try:
            # 1. 获取可见锚点的 Level 数据
            valid_levels = pc.get_level[visible_mask].flatten() 

            # 2. 对齐 Level 数量到高斯球数量
            if not pc.explicit_gs:
                k = pc.n_offsets
                potential_levels = valid_levels.repeat_interleave(k)
                if selection_mask is not None:
                    gauss_levels = potential_levels[selection_mask]
                else:
                    gauss_levels = potential_levels
            else:
                if selection_mask is not None and selection_mask.shape[0] == valid_levels.shape[0]:
                    gauss_levels = valid_levels[selection_mask]
                else:
                    gauss_levels = valid_levels

            # 3. 检查形状匹配
            if gauss_levels.shape[0] == xyz.shape[0]:
                SH_C0 = 0.28209479177387814
                
                # 定义转换函数：将 RGB [0,1] 转为 SH 系数
                # SH = (RGB - 0.5) / 0.282
                def get_sh_color(r, g, b):
                    return torch.tensor([
                        (r - 0.5) / SH_C0, 
                        (g - 0.5) / SH_C0, 
                        (b - 0.5) / SH_C0
                    ], device=xyz.device)

                # 默认颜色 (灰色)
                # RGB(0.5, 0.5, 0.5) -> SH(0, 0, 0)
                debug_colors = torch.zeros_like(xyz) 

                # Level 0 -> 鲜红色 [1, 0, 0]
                mask_l0 = (gauss_levels == 0)
                if mask_l0.any():
                    debug_colors[mask_l0] = get_sh_color(1.0, 0.0, 0.0)
                
                # Level 1 -> 鲜蓝色 [0, 0, 1]
                mask_l1 = (gauss_levels == 1)
                if mask_l1.any():
                    debug_colors[mask_l1] = get_sh_color(0.0, 0.0, 1.0)
                
                # Level 2+ -> 鲜绿色 [0, 1, 0]
                mask_l2 = (gauss_levels >= 2)
                if mask_l2.any():
                    debug_colors[mask_l2] = get_sh_color(0.0, 1.0, 0.0)

                # 4. 应用颜色并调整维度 [N, 1, 3]
                color = debug_colors.unsqueeze(1) 
                sh_degree = 0
            else:
                pass # 忽略形状不匹配的情况
        
        except Exception as e:
            print(f"[DebugColor Error] {e}")
    # ================================================================

    # Set up rasterization configuration
    K = torch.tensor([
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],dtype=torch.float32, device="cuda")
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    if pc.gs_attr == "3D":
        render_colors, render_alphas, info = gsplat.rasterization(
            means=xyz,  # [N, 3]
            quats=rot,  # [N, 4]
            scales=scaling,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=color,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=bg_color[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=sh_degree,
            render_mode=pc.render_mode,
        )
    elif pc.gs_attr == "2D":
        (render_colors, 
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,), info = \
        gsplat.rasterization_2dgs(
            means=xyz,  # [N, 3]
            quats=rot,  # [N, 4]
            scales=scaling,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=color,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=bg_color[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=sh_degree,
            render_mode=pc.render_mode,
        )
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")

    # [1, H, W, 3] -> [3, H, W]
    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    render_alphas = render_alphas[0].permute(2, 0, 1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return_dict = {
        "render": rendered_image,
        "scaling": scaling,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "visible_mask": visible_mask,
        "selection_mask": selection_mask,
        "opacity": opacity,
        "render_depth": depth,
        "radii": radii,
        "render_alphas": render_alphas,
    }
    
    if pc.gs_attr == "2D":
        return_dict.update({
            "render_normals": render_normals,
            "render_normals_from_depth": render_normals_from_depth,
            "render_distort": render_distort,
        })

    return return_dict

def prefilter_voxel(viewpoint_camera, pc):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    means = pc.get_anchor[pc._anchor_mask]
    scales = pc.get_scaling[pc._anchor_mask][:, :3]
    quats = pc.get_rotation[pc._anchor_mask]
    
    # Set up rasterization configuration
    Ks = torch.tensor([
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],dtype=torch.float32, device="cuda")[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    if pc.gs_attr == "3D":
        proj_results = fully_fused_projection(
            means,
            None,  # covars,
            quats,
            scales,
            viewmats,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
            calc_compensations=False,
        )
    elif pc.gs_attr == "2D":
        densifications = (
            torch.zeros((C, N, 2), dtype=means.dtype, device="cuda")
        )
        # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
        proj_results = fully_fused_projection_2dgs(
            means,
            quats,
            scales,
            viewmats,
            densifications,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
        )
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")
    
    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results
    camera_ids, gaussian_ids = None, None
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii.squeeze(0) > 0

    return visible_mask