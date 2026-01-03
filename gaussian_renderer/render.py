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

def render(viewpoint_camera, pc, pipe, bg_color, override_mask=None, override_color=None, force_opaque=False):
    """
    Render the scene. 
    
    Args:
        override_mask (Tensor, optional): If provided, ONLY renders anchors where mask is True.
        override_color (Tensor, optional): [N_anchors, 3] RGB Tensor. Forces visible points to use this color.
        force_opaque (bool, optional): If True, forces opacity to ~1.0 for visualization clarity.
    """
    # 1. 基础几何生成与筛选
    if pc.explicit_gs:
        pc.set_gs_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        visible_mask = pc._gs_mask
        
        # [Override Mask]
        if override_mask is not None:
            visible_mask = visible_mask & override_mask

        xyz, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_explicit_gaussians(visible_mask)
    else:
        pc.set_anchor_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        visible_mask = prefilter_voxel(viewpoint_camera, pc).squeeze() if pipe.add_prefilter else pc._anchor_mask    
        
        # [Override Mask]
        if override_mask is not None:
            # Intersection: Visible AND in Override Mask
            visible_mask = visible_mask & override_mask

        xyz, offset, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_neural_gaussians(viewpoint_camera, visible_mask)

    # 2. [Color Overrides] 颜色覆盖逻辑
    if override_color is not None:
        try:
            # Ensure override_color is on the correct device
            if override_color.device != visible_mask.device:
                override_color = override_color.to(visible_mask.device)

            # A. Extract colors for currently visible anchors
            # override_color 长度是所有锚点的数量，我们需要取 visible_mask 对应的部分
            visible_colors = override_color[visible_mask] # [N_visible_anchors, 3]

            # B. Handle Neural Expansion (1 Anchor -> k Gaussians)
            # 如果是神经高斯（HorizonGS），一个锚点会生成 k 个高斯，颜色需要复制
            if not pc.explicit_gs:
                k = getattr(pc, "n_offsets", 1)
                # Expand colors: [ColorA, ColorB] -> [ColorA, ColorA..., ColorB, ColorB...]
                visible_colors = visible_colors.repeat_interleave(k, dim=0) 
                
                # C. Handle Pruning (if generate_neural_gaussians dropped some offsets based on opacity)
                if selection_mask is not None:
                    # selection_mask 是 generate_neural_gaussians 内部生成的，用于剔除透明度极低的高斯
                    # 它的长度等于 N_visible_anchors * k
                    if visible_colors.shape[0] != xyz.shape[0] and selection_mask.shape[0] == visible_colors.shape[0]:
                        visible_colors = visible_colors[selection_mask]
            
            # D. Final Shape Check & Apply
            if visible_colors.shape[0] == xyz.shape[0]:
                # Convert RGB to SH DC component
                # Formula: SH_0 = (RGB - 0.5) / C0
                SH_C0 = 0.28209479177387814
                
                # Apply override
                color = (visible_colors - 0.5) / SH_C0
                color = color.unsqueeze(1) # [N_gaussians, 1, 3]
                sh_degree = 0 # [重要] 强制使用 DC 颜色，忽略高阶 SH
            else:
                # 形状不匹配（极少情况），跳过覆盖
                pass 

        except Exception as e:
            print(f"[Render Error] Failed to apply override_color: {e}")

    # 3. [Force Opaque] 强制不透明 (用于 Debug 分类)
    if force_opaque:
        # 强制设置为 0.99 (完全不透明)
        opacity = torch.ones_like(opacity) * 0.99

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
    
    # Project Gaussians to 2D. 
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
    
    radii, means2d, depths, conics, compensations = proj_results
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii.squeeze(0) > 0

    return visible_mask