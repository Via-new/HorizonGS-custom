#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modified for HorizonGS Source-Aware Rendering (Scheme 1)
#

import torch
import math
import os
import gsplat
from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs

def render(viewpoint_camera, pc, pipe, bg_color):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # A. 尝试加载 Mask (Lazy Loading)
    if not hasattr(pc, "source_mask"):
        # [Fix] 增加鲁棒性检查
        model_path = getattr(pc, "model_path", None)
        
        if model_path and os.path.exists(os.path.join(model_path, "anchor_source_mask.pt")):
            mask_path = os.path.join(model_path, "anchor_source_mask.pt")
            try:
                print(f"[Render] Loading Source Mask from {mask_path}")
                pc.source_mask = torch.load(mask_path).to("cuda")
            except Exception as e:
                print(f"[Render] Warning: Failed to load mask: {e}")
                pc.source_mask = None
        else:
            # 如果找不到 model_path 或文件不存在
            # print("[Render] No source mask found. Skipping filtering.")
            pc.source_mask = None

    # 计算基于源的过滤掩码 (Source Filter)
    source_filter = None
    if getattr(pc, "source_mask", None) is not None:
        # 判断相机类型
        img_name = viewpoint_camera.image_name.lower()
        is_aerial = "aerial" in img_name
        
        # 0=Air, 1=Street, 2=Shared
        if is_aerial:
            # 航拍相机：不看 Street(1)
            source_filter = (pc.source_mask != 1)
        else:
            # 街景相机：不看 Air(0)
            source_filter = (pc.source_mask != 0)
    # =============================================================

    if pc.explicit_gs:
        pc.set_gs_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        visible_mask = pc._gs_mask
        
        # [应用过滤]
        if source_filter is not None:
            # 确保长度一致 (Explicit 模式下 mask 对应的是 GS)
            if source_filter.shape[0] == visible_mask.shape[0]:
                visible_mask = visible_mask & source_filter
                
        xyz, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_explicit_gaussians(visible_mask)
    else:
        pc.set_anchor_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        visible_mask = prefilter_voxel(viewpoint_camera, pc).squeeze() if pipe.add_prefilter else pc._anchor_mask    
        
        # [应用过滤] 核心修改点
        if source_filter is not None:
            # 确保长度一致 (Implicit 模式下 mask 对应的是 Anchor)
            if source_filter.shape[0] == visible_mask.shape[0]:
                visible_mask = visible_mask & source_filter
        
        # 将过滤后的 mask 传给解码器，这样被屏蔽的锚点根本不会生成高斯球
        xyz, offset, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_neural_gaussians(viewpoint_camera, visible_mask)

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