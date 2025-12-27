#
# gaussian_renderer/render_scheme1.py
#
# 基于 render_origin.py 修改，实现了 Scheme 1 的 Mask 过滤逻辑
#

import torch
import math
import gsplat
from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs

def render(viewpoint_camera, pc, pipe, bg_color):
    """
    Render the scene with Anchor Splitting Scheme 1.
    Filters anchors based on pc.anchor_source_mask and viewpoint_camera.image_type.
    """
    if pc.explicit_gs:
        pc.set_gs_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        visible_mask = pc._gs_mask
        xyz, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_explicit_gaussians(visible_mask)
    else:
        # 1. 基础视锥剔除与 Level 选择
        pc.set_anchor_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        visible_mask = prefilter_voxel(viewpoint_camera, pc).squeeze() if pipe.add_prefilter else pc._anchor_mask    
        
        # ================= [Scheme 1 核心修改 Start] =================
        # 根据 anchor_source_mask 和 相机类型 进一步筛选
        # Mask 定义: 0=仅航拍(Air), 1=仅街景(Street), 2=共享(Shared)
        if hasattr(pc, "anchor_source_mask") and pc.anchor_source_mask is not None:
            # 获取相机类型，默认为 aerial 以防万一
            cam_type = getattr(viewpoint_camera, "image_type", "aerial")
            
            if cam_type == "aerial":
                # 航拍视角：看 (Source==0) 和 (Source==2) -> 屏蔽 (Source==1)
                valid_source = (pc.anchor_source_mask != 1)
                visible_mask = visible_mask & valid_source
            elif cam_type == "street":
                # 街景视角：看 (Source==1) 和 (Source==2) -> 屏蔽 (Source==0)
                valid_source = (pc.anchor_source_mask != 0)
                visible_mask = visible_mask & valid_source
            # 其他情况（如 unknown）默认全看，不进行额外过滤
        # ================= [Scheme 1 核心修改 End] =================

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