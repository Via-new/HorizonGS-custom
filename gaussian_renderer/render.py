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

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 1. 获取所有高斯属性 (Decoding)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # =========================================================================
    # [新增] 方案一：Source-Aware Filtering (源感知过滤)
    # =========================================================================
    
    # A. 尝试加载 Mask (如果还没加载过)
    if not hasattr(pc, "source_mask"):
        # 尝试从模型路径查找 anchor_source_mask.pt
        mask_path = os.path.join(pc.model_path, "anchor_source_mask.pt")
        if os.path.exists(mask_path):
            print(f"[Render] Loading Source Mask from {mask_path}")
            pc.source_mask = torch.load(mask_path).to(means3D.device)
        else:
            # 如果找不到文件，设为 None，后续不做过滤
            pc.source_mask = None

    # B. 执行过滤逻辑
    if hasattr(pc, "source_mask") and pc.source_mask is not None:
        
        # 1. 确定当前相机身份 (简单判断逻辑，可根据实际情况修改)
        img_name = viewpoint_camera.image_name.lower()
        is_aerial = "aerial" in img_name
        # 如果既不是 aerial 也没明确标记 street (比如测试集), 默认都看
        # 或者你可以定义 else 为 street
        
        # 2. 生成过滤掩码 (基于 Anchor)
        # 0=Air, 1=Street, 2=Shared
        if is_aerial:
            # 航拍：不看 Street(1)
            visibility_filter = (pc.source_mask != 1)
        else:
            # 街景：不看 Air(0)
            visibility_filter = (pc.source_mask != 0)
            
        # 3. 处理 LOD 导致的高斯倍增 (Anchor -> Gaussians)
        # means3D 的数量可能是 Anchors 的 K 倍 (K = n_offsets)
        n_anchors = pc.source_mask.shape[0]
        n_gaussians = means3D.shape[0]
        
        if n_gaussians != n_anchors:
            # 确保是整数倍关系
            if n_gaussians % n_anchors == 0:
                repeat_factor = n_gaussians // n_anchors
                # 将 Mask 扩展以匹配高斯数量
                visibility_filter = visibility_filter.repeat_interleave(repeat_factor)
            else:
                # 异常情况：数量对不上，放弃过滤（防止报错）
                pass 
        
        # 4. 应用过滤
        # 注意：必须过滤所有传给 Rasterizer 的属性
        if visibility_filter.sum() > 0: # 确保还有点剩下
            means3D = means3D[visibility_filter]
            means2D = means2D[visibility_filter]
            opacity = opacity[visibility_filter]
            
            if scales is not None: scales = scales[visibility_filter]
            if rotations is not None: rotations = rotations[visibility_filter]
            if cov3D_precomp is not None: cov3D_precomp = cov3D_precomp[visibility_filter]
            if shs is not None: shs = shs[visibility_filter]
            if colors_precomp is not None: colors_precomp = colors_precomp[visibility_filter]
    # =========================================================================

    # Rasterize visible Gaussians to image
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from the accumulated Histograms later.
    # [注意] 如果进行了 Mask 过滤，这里的 visible_mask 是相对于“过滤后的子集”的
    # 为了保持外部逻辑（如 Viewspace 梯度统计）的兼容性，通常这里不需要改动，
    # 因为反向传播会自动处理经过 Mask 的计算图。
    
    return {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            # [Fix] 传递 source_mask 状态给外部 (可选)
            # "visible_mask": radii > 0 
            }

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