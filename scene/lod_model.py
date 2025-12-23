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
import time
import torch
import math
import numpy as np
from torch import nn
from einops import repeat
from functools import reduce
from torch_scatter import scatter_max
from utils.general_utils import get_expon_lr_func, knn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from scene.embedding import Embedding
from scene.basic_model import BasicModel
from utils.system_utils import searchForMaxIteration
    
class GaussianLoDModel(BasicModel):

    def __init__(self, **model_kwargs):

        for key, value in model_kwargs.items():
            setattr(self, key, value)

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._level = torch.empty(0)
        self._extra_level = torch.empty(0)

        self.offset_opacity_accum = torch.empty(0)
        self.anchor_opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.max_radii2D = torch.empty(0)
                
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.padding =  0.0
        self.ape_code = -1
        self.dist2level = 'floor'
        self.setup_functions()

        if self.color_attr == "RGB":  
            self.active_sh_degree = None
            self.max_sh_degree = None   
            self.color_dim = 3  
        else:
            self.active_sh_degree = 0
            self.max_sh_degree = int(''.join(filter(str.isdigit, self.color_attr)))
            self.color_dim = 3 * ((self.max_sh_degree + 1) ** 2)

        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()
        
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim+self.appearance_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.color_dim*self.n_offsets),
        ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
    
    def freeze(self):
        self.freeze_mlp(self.mlp_opacity)
        self.freeze_mlp(self.mlp_cov)
        self.freeze_mlp(self.mlp_color)
    
    def freeze_mlp(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def capture(self):
        param_dict = {}
        param_dict['optimizer'] = self.optimizer.state_dict()
        param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
        param_dict['cov_mlp'] = self.mlp_cov.state_dict()
        param_dict['color_mlp'] = self.mlp_color.state_dict()
        if self.appearance_dim > 0:
            param_dict['appearance'] = self.embedding_appearance.state_dict()
        return (
            self.voxel_size,
            self.standard_dist,
            self._anchor,
            self._level,
            self._extra_level,
            self._offset,
            self._scaling,
            self._rotation,
            self.offset_opacity_accum, 
            self.anchor_opacity_accum, 
            self.anchor_demon,
            self.offset_gradient_accum,
            self.offset_denom,
            param_dict,
            self.spatial_lr_scale,
            self.max_radii2D,
        )
    
    def restore(self, model_args, training_args):
        (self.voxel_size,
        self.standard_dist,
        self._anchor,
        self._level,
        self._extra_level,
        self._offset,
        self._scaling,
        self._rotation,
        self.offset_opacity_accum,
        self.max_radii2D,
        self.anchor_opacity_accum, 
        self.anchor_demon,
        self.offset_gradient_accum,
        self.offset_denom,
        param_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(param_dict['optimizer'])
        self.mlp_opacity.load_state_dict(param_dict['opacity_mlp'])
        self.mlp_cov.load_state_dict(param_dict['cov_mlp'])
        self.mlp_color.load_state_dict(param_dict['color_mlp'])
        if self.appearance_dim > 0:
            self.embedding_appearance.load_state_dict(param_dict['appearance'])

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_level(self):
        return self._level
    
    @property
    def get_extra_level(self):
        return self._extra_level
        
    @property
    def get_anchor_feat(self):
        return self._anchor_feat

    @property
    def get_offset(self):
        return self._offset

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
        else:
            self.embedding_appearance = None
            
    @property
    def get_appearance(self):
        return self.embedding_appearance
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    def smooth_complement(self, visible_mask):   
        if self.dist2level == "progressive":
            prog = self._prog_ratio[visible_mask]
            transition_mask = self.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
        else:
            prog = torch.ones((visible_mask.sum(), 1), dtype=torch.float, device="cuda")
        return prog
        
    def octree_sample(self, data):
        points = torch.empty(0, 3).float().cuda()
        levels = torch.empty(0).int().cuda() 
        for cur_level in range(self.aerial_levels):
            cur_size = self.voxel_size / (float(self.fork) ** (cur_level + 1 - self.aerial_levels))
            new_points = torch.unique(torch.round(data / cur_size), dim=0) * cur_size
            new_points += self.padding * cur_size
            new_levels = torch.ones(new_points.shape[0], dtype=torch.int, device="cuda") * cur_level
            points = torch.concat((points, new_points), dim=0)
            levels = torch.concat((levels, new_levels), dim=0)
        return points, levels

    def weed_out(self, positions, levels):
        if self.weed_ratio > 0:
            visible_count = torch.zeros(positions.shape[0], dtype=torch.int, device="cuda")
            for cam in self.cam_infos:
                cam_center, scale = cam[:3], cam[3]
                dist = torch.sqrt(torch.sum((positions - cam_center)**2, dim=1)) * scale
                pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   
                int_level = self.map_to_int_level(pred_level, self.street_levels - 1)
                visible_count += (levels <= int_level).int()
            visible_count = visible_count/len(self.cam_infos)
            weed_mask = (visible_count > self.weed_ratio)
        else:
            weed_mask = torch.ones(positions.shape[0], dtype=torch.bool, device="cuda")
        return weed_mask

    def create_from_pcd(self, pcd, spatial_lr_scale, global_appearance,  logger):
        self.spatial_lr_scale = spatial_lr_scale
        self.training_stage = "coarse"
        points = torch.tensor(pcd.points).float().cuda()
        fused_point_cloud, levels = self.octree_sample(points)

        weed_mask = self.weed_out(fused_point_cloud, levels)
        fused_point_cloud = fused_point_cloud[weed_mask]
        levels = levels[weed_mask]
        
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        logger.info(f'Initial Voxel Number: {fused_point_cloud.shape[0]}')
        logger.info(f'Voxel Size: {self.voxel_size}')

        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self._level = levels.unsqueeze(dim=1)
        self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")

        if global_appearance != "":
            loaded_iter = searchForMaxIteration(os.path.join(global_appearance, "point_cloud"))
            self.load_mlp_checkpoints(os.path.join(global_appearance, "point_cloud", "iteration_{:d}".format(loaded_iter)))
            self.freeze_mlp(self.embedding_appearance)

    def set_anchor_mask(self, cam_center, resolution_scale):
        dist = torch.sqrt(torch.sum((self.get_anchor - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, self.street_levels - 1)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)
    
    def set_gs_mask(self, cam_center, resolution_scale):
        dist = torch.sqrt(torch.sum((self._xyz - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, self.street_levels - 1)
        self._gs_mask = (self._level.squeeze(dim=1) <= int_level)

    def training_setup(self, training_args):
        
        self.anchor_opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_opacity_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros(self.get_anchor.shape[0]*self.n_offsets, dtype=torch.float, device="cuda")
        
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        ]
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
        if iteration % 1000 == 0:
            self.oneupSHdegree()

    def save_ply(self, path):
        def construct_list_of_attributes():
            l = []
            l.append('x')
            l.append('y')
            l.append('z')
            l.append('level')
            l.append('extra_level')
            for i in range(self._offset.shape[1]*self._offset.shape[2]):
                l.append('f_offset_{}'.format(i))
            for i in range(self._anchor_feat.shape[1]):
                l.append('f_anchor_feat_{}'.format(i))
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
        anchor_feats = self._anchor_feat.detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, offsets, anchor_feats, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        plydata = PlyData([el], obj_info=[
            'standard_dist {:.6f}'.format(self.standard_dist),
            'aerial_levels {:.6f}'.format(self.aerial_levels),
            'street_levels {:.6f}'.format(self.street_levels),
            ])
        plydata.write(path)
    
    def load_ply(self, path):
        plydata = PlyData.read(path)
        infos = plydata.obj_info
        for info in infos:
            var_name = info.split(' ')[0]
            self.__dict__[var_name] = float(info.split(' ')[1])
    
        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int16)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.aerial_levels = round(self.aerial_levels)
        self.street_levels = round(self.street_levels)
        self.active_sh_degree = self.max_sh_degree

    def prune_anchor(self, mask):
        if self.training_stage == 'coarse':
            mask = torch.bitwise_and(mask, (self._level < self.aerial_levels).squeeze())
        elif self.training_stage == 'fine':
            mask = torch.bitwise_and(mask, (self._level >= self.aerial_levels).squeeze())
        else:
            raise ValueError(f'invalid training stage {self.training_stage}')
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level = self._level[valid_points_mask]    
        self._extra_level = self._extra_level[valid_points_mask]

        return mask

    def anchor_growing(self, grads, opt, offset_mask, iteration):
        init_length = self.get_anchor.shape[0]
        grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        for cur_level in range(self.street_levels):
            update_value = self.fork ** opt.update_ratio
            if self.training_stage == 'coarse':
                add_level = cur_level
            elif self.training_stage == 'fine':
                add_level = max(cur_level + 1, self.aerial_levels)
            else:
                raise ValueError(f'invalid training stage {self.training_stage}')
            cur_level_mask = (self.get_level == cur_level).squeeze(dim=1)
            add_level_mask = (self.get_level == add_level).squeeze(dim=1)
            if torch.sum(cur_level_mask) == 0:
                continue
            cur_threshold = opt.densify_grad_threshold * (update_value ** cur_level)
            extra_threshold = cur_threshold * opt.extra_ratio
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            repeated_mask = repeat(cur_level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_extra_mask = torch.logical_and(candidate_extra_mask, cur_level_mask)
            if self.training_stage == 'coarse':
                candidate_extra_mask = torch.bitwise_and(candidate_extra_mask, (self._level < self.aerial_levels).squeeze())
            elif self.training_stage == 'fine':
                candidate_extra_mask = torch.bitwise_and(candidate_extra_mask, (self._level >= self.aerial_levels).squeeze())
            else:
                raise ValueError(f'invalid training stage {self.training_stage}')
            self._extra_level += opt.extra_up * candidate_extra_mask.float()    

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            voxel_size = self.voxel_size / (float(self.fork) ** (add_level - self.aerial_levels))
            
            grid_coords = torch.round(self.get_anchor[add_level_mask]/voxel_size - self.padding).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz/voxel_size - self.padding).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if opt.overlap:
                remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * voxel_size + self.padding * voxel_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * add_level
            elif selected_grid_coords_unique.shape[0] > 0:
                if grid_coords.shape[0] > 0:
                    remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                    remove_duplicates = ~remove_duplicates
                else:
                    remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*voxel_size + self.padding * voxel_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * add_level
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            if candidate_anchor.shape[0] > 0:
                weed_mask = self.weed_out(candidate_anchor, new_level)
                candidate_anchor = candidate_anchor[weed_mask]
                new_level = new_level[weed_mask]
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask

            if candidate_anchor.shape[0] > 0:
                # =========== [Fix OOM Start] ===========
                # 优化内存：避免显式创建巨大的重复张量，改用索引提取
                # 1. 找到 mask 中为 True 的索引位置
                valid_indices = torch.nonzero(candidate_mask).squeeze()
                
                # 2. 计算这些位置对应的 anchor 索引 (整除 n_offsets)
                anchor_indices = torch.div(valid_indices, self.n_offsets, rounding_mode='floor')
                
                # 3. 直接提取特征
                new_feat = self._anchor_feat[anchor_indices]
                # =========== [Fix OOM End] ===========
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*voxel_size # *0.05
                new_scaling = torch.log(new_scaling)
                
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation[:,0] = 1.0

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                new_level = new_level.unsqueeze(dim=1).float().cuda()
                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                
                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                }   

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.anchor_opacity_accum, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_opacity_accum
                self.anchor_opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)

    def save_mlp_checkpoints(self, path):#split or unite
        mkdir_p(os.path.dirname(path))
        self.eval()
        opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
        cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
        color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.appearance_dim).cuda()))
        color_mlp.save(os.path.join(path, 'color_mlp.pt'))
        if self.appearance_dim > 0:
            emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
            emd.save(os.path.join(path, 'embedding_appearance.pt'))
        self.train()

    def load_mlp_checkpoints(self, path):
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
        if self.appearance_dim > 0:
            self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()

    def create_from_pretrained(self, pcd, spatial_lr_scale, model_path, logger):
        points = torch.tensor(pcd.points).float().cuda()
        self.training_stage = "fine"
        self.spatial_lr_scale = spatial_lr_scale
        self.load_mlp_checkpoints(model_path)
        self.freeze()
        ply_path = os.path.join(model_path,"point_cloud.ply")

        plydata = PlyData.read(ply_path)
        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int16)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self.base_anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self.base_anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self.base_offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.base_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self.base_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def roll_back(self):
        base_mask = (self._level < self.aerial_levels).squeeze()
        self._anchor[base_mask] = self.base_anchor
        self._anchor_feat[base_mask] = self.base_anchor_feat
        self._offset[base_mask] = self.base_offset
        self._scaling[base_mask] = self.base_scaling
        self._rotation[base_mask] = self.base_rotation
        # [核心修复]：左右两边必须同时应用 mask，保证形状对齐
        # 1. 恢复位置
        # self._anchor[base_mask] = self.base_anchor[base_mask]
        # # 2. 恢复特征 (你报错的地方)
        # self._anchor_feat[base_mask] = self.base_anchor_feat[base_mask]
        # # 3. 恢复偏移
        # self._offset[base_mask] = self.base_offset[base_mask]
        # # 4. 恢复缩放
        # self._scaling[base_mask] = self.base_scaling[base_mask]
        # # 5. 恢复旋转
        # self._rotation[base_mask] = self.base_rotation[base_mask]

    # def roll_back(self):
    #     # 1. 计算当前模型中属于 Level 0 的掩膜
    #     base_mask = (self._level < self.aerial_levels).squeeze()

    #     # ================== [新增调试代码 Start] ==================
    #     current_total_count = self._anchor.shape[0]
    #     backup_total_count = self.base_anchor.shape[0]
    #     current_level0_count = base_mask.sum().item()
        
    #     print(f"\n[DEBUG ROLL_BACK] Iteration Check:")
    #     print(f"  > self._level : {self._level}")
    #     print(f"  > self.aerial_levels : {self.aerial_levels}")
    #     print(f"  > Current Model Total Anchors : {current_total_count}")
    #     print(f"  > Backup Snapshot Total Anchors: {backup_total_count}")
    #     print(f"  > Current Level 0 Anchors (Mask True): {current_level0_count}")
        
    #     # 检查是否会有形状不匹配
    #     if current_total_count != backup_total_count:
    #         print(f"  [!] WARNING: Anchor count mismatch! Diff: {current_total_count - backup_total_count}")
    #         # 如果你开启了致密化，这里肯定会不一致。
    #         # 如果不一致，使用 base_mask 去索引 base_anchor 可能会越界（如果 mask 长度 > backup 长度）
    #         if base_mask.shape[0] > self.base_anchor.shape[0]:
    #              print(f"  [!] CRITICAL: Mask length ({base_mask.shape[0]}) > Backup length ({self.base_anchor.shape[0]}). Indexing will fail!")
    #     else:
    #         print(f"  [OK] Anchor counts match.")
    #     # ================== [新增调试代码 End] ==================

    #     # [核心修复]：左右两边必须同时应用 mask
    #     # 注意：如果 densification 开启且点数增加了，下面这行 self.base_anchor[base_mask] 依然可能报错
    #     # 因为 base_mask 是针对“当前变长了的数组”生成的，直接套用到“旧的短数组”上会索引越界
    #     try:
    #         self._anchor[base_mask] = self.base_anchor[base_mask]
    #         self._anchor_feat[base_mask] = self.base_anchor_feat[base_mask]
    #         self._offset[base_mask] = self.base_offset[base_mask]
    #         self._scaling[base_mask] = self.base_scaling[base_mask]
    #         self._rotation[base_mask] = self.base_rotation[base_mask]
    #         print("  [SUCCESS] Roll back executed successfully.\n")
    #     except IndexError as e:
    #         print(f"  [ERROR] Roll back failed: {e}")
    #         # 这里可以选择 raise e 让程序停止，或者 pass 跳过
    #         raise e

    def save_explicit(self, path):
    
        def construct_list_of_attributes():
            l = ['x', 'y', 'z']
            l.append('level')
            l.append('extra_level')
            # All channels except the 3 DC
            for i in range(3):
                l.append('f_dc_{}'.format(i))
            for i in range(3 * (self.max_sh_degree + 1) ** 2 - 3):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(3):
                l.append('scale_{}'.format(i))
            for i in range(4):
                l.append('rot_{}'.format(i))
            return l

        anchor = self.get_anchor
        feat = self.get_anchor_feat
        levels = self._level
        extra_levels = self._extra_level.unsqueeze(dim=1)
        grid_offsets = self.get_offset
        grid_scaling = self.get_scaling

        # get offset's opacity
        neural_opacity = self.get_opacity_mlp(feat) # [N, k]

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)

        # select opacity 
        opacity = neural_opacity[mask]

        # get offset's color
        if self.appearance_dim > 0:
            camera_indicies = torch.zeros_like(feat[:,0], dtype=torch.long, device=feat.device)
            appearance = self.get_appearance(camera_indicies)
            color = self.get_color_mlp(torch.cat([feat, appearance], dim=1))
        else:
            color = self.get_color_mlp(feat)
        if self.color_attr == "RGB": 
            color = color.reshape([anchor.shape[0]*self.n_offsets, 3])# [mask]
        else:
            color = color.reshape([anchor.shape[0]*self.n_offsets, -1])# [mask]
        color_dim = color.shape[1]

        # get offset's cov
        scale_rot = self.get_cov_mlp(feat)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, 7]) # [mask]
        
        # offsets
        offsets = grid_offsets.view([-1, 3]) # [mask]
        
        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor, levels, extra_levels], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        
        scaling_repeat, repeat_anchor, repeat_levels, repeat_extra_levels, color, scale_rot, offsets = masked.split([6, 3, 1, 1, color_dim, 7, 3], dim=-1)
        
        # post-process cov
        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
        rot = self.rotation_activation(scale_rot[:,3:7])
        
        # post-process color
        color = color.view([color.shape[0], -1, 3])
        features_dc = color[:, 0:1, :]
        features_rest = color[:, 1:, :]
        
        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets 

        xyz = xyz.detach().cpu().numpy()
        level = repeat_levels.detach().cpu().numpy()
        extra_level = repeat_extra_levels.detach().cpu().numpy()
        f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scale = scaling.detach().cpu().numpy()
        rotation = rot.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, level, extra_level, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        
        plydata = PlyData([el], obj_info=[
            'standard_dist {:.6f}'.format(self.standard_dist),
            'aerial_levels {:.6f}'.format(self.aerial_levels),
            'street_levels {:.6f}'.format(self.street_levels),
            ])
        plydata.write(path)

    def load_explicit(self, path):
        plydata = PlyData.read(path)
        infos = plydata.obj_info
        for info in infos:
            var_name = info.split(' ')[0]
            self.__dict__[var_name] = float(info.split(' ')[1])
    
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int16)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._gs_mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        self.aerial_levels = round(self.aerial_levels)
        self.street_levels = round(self.street_levels)
        self.active_sh_degree = self.max_sh_degree