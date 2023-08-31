import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from torch_scatter import scatter_add, segment_coo

from .dvgo import Raw2Alpha, Alphas2Weights, render_utils_cuda, total_variation_cuda, MaskCache


#----------------------------added by jhaprk-----------------------------------

from torch_scatter import segment_coo
from .lowrank_volume import  TensorDVGO,TensorDVGORes,TensorDVGODeform
from codec import encode_motion,decode_motion
from torch.utils.cpp_extension import load

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#-----------------------------------------------------------------------------

'''Model'''
class DirectMPIGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, mpi_depth=0,
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=0, rgbnet = None, use_pca = True, cfg = None, rgbfeat_sigmoid=False,   
                 use_res=False, use_deform='',
                 **kwargs):
        super(DirectMPIGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self.cfg=cfg
        
        print("----------use res dvgo-------", use_res)
        print("----------use deform dvgo-------", use_deform)
        self.use_deform=use_deform

        self.rgbfeat_sigmoid=rgbfeat_sigmoid
        
        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('dmpigo: set density bias shift to', self.act_shift)
        
        # determine init grid resolution
        self._set_grid_resolution(num_voxels, mpi_depth)

        # init density voxel grid
        self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))
        with torch.no_grad():
            g = np.full([mpi_depth], 1./mpi_depth - 1e-6)
            p = [1-g[0]]
            for i in range(1, len(g)):
                p.append((1-g[:i+1].sum())/(1-g[:i].sum()))
            for i in range(len(p)):
                self.density[..., i].fill_(np.log(p[i] ** (-1/self.voxel_size_ratio) - 1))
            self.density[..., -1].fill_(10)
            
        if use_deform:
            self.former_density = torch.tensor(torch.zeros([1, 1, *self.world_size]))
            self.deformation_field = torch.nn.Parameter(torch.zeros([1, 3, *self.world_size])) 
        # init color representation
        # feature voxel grid + shallow MLP  (fine stage)
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        
        # ------------------- added by jhpark --------------------
        
        self.k0_class = TensorDVGO

        self.use_pca = use_pca
        self.use_res = use_res
        if use_pca:
            self.k0_class = TensorDCT
        if use_res:
            self.k0_class = TensorDVGORes
        elif use_deform:
            self.k0_class = TensorDVGODeform

        self.rgbnet_full_implicit = rgbnet_full_implicit
        
        # --------------------------------------------------------
        
        
        
        
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            
            # modified by jhaprk
            self.k0 = TensorDVGO(tuple(self.world_size),self.xyz_min.device, feat_dim= self.k0_dim)
            #self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            
            
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            
            # modified by jhaprk
            self.k0 = self.k0_class(tuple(self.world_size),self.xyz_min.device, feat_dim= self.k0_dim,cfg=cfg)
            #self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            self.posfreq = torch.FloatTensor([(2**i) for i in range(10)]).cuda()
            dim0 = (3+3*viewbase_pe*2)
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3
            
            
            # ---------------edited by jhpark -------------------------
            
            self.rgbnet = rgbnet.set_params(dim0,rgbnet_width,rgbnet_depth)
            
            """
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            """
            # ---------------------------------------------------------------
            
        print('dmpigo: self.density.shape', self.density.shape)
        #print('dmpigo: self.k0.shape', self.k0.shape)
        print('dmpigo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        
        """
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = MaskCache(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(self.world_size), dtype=torch.bool)
        self.mask_cache = MaskCache(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        """

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256. / mpi_depth
        print('dmpigo: world_size      ', self.world_size)
        print('dmpigo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'mpi_depth': self.mpi_depth,
            'alpha_init': self.alpha_init,
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'fast_color_thres': self.fast_color_thres,
            'use_pca': self.use_pca,
            'use_res': self.use_res,
            'use_deform': self.use_deform,
            'rgbfeat_sigmoid':self.rgbfeat_sigmoid,
            **self.rgbnet_kwargs,
        }
        
    #---------------------------------modified by jhpark -----------------------------------------
    @torch.no_grad()
    def load_pretrain(self,filepath,current_frame):
        ckpt=torch.load(filepath)

        k0 = ckpt['model_state_dict']['k0.k0'].to(self.k0.device)

        if current_frame==1:
            self.k0.former_k0 =k0.clone()
            self.k0.former_k0_cur= torch.tensor(
            F.interpolate(k0, size=tuple(self.world_size), mode='trilinear', align_corners=True),requires_grad = False)

        else:
            former_k0 = ckpt['model_state_dict']['k0.former_k0'].to(self.k0.device)
            self.k0.former_k0=(k0+former_k0).clone()
            self.k0.former_k0_cur = torch.tensor(
                F.interpolate(self.k0.former_k0, size=tuple(self.world_size), mode='trilinear', align_corners=True),
                requires_grad=False)
            
    @torch.no_grad()
    def load_pretrain_deform_res(self, filepath, deform_res_stage="res",deform_low_reso=False):
        if deform_res_stage=="res":
            ckpt = torch.load(filepath)


            k0 = ckpt['model_state_dict']['k0.k0'].to(self.k0.device)
            deformation_field = ckpt['model_state_dict']['deformation_field'].to(self.k0.device)
            if deform_low_reso:
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<using low deformation field>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # deform_origin_size=deformation_field.size()
                # print(deform_origin_size)
                # deform_low_size=(int(deform_origin_size[2]/8),int(deform_origin_size[3]/8),int(deform_origin_size[4]/8))
                # deformation_field_low= F.interpolate(deformation_field, size=deform_low_size, mode='trilinear', align_corners=True)
                # deformation_field= F.interpolate(deformation_field_low, size=deform_origin_size[2:], mode='trilinear', align_corners=True)
                deform, grid_size, origin_size=encode_motion(deformation_field)
                deformation_field=decode_motion(deform,grid_size,origin_size)


            print("deform size", deformation_field.size())
            print("self.density size", self.density.size())
            #print("density size", density.size())
            print("k0 size", k0.size())

            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], deformation_field.shape[2]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], deformation_field.shape[3]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], deformation_field.shape[4]),
            ), -1)
            deform_xyz = self.deform_warp(self_grid_xyz, deformation_field)


            self.k0.former_k0 = self.grid_sampler(deform_xyz, k0).permute(3, 0, 1, 2).unsqueeze(0) #因为是tenor，而不是k0
            print(self.k0.former_k0.size())

            if self.cfg.density_deform:
                density = ckpt['model_state_dict']['density'].to(self.k0.device)
                density = self.grid_sampler(deform_xyz, density).unsqueeze(-1).permute(3, 0, 1, 2).unsqueeze(0)
                self.density = torch.nn.Parameter(
                    F.interpolate(density, size=tuple(self.world_size), mode='trilinear', align_corners=True))
            self.k0.former_k0_cur = torch.tensor(
                F.interpolate(self.k0.former_k0, size=tuple(self.world_size), mode='trilinear', align_corners=True),
                requires_grad=False)
        elif deform_res_stage=="deform":
            ckpt = torch.load(filepath)

            density = ckpt['model_state_dict']['density'].to(self.k0.device)
            k0 = ckpt['model_state_dict']['k0.k0'].to(self.k0.device)
            former_k0 = ckpt['model_state_dict']['k0.former_k0'].to(self.k0.device)
            self.k0.former_k0 = (k0 + former_k0).clone()
            self.former_density = density
            self.density = torch.nn.Parameter(
                F.interpolate(self.former_density.data, size=tuple(self.world_size), mode='trilinear',
                              align_corners=True),
                requires_grad=False)
            self.k0.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.former_k0.data, size=tuple(self.world_size), mode='trilinear',
                              align_corners=True),
                requires_grad=False)

    @torch.no_grad()
    def load_pretrain_deform(self, filepath, current_frame,deform_from_start=False ):
        ckpt = torch.load(filepath)

        density = ckpt['model_state_dict']['density'].to(self.k0.device)
        k0 = ckpt['model_state_dict']['k0.k0'].to(self.k0.device)


        if current_frame == 1 or deform_from_start:
            print("---------deform from start--------------")
            self.former_density = density
            self.k0.former_k0 = k0
            self.density = torch.nn.Parameter(
                F.interpolate(self.former_density.data, size=tuple(self.world_size), mode='trilinear',
                              align_corners=True),
                requires_grad=False)
            self.k0.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.former_k0.data, size=tuple(self.world_size), mode='trilinear',
                              align_corners=True),
                requires_grad=False)
        else:

            deformation_field= ckpt['model_state_dict']['deformation_field'].to(self.k0.device)
            print("deform size", deformation_field.size())
            print("self.density size", self.density.size())
            print("k0 size", k0.size())
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], density.shape[2]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], density.shape[3]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], density.shape[4]),
            ), -1)
            deform_xyz=self.deform_warp(self_grid_xyz,deformation_field)
            self.former_density = self.grid_sampler(deform_xyz, density).unsqueeze(0).unsqueeze(0)
            print(self.former_density.size())
            self.k0.former_k0 = self.grid_sampler(deform_xyz, k0).permute(3,0,1,2).unsqueeze(0)
            print(self.k0.former_k0.size())
            self.density = torch.nn.Parameter(
                F.interpolate(self.former_density.data, size=tuple(self.world_size), mode='trilinear',
                              align_corners=True),
                requires_grad=False)
            self.k0.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.former_k0.data, size=tuple(self.world_size), mode='trilinear',
                              align_corners=True),
                requires_grad=False)


    @torch.no_grad()
    def test_pretrain(self, filepath, stage):
        ckpt = torch.load(filepath)

        k0 = ckpt['model_state_dict']['k0.k0'].to(self.k0.device)

        print(torch.all(self.k0.former_k0==
            F.interpolate(k0, size=tuple(self.world_size), mode='trilinear', align_corners=True)))

    # -----------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, mpi_depth):
        print('dmpigo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        print('dmpigo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        if self.use_deform:
            self.deformation_field = torch.nn.Parameter(
                F.interpolate(self.deformation_field.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
            self.density = torch.nn.Parameter(
                F.interpolate(self.former_density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.density = torch.nn.Parameter(
                F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
   
        # --------------modified by jhpark-------------------------------------------
        if self.k0_dim > 0:
            self.k0.upsample_volume_grid(tuple(self.world_size))
            #self.vp.upsample_volume_grid(tuple(self.world_size))
        else:
            self.k0 = self.k0_class(tuple(self.world_size),self.xyz_min.device, feat_dim= self.k0_dim)
        
        """
        self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        """
        # ---------------------------------------------------------------------------
        print('dmpigo: scale_volume_grid finish')

    def density_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        total_variation_cuda.total_variation_add_grad(
            self.density, self.density.grad, wxy, wxy, wz, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        total_variation_cuda.total_variation_add_grad(
            self.k0, self.k0.grad, wxy, wxy, wz, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

    
    # -----------------------------modified by jhpark------------------------------
    """
    def grid_sampler(self, xyz, grid):
        '''Wrapper for the interp operation'''
        num_ch = grid.shape[1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
        ret = ret.reshape(num_ch,-1).T.squeeze(1)
        return ret
    """
    
    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1])
            for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst
    
    
    def grid_sampler_deform(self, xyz,  mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]

        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1

        deform_vector = F.grid_sample(self.deformation_field, ind_norm, mode=mode, align_corners=align_corners).\
            reshape(self.deformation_field.shape[1],-1).T.reshape(*shape, self.deformation_field.shape[1])

        xyz=xyz+deform_vector

        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        grid=self.density
        # TODO: use `rearrange' to make it readable
        ret_lst = F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1], -1).T.reshape(
                *shape, grid.shape[1])

        if ret_lst.shape[-1] == 1:
            ret_lst = ret_lst.squeeze(-1)

        return ret_lst

    def grid_sampler_new(self, xyz, tensor, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        shape = xyz.shape[:-1]
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1

        ret = tensor.compute_features(ind_norm).reshape(*shape, tensor.real_dim)

        return ret.squeeze(-1)

    def grid_sampler_new_deform(self, xyz, tensor, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        shape = xyz.shape[:-1]
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1

        deform_vector = F.grid_sample(self.deformation_field, ind_norm, mode=mode, align_corners=align_corners). \
            reshape(self.deformation_field.shape[1], -1).T.reshape(*shape, self.deformation_field.shape[1])

        xyz = xyz + deform_vector

        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1

        ret = tensor.compute_features(ind_norm).reshape(*shape,tensor.real_dim)
        
        return ret.squeeze(-1)
    
    
    # -----------------------------------------------------------------------------
    
    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        assert near==0 and far==1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth-1)/stepsize) + 1
        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).view(-1,1).expand_as(mask_inbbox)[mask_inbbox]
            step_id = torch.arange(mask_inbbox.shape[1]).view(1,-1).expand_as(mask_inbbox)[mask_inbbox]
        return ray_pts, ray_id, step_id
    
    def deform_warp(self,xyz,deformation_field, align_corners=True):
        mode = 'bilinear'

        shape = xyz.shape[:-1]

        xyz_r = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz_r - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1

        deform_vector = F.grid_sample(deformation_field, ind_norm, mode=mode, align_corners=align_corners). \
            reshape(deformation_field.shape[1], -1).T.reshape(*shape, deformation_field.shape[1])

        xyz = xyz + deform_vector

        return xyz

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio


        # query for alpha w/ post-activation
        density = self.grid_sampler(ray_pts, self.density)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            weights = weights[mask]

        # query for color
        vox_emb = self.grid_sampler_new(ray_pts, self.k0)
        #vox_emb = self.grid_sampler(ray_pts, self.k0)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(vox_emb)
        else:
            # view-dependent color emission
            # ------------------added by jhpark -----------------------------
            torch.cuda.empty_cache()
            #viewdirs = viewdirs.cpu # viewdirs를 GPU로 이동
            #self.viewfreq = self.viewfreq.to(device)  # self.viewfreq를 GPU로 이동  
            # --------------------------------------------------------------------  
            self.viewfreq=self.viewfreq.to(viewdirs.device)  
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * step_id),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict


@functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1,1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1,-1).expand(shape).flatten()
    return ray_id, step_id

''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''

"""
class MaskCache(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super().__init__()
        print("coarse mask ",path)
        if path is not None:
            st = torch.load(path,map_location=device)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_kwargs']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            # mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask
"""

''' Misc
'''
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, frame_ids):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    frame_ids_tr = torch.ones([len(rgb_tr), H, W])
    imsz = [H*W]*len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=Ks[i], c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        frame_ids_tr[i] = frame_ids_tr[i]*frame_ids[i]
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_ids_tr


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, frame_ids):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW) and len(rgb_tr_ori)==len(frame_ids)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    frame_ids_tr = []
    top = 0
    for c2w, img, (H, W), K, id in zip(train_poses, rgb_tr_ori, HW, Ks, frame_ids):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        frame_ids_tr.append(torch.ones(n)*id)
        top += n

    assert top == N
    frame_ids_tr = torch.cat(frame_ids_tr)
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_ids_tr


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model,frame_ids, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW) and len(rgb_tr_ori)==len(frame_ids)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    frame_ids_tr = []
    top = 0
    for c2w, img, (H, W), K, id in zip(train_poses, rgb_tr_ori, HW, Ks, frame_ids):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)

        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.hit_coarse_geo(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        frame_ids_tr.append(torch.ones(n)*id)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    frame_ids_tr = torch.cat(frame_ids_tr)
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_ids_tr


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS



