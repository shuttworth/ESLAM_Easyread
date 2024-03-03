# This file is a part of ESLAM.
#
# ESLAM is a NeRF-based SLAM system. It utilizes Neural Radiance Fields (NeRF)
# to perform Simultaneous Localization and Mapping (SLAM) in real-time.
# This software is the implementation of the paper "ESLAM: Efficient Dense SLAM
# System Based on Hybrid Representation of Signed Distance Fields" by
# Mohammad Mahdi Johari, Camilla Carta, and Francois Fleuret.
#
# Copyright 2023 ams-OSRAM AG
#
# Author: Mohammad Mahdi Johari <mohammad.johari@idiap.ch>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/utils/Renderer.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2022 Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, Marc Pollefeys
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

import torch
from src.common import get_rays, sample_pdf, normalize_3d_coordinate

class Renderer(object):
    """
    Renderer class for rendering depth and color.
    Args:
        cfg (dict): configuration.
        eslam (ESLAM): ESLAM object.
        ray_batch_size (int): batch size for sampling rays.
    """
    def __init__(self, cfg, eslam, ray_batch_size=10000):
        # 初始化射线批处理的大小(渲染过程中每次处理的射线数量)
        self.ray_batch_size = ray_batch_size

        # 配置是否在射线采样时加入随机扰动、分层采样的数量和重要性采样的数量
        # configs/ESLAM.yaml里，perturb = True，n_importance = 8, n_stratified = 32
        self.perturb = cfg['rendering']['perturb']
        self.n_stratified = cfg['rendering']['n_stratified']
        self.n_importance = cfg['rendering']['n_importance']

        # 获取用于场景渲染的缩放因子
        self.scale = cfg['scale']
        # 获取场景边界
        self.bound = eslam.bound.to(eslam.device, non_blocking=True)

        # 提取图像的高度和宽度、焦距（fx、fy）以及主点坐标（cx、cy）
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy


    # 函数作用：对射线上采样的深度值添加扰动，在render_batch_ray()中调用
    def perturbation(self, z_vals):
        """
        Add perturbation to sampled depth values on the rays.
        Args:
            z_vals (tensor): sampled depth values on the rays. 射线上采样的深度值
        Returns:
            z_vals (tensor): perturbed depth values on the rays. 添加了扰动的深度值张量
        """
        # get intervals between samples
        # 使用 z_vals 中相邻深度值的平均值计算每对采样点之间的中点
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 得到上界和下界
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        # 生成随机扰动
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        # 应用扰动
        return lower + (upper - lower) * t_rand


    # 重点:渲染的核心函数，用于渲染射线的深度和颜色，在Tracker.py和Mapper.py中得到调用
    def render_batch_ray(self, all_planes, decoders, rays_d, rays_o, device, truncation, gt_depth=None):
        """
        Render depth and color for a batch of rays.
        Args:
            all_planes (Tuple): all feature planes. 所有特征平面
            decoders (torch.nn.Module): decoders for TSDF and color. 用于TSDF和颜色的解码器
            rays_d (tensor): ray directions. 射线方向
            rays_o (tensor): ray origins. 射线原点
            device (torch.device): device to run on. 运行设备
            truncation (float): truncation threshold. 截断阈值
            gt_depth (tensor): ground truth depth. 真实深度
        Returns:
            depth_map (tensor): depth map. 深度图
            color_map (tensor): color map. 颜色图
            volume_densities (tensor): volume densities for sampled points. 采样点的体积密度
            z_vals (tensor): sampled depth values on the rays. 射线上采样的深度值

        """
        n_stratified = self.n_stratified
        n_importance = self.n_importance
        n_rays = rays_o.shape[0]

        # 初始化z_vals，用来存储所有采样点的深度值
        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=device)
        # near为近平面距离
        near = 0.0
        ### 重点：t_vals_uni和t_vals_surface,分别用于‘分层采样N_strat’和‘表面采样N_imp’的均匀分布采样点
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

        # 具有真实深度的像素
        ### pixels with gt depth:
        gt_depth = gt_depth.reshape(-1, 1)
        gt_mask = (gt_depth > 0).squeeze()
        gt_nonezero = gt_depth[gt_mask]

        # 在真实深度（表面）周围采样点，分别计算表面采样z_vals_surface和分层采样z_vals_free的深度值
        ## Sampling points around the gt depth (surface)
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        
        ### 对应论文原文： "For pixels with ground truth depths, the N_imp additional points are sampled uniformly 
        # inside the truncation distance T w.r.t. the depth measurement"
        z_vals_surface = gt_depth_surface - (1.5 * truncation)  + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
        if self.perturb:
            z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero

        ### pixels without gt depth (importance sampling):
        # gt_mask.all(): 如果张量gt_mask中的所有元素都是True，则返回True，否则返回False
        # gt_mask.all()什么时候会返回False？当至少有一条射线没有击中物体表面，即存在一些射线没有有效的深度信息的时候
        # 当满足上述False条件，则使用重要性采样来估计深度
        if not gt_mask.all():
            with torch.no_grad():
                # 分离没有真实深度的射线原点和方向
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                # 计算射线与边界框的交点
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                # 基于统一分布和边界框计算深度值
                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                if self.perturb:
                    z_vals_uni = self.perturbation(z_vals_uni)
                # 计算统一采样点的位置
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]

                # 标准化统一采样点坐标，计算SDF和alpha值，用于权重计算
                pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
                sdf_uni = decoders.get_raw_sdf(pts_uni_nor, all_planes)
                sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                alpha_uni = self.sdf2alpha(sdf_uni, decoders.beta)
                weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=device)
                                                        , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]

                # 计算权重并进行重要性采样，更新深度值
                z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False, device=device)
                z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                z_vals[~gt_mask] = z_vals_uni

        # 根据深度值计算采样点位置，获取原始SDF、alpha值、权重和渲染结果
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]

        raw = decoders(pts, all_planes)

        # 对应公式(4)
        alpha = self.sdf2alpha(raw[..., -1], decoders.beta)
        
        # 对应公式(5)的上半部分
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
                                                , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

        # 对应公式(5)的下半部分
        rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)

        return rendered_depth, rendered_rgb, raw[..., -1], z_vals

    # 函数目的：将有符号距离函数（SDF）值转换为透明度（alpha）值，实现公式(4)的内容
    def sdf2alpha(self, sdf, beta=10):
        """
        torch.sigmoid(-sdf * beta): 将SDF值映射到 [0, 1] 区间
        """
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

    # 函数目的：渲染深度和彩色图像
    def render_img(self, all_planes, decoders, c2w, truncation, device, gt_depth=None):
        """
        Renders out depth and color images.
        Args:
            all_planes (Tuple): feature planes
            decoders (torch.nn.Module): decoders for TSDF and color.
            c2w (tensor, 4*4): camera pose.
            truncation (float): truncation distance.
            device (torch.device): device to run on.
            gt_depth (tensor, H*W): ground truth depth image.
        Returns:
            rendered_depth (tensor, H*W): rendered depth image.
            rendered_rgb (tensor, H*W*3): rendered color image.

        """
        with torch.no_grad():
            # 获取图像尺寸和射线方向
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            # 初始化空列表
            depth_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            # 遍历所有射线，每次取 ray_batch_size 个射线进行分批渲染
            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                # 根据 gt_depth 的存在与否，分别调用 render_batch_ray 函数，实现具体渲染过程
                if gt_depth is None:
                    ret = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=gt_depth_batch)

                # 从 ret 中解包深度值 depth 和颜色值 color，并丢弃其他返回值
                depth, color, _, _ = ret
                depth_list.append(depth.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            color = color.reshape(H, W, 3)

            return depth, color