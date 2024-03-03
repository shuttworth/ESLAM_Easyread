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
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/Tracker.py
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
import copy
import os
import time

from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (matrix_to_cam_pose, cam_pose_to_matrix, get_samples)
from src.utils.datasets import get_dataset
from src.utils.Frame_Visualizer import Frame_Visualizer

class Tracker(object):
    """
    Tracking main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        eslam (ESLAM): ESLAM object
    """
    def __init__(self, cfg, args, eslam):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']

        self.idx = eslam.idx
        self.bound = eslam.bound
        self.mesher = eslam.mesher
        self.output = eslam.output
        self.verbose = eslam.verbose
        self.renderer = eslam.renderer
        self.gt_c2w_list = eslam.gt_c2w_list
        self.mapping_idx = eslam.mapping_idx
        self.mapping_cnt = eslam.mapping_cnt
        self.shared_decoders = eslam.shared_decoders
        self.estimate_c2w_list = eslam.estimate_c2w_list
        self.truncation = eslam.truncation

        self.shared_planes_xy = eslam.shared_planes_xy
        self.shared_planes_xz = eslam.shared_planes_xz
        self.shared_planes_yz = eslam.shared_planes_yz

        self.shared_c_planes_xy = eslam.shared_c_planes_xy
        self.shared_c_planes_xz = eslam.shared_c_planes_xz
        self.shared_c_planes_yz = eslam.shared_c_planes_yz

        self.cam_lr_T = cfg['tracking']['lr_T']
        self.cam_lr_R = cfg['tracking']['lr_R']
        self.device = cfg['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.w_sdf_fs = cfg['tracking']['w_sdf_fs']
        self.w_sdf_center = cfg['tracking']['w_sdf_center']
        self.w_sdf_tail = cfg['tracking']['w_sdf_tail']
        self.w_depth = cfg['tracking']['w_depth']
        self.w_color = cfg['tracking']['w_color']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False,
                                       num_workers=1, pin_memory=True, prefetch_factor=2)

        self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy

        self.decoders = copy.deepcopy(self.shared_decoders)

        self.planes_xy = copy.deepcopy(self.shared_planes_xy)
        self.planes_xz = copy.deepcopy(self.shared_planes_xz)
        self.planes_yz = copy.deepcopy(self.shared_planes_yz)

        self.c_planes_xy = copy.deepcopy(self.shared_c_planes_xy)
        self.c_planes_xz = copy.deepcopy(self.shared_c_planes_xz)
        self.c_planes_yz = copy.deepcopy(self.shared_c_planes_yz)

        for p in self.decoders.parameters():
            p.requires_grad_(False)

    # 函数作用：计算符号距离函数 (SDF) 的损失
    # 输入 SDF值、采样深度值和地面真值深度
    # 输出sdf_losses，表示 SDF 的自由空间损失、中心损失和尾部损失的加权和
    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        # 创建四种掩码
        # 第一种front_mask：标记点的掩码，这些点的深度小于[地面真值深度gt_depth-截断距离truncation]
        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        # 第二种back_mask：标记点的掩码，这些点的深度大于[地面真值深度gt_depth+截断距离truncation]
        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        # 第三种center_mask：标记点的掩码，这些点的深度深度在 (地面真值深度gt_depth - 0.4 * 截断距离truncation) 和 (地面真值深度gt_depth + 0.4 * 截断距离truncation) 之间
        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        # 第四种：除了上述三种情况之外的点的掩码(对这些掩码都执行非运算)
        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        # 自由空间损失，对应公式(6)
        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        
        # 中心损失，对应公式(8)前半段
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        
        # 尾部损失，对应公式(8)后半段
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        # 以上三种损失加权得到sdf_losses
        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    # 核心函数：相机位姿优化，内部包含重要函数get_samples，render_batch_ray等，计算loss损失并反向传播
    def optimize_tracking(self, cam_pose, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            cam_pose (tensor): camera pose.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        # 变量的定义与初始化，注意这里用上了与NICE-SLAM不一样的all_planes，此参数后续传入render_batch_ray()函数[在Renderer.py中调用]
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = cam_pose_to_matrix(cam_pose)

        # 采样一批射线，得到方向和原点 (batch_rays_o, batch_rays_d)、以及对应的深度和颜色 (batch_gt_depth, batch_gt_color)
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
                                                                                 self.ignore_edge_W, W-self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w,
                                                                                 gt_depth, gt_color, device)

        # should pre-filter those out of bounding box depth value
        with torch.no_grad():
            det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(
                device) - det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_gt_depth
            inside_mask = inside_mask & (batch_gt_depth > 0)

        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]

        # 调用重要函数render_batch_ray()，此函数在Renderer.py里已经给大家仔细讲解
        # 从返回值中可以拿到深度，颜色，sdf，深度，用于下述环节计算losses
        depth, color, sdf, z_vals = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d, batch_rays_o,
                                                                   self.device, self.truncation, gt_depth=batch_gt_depth)

        ## 此部分继承了NICE-SLAM的动态处理模块"Robust to Dynamic Objects"，滤除超过10倍中值深度误差的值
        ## Filtering the rays for which the rendered depth error is greater than 10 times of the median depth error
        depth_error = (batch_gt_depth - depth.detach()).abs()
        error_median = depth_error.median()
        depth_mask = (depth_error < 10 * error_median)

        ## SDF losses，内部有公式(6)-(8)的对应
        loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

        ## Color Loss，对应公式(10)
        loss = loss + self.w_color * torch.square(batch_gt_color - color)[depth_mask].mean()

        ### Depth loss，对应公式(9)
        loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()
        
        ### 截止以上，loss不断累加直至完毕，此对应着公式(11)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def update_params_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.
        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')

            # 更新解码器参数
            self.decoders.load_state_dict(self.shared_decoders.state_dict())

            # 更新平面参数
            for planes, self_planes in zip(
                    [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz],
                    [self.planes_xy, self.planes_xz, self.planes_yz]):
                for i, plane in enumerate(planes):
                    self_planes[i] = plane.detach()

            for c_planes, self_c_planes in zip(
                    [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz],
                    [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]):
                for i, c_plane in enumerate(c_planes):
                    self_c_planes[i] = c_plane.detach()

            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        """
            Runs the tracking thread for the input RGB-D frames.

            Args:
                None

            Returns:
                None
        """
        device = self.device
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader, smoothing=0.05)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            gt_color = gt_color.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            gt_c2w = gt_c2w.to(device, non_blocking=True)

            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            idx = idx[0]

            # 不再有NICE-SLAM里strict/loose/free的模式区分，此处仅保留“strict mapping and then tracking”
            # initiate mapping every self.every_frame frames
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                while self.mapping_idx[0] != idx - 1:
                    time.sleep(0.001)
                pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)

            self.update_params_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.save_imgs(idx, 0, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders)

            else:
                if self.const_speed_assumption and idx - 2 >= 0:
                    ## Linear prediction for initialization
                    ## 基于恒定速度假设的预测
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                    pre_poses = matrix_to_cam_pose(pre_poses)
                    cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]
                else:
                    ## Initialize with the last known pose
                    ## 未启用恒定速度假设时，用前一个姿态进行预测
                    cam_pose = matrix_to_cam_pose(pre_c2w)

                T = torch.nn.Parameter(cam_pose[:, -3:].clone())
                R = torch.nn.Parameter(cam_pose[:,:4].clone())
                cam_para_list_T = [T]
                cam_para_list_R = [R]
                # Adam优化器optimizer_camera的初始化
                optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas':(0.5, 0.999)},
                                                     {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas':(0.5, 0.999)}])

                current_min_loss = torch.tensor(float('inf')).float().to(device)
                # for循环内，优化相机姿态
                for cam_iter in range(self.num_cam_iters):
                    cam_pose = torch.cat([R, T], -1)

                    # 可视化
                    self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, all_planes, self.decoders)

                    # 调用optimize_tracking()函数优化相机pose，记录最优current_min_loss
                    loss = self.optimize_tracking(cam_pose, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_pose = cam_pose.clone().detach()

                c2w = cam_pose_to_matrix(candidate_cam_pose)

            self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
            pre_c2w = c2w.clone()
            self.idx[0] = idx