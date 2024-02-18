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
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/conv_onet/models/decoder.py
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
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate

class Decoders(nn.Module):
    """
    Decoders for SDF and RGB.
    Args:
        c_dim: feature dimensions
        hidden_size: hidden size of MLP
        truncation: truncation of SDF
        n_blocks: number of MLP blocks
        learnable_beta: whether to learn beta

    """
    def __init__(self, c_dim=32, hidden_size=16, truncation=0.08, n_blocks=2, learnable_beta=True):
        super().__init__()

        self.c_dim = c_dim
        self.truncation = truncation
        self.n_blocks = n_blocks

        ## layers for SDF decoder
        # 用于存放SDF（有符号距离函数）解码器的线性层
        self.linears = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

        ## layers for RGB decoder
        # 这些层用于处理RGB颜色输出
        self.c_linears = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size)  for i in range(n_blocks - 1)])

        # self.output_linear: 一个线性层，用于SDF解码器的输出，将 hidden_size 转换为 1，输出SDF值
        self.output_linear = nn.Linear(hidden_size, 1)
        # self.c_output_linear：一个线性层，用于RGB解码器的输出，将 hidden_size 转换为 3，输出RGB颜色
        self.c_output_linear = nn.Linear(hidden_size, 3)

        if learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10

    # 函数作用：从平面中采样特征，在get_raw_sdf()函数中调用
    def sample_plane_feature(self, p_nor, planes_xy, planes_xz, planes_yz):
        """
        Sample feature from planes
        Args:
            p_nor (tensor): normalized 3D coordinates
            planes_xy (list): xy planes
            planes_xz (list): xz planes
            planes_yz (list): yz planes
        Returns:
            feat (tensor): sampled features
        """
        # 将标准化的三维坐标扩展为采样网格
        vgrid = p_nor[None, :, None]

        feat = [] # 初始化一个列表，后续用来存储采样得到的特征
        # 遍历每个平面进行特征采样，调用F.grid_sample方法
        for i in range(len(planes_xy)):
            # 从xy平面采样特征
            xy = F.grid_sample(planes_xy[i], vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            # 从xz平面采样特征
            xz = F.grid_sample(planes_xz[i], vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            # 从yz平面采样特征
            yz = F.grid_sample(planes_yz[i], vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            ### 对这一行公式的细致解析：
            # F.grid_sample: 这是PyTorch(torch.nn.functional)提供的一个函数grid_sample，用于根据给定的网格坐标在输入的平面上进行采样
            # vgrid[..., [0, 1]]: 索引 [0, 1] 选择了其中的X和Y坐标，用于在XY平面上进行采样；同理，xz平面是索引[0, 2]，yz平面是索引[1, 2]
            # padding_mode='border': 指定了当采样点落在输入数据的边界之外时的行为。'border' 表示会使用边界上的值
            # mode='bilinear': 指定了采样使用的插值方法，'bilinear' 表示双线性插值
            ### 论文对照：以上内容对应论文的公式(1)与公式(2)

            # 将三个方向的特征相加并存储到feat中
            feat.append(xy + xz + yz)
        # 将所有特征沿最后一个维度拼接起来
        feat = torch.cat(feat, dim=-1)

        return feat

    # 函数目的：从特征平面中，计算原始的有符号距离函数（SDF）
    def get_raw_sdf(self, p_nor, all_planes):
        """
        Get raw SDF
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            sdf (tensor): raw SDF
        """
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        # 从XY、XZ、YZ方向的特征平面中采样特征，注意，只用到了planes_xy, planes_xz, planes_yz，没有用到c_planes_xy, c_planes_xz, c_planes_yz
        feat = self.sample_plane_feature(p_nor, planes_xy, planes_xz, planes_yz)

        h = feat  # 将采样得到的特征赋值给变量h作为输入
        # 遍历所有线性层进行处理
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)  # 应用线性变换
            h = F.relu(h, inplace=True)  # 应用ReLU激活函数
        sdf = torch.tanh(self.output_linear(h)).squeeze()  # 通过输出层获取SDF值，并使用tanh函数进行归一化
        ### 对应公式(3)前半部分

        return sdf


    # 函数目的：从特征平面中，计算原始的颜色（RGB）
    def get_raw_rgb(self, p_nor, all_planes):
        """
        Get raw RGB
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            rgb (tensor): raw RGB
        """
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        c_feat = self.sample_plane_feature(p_nor, c_planes_xy, c_planes_xz, c_planes_yz)

        h = c_feat
        for i, l in enumerate(self.c_linears):
            h = self.c_linears[i](h)
            h = F.relu(h, inplace=True)
        rgb = torch.sigmoid(self.c_output_linear(h))
        ### 对应公式(3)后半部分

        return rgb

    # 对于任何继承自 torch.nn.Module 的类，forward 方法不会直接被显式调用
    # 当对该类的实例进行调用时，PyTorch会自动执行 forward 方法
    def forward(self, p, all_planes):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            raw (tensor): raw SDF and RGB
        """
        p_shape = p.shape

        p_nor = normalize_3d_coordinate(p.clone(), self.bound)

        sdf = self.get_raw_sdf(p_nor, all_planes)
        rgb = self.get_raw_rgb(p_nor, all_planes)

        raw = torch.cat([rgb, sdf.unsqueeze(-1)], dim=-1)
        raw = raw.reshape(*p_shape[:-1], -1)

        return raw
