import torch


def denormalize_img(imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:, 0, :, :] = imgs[:, 0, :, :] * std[0] + mean[0]
    _imgs[:, 1, :, :] = imgs[:, 1, :, :] * std[1] + mean[1]
    _imgs[:, 2, :, :] = imgs[:, 2, :, :] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs

def denormalize_img2(imgs=None):
    # _imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs)

    return imgs / 255.0


###
# local pixel refinement
###

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_kernel():
    weight = torch.zeros(8, 1, 3, 3)
    weight[0, 0, 0, 0] = 1
    weight[1, 0, 0, 1] = 1
    weight[2, 0, 0, 2] = 1

    weight[3, 0, 1, 0] = 1
    weight[4, 0, 1, 2] = 1

    weight[5, 0, 2, 0] = 1
    weight[6, 0, 2, 1] = 1
    weight[7, 0, 2, 2] = 1

    return weight


class AFF(nn.Module):

    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)#对图像边缘进行填充
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        # masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)  # 对mask下采样
        # # fg_mask = (masks.argmax(dim=1)>0).float()
        # print("imgs.size)",imgs.size())#imgs.size() torch.Size([1, 3, 18, 32])

        b, c, h, w = imgs.shape
        _imgs = self.get_dilated_neighbors(imgs)#输出张量的每个像素位置都有一个48维的向量,包含了输入像素在不同膨胀率下的8个邻居的信息
        # print("_imgs.size()",_imgs.size())#_imgs.size() torch.Size([1, 3, 48, 18, 32])
        _pos = self.pos.to(_imgs.device)

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _pos_rep = _pos.repeat(b, 1, 1, h, w)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)#计算每个像素和邻域的绝对值
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)#计算每个像素和邻域的方差
        _pos_std = torch.std(_pos_rep, dim=self.dim, keepdim=True)

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)

        pos_aff = -(_pos_rep / (_pos_std + 1e-8) / self.w1) ** 2
        # pos_aff = pos_aff.mean(dim=1, keepdim=True)

        aff = F.softmax(aff, dim=2) + self.w2 * F.softmax(pos_aff, dim=2)
        # print("aff.size()",aff.size())#aff.size() torch.Size([1, 1, 48, 18, 32]


        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)#获取mask每个像素的邻域  mask [n, 21, 48, 18, 32]
            masks = (_masks * aff).sum(2)#和每个邻域的亲和度加权后相加   加入邻域信息
            # print(masks.size())#torch.Size([1, 21, 18, 32])

        return masks
