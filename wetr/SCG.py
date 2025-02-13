# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import torch
from torch import nn
from torch.nn import functional as F


def normalize(value, mode='plus', dim=1, keepdim=True, eps=1e-10):
    if mode == 'plus':
        return value / (torch.sum(value, dim=dim, keepdim=True) + eps)
    elif mode == 'norm1':
        # print(torch.max(value, dim=dim, keepdim=True)[0].size())
        return value / (torch.max(value, dim=dim, keepdim=True)[0] + eps)
    elif mode == 'minmax':
        if keepdim:
            min_value = torch.min(value, dim=dim, keepdim=True)[0]
            max_value = torch.max(value, dim=dim, keepdim=True)[0]
            return (value - min_value) / (max_value - min_value + eps)
        else:
            return (value - value.min()) / (value.max() - value.min() + eps)


def generate_SC(features, first_th=0.1, second_th=0.05, iteration=1):
    # print("first_th",first_th)
    b, c, h, w = features.size()

    # B, C, H, W -> B, H, W, C
    features = features.permute(0, 2, 3, 1).contiguous()
    # B, H, W, C -> B, H * W, C
    features = features.view(b, h * w, c)

    # frobenius normalization
    norm_f = features / (torch.norm(features, dim=2, keepdim=True) + 1e-10)

    # first order
    SC = F.relu(torch.matmul(norm_f, norm_f.transpose(1, 2)))
    # print("sc:",SC.min(), SC.max(), SC.size())

    if first_th > 0:
        SC[SC < first_th] = 0
    # print("SC",SC.size())
    SC1 = normalize(SC)

    # second order
    SC[:, torch.arange(h * w), torch.arange(w * h)] = 0
    SC = SC / (torch.sum(SC, dim=1, keepdim=True) + 1e-10)

    base_th = 1 / (h * w)
    second_th_2 = base_th * second_th

    SC2 = SC.clone()

    for _ in range(iteration):
        SC2 = torch.matmul(SC2, SC)
        SC2 = normalize(SC2)

    # print("sc2",SC2.min(), SC2.max(), SC2.size(), second_th_2)

    if second_th > 0:
        SC2[SC2 < second_th_2] = 0

    return SC1, SC2


def generate_att_SC(att,feature, first_th=0.45, second_th=0.22, iteration=1):
    SC = normalize(att,mode='minmax')
    # print("sc:",SC.min(), SC.max(), SC.size())
    if first_th > 0:
        SC[SC < first_th] = 0
    SC1 = normalize(SC)


    b, c, h, w = feature.size()
    # second order
    SC[:, torch.arange(h * w), torch.arange(w * h)] = 0
    SC = SC / (torch.sum(SC, dim=1, keepdim=True) + 1e-10)

    base_th = 1 / (h * w)
    second_th_2 = base_th * second_th

    SC2 = SC.clone()

    for _ in range(iteration):
        SC2 = torch.matmul(SC2, SC)
        SC2 = normalize(SC2)

    # print("sc2",SC2.min(), SC2.max(), SC2.size(), second_th_2)

    if second_th > 0:
        SC2[SC2 < second_th_2] = 0

    return SC1, SC2


def generate_SCM(cams, HSC, foreground_th=0.55, background_th=0.35):
    batch_size, classes, h, w = cams.size()
    ids = torch.arange(h * w)

    flatted_cams = cams.view(batch_size, classes, h * w)

    scms = []
    for b in range(batch_size):
        scm_per_batch = []
        for c in range(classes):
            # flatted_cam = cams[b][c].view(-1)
            flatted_cam = flatted_cams[b, c]

            fg_ids = ids[flatted_cam >= foreground_th]
            if len(fg_ids) > 0:
                fg_HSC = normalize(HSC[b, :, fg_ids], 'minmax', dim=0)

                fg_HSC = torch.sum(fg_HSC, dim=1).view((h, w))
                fg_HSC = normalize(fg_HSC, 'minmax', keepdim=False)
            else:
                fg_HSC = torch.zeros(h, w).cuda()

            bg_ids = ids[flatted_cam <= background_th]
            if len(bg_ids) > 0:
                bg_HSC = normalize(HSC[b, :, bg_ids], 'minmax', dim=0)

                bg_HSC = torch.sum(bg_HSC, dim=1).view((h, w))
                bg_HSC = normalize(bg_HSC, 'minmax', keepdim=False)
            else:
                bg_HSC = torch.zeros(h, w).cuda()

            scm = F.relu(fg_HSC - bg_HSC)

            scm = normalize(scm, 'minmax', keepdim=False)
            scm_per_batch.append(scm)

        scm = torch.stack(scm_per_batch)
        scms.append(scm)

    return torch.stack(scms)


def get_HSC(SC1, SC2, second_weight=2):
    return torch.max(SC1, second_weight * SC2)
    # return second_weight * SC2



def generate_HSC(fs):
    HSC_list = []

    for f in fs:
        SC1, SC2 = generate_SC(f)

        HSC = get_HSC(SC1, SC2)
        HSC = normalize(HSC)

        HSC_list.append(HSC)

    HSC = torch.sum(torch.stack(HSC_list), dim=0)
    return HSC


def generate_att_HSC(att,feature):
    att = att + att.permute(0, 2, 1)
    SC1, SC2 = generate_att_SC(att,feature)
    HSC = get_HSC(SC1, SC2)
    HSC = normalize(HSC)
    return HSC