import torch
import torch.nn.functional as F
from .imutils import denormalize_img, encode_cmap
from .dcrf import crf_inference_label
import numpy as np
import imageio
from utils import imutils

def dlcam_to_label(cam, cls_label, img_box=None, ignore_mid=False, cfg=None):
    b, c, h, w = cam.shape# 21
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
        [1, 1, h, w])
    # print(cls_label_rep)
    valid_cam = cam * cls_label_rep
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)#如果cam进行标签清洗则会最大值不对
    _pseudo_label[cam_value <= cfg.dlcam.low_thre] = cfg.dataset.ignore_index#cfg.dataset.ignore_index
    if img_box is None:
        return _pseudo_label

    # _pseudo_label = cam.argmax(dim=1)

    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return pseudo_label

def get_neigbor_tensors(X: torch.tensor, n: int = 4, entrophy=False):
    """
    Args:
        X: tensor of shape (B, C, H, W)
        n: number of neighbors to get 4 or 8
    Returns:
        list of tensors of shape (B, C, H, W)
    """
    assert n in [4, 8]
    if entrophy:
        X = X.unsqueeze(1)
        X = torch.nn.functional.pad(X, (1, 1, 1, 1, 0, 0, 0, 0)).squeeze(1)
    else:
        X = torch.nn.functional.pad(X, (1, 1, 1, 1, 0, 0, 0, 0))

    neigbors = []
    if n == 8:
        for i, j in [(None, -2), (1, -1), (2, None)]:
            for k, l in [(None, -2), (1, -1), (2, None)]:
                if i == k and i == 1:
                    continue
                if entrophy:
                    neigbors.append(X[:, i:j, k:l])
                else:
                    neigbors.append(X[:, :, i:j, k:l])
    elif n == 4:
        if entrophy:
            neigbors.append(X[:, :-2, 1:-1])
            neigbors.append(X[:, 2:, 1:-1])
            neigbors.append(X[:, 1:-1, :-2])
            neigbors.append(X[:, 1:-1, 2:])
        else:
            neigbors.append(X[:, :, :-2, 1:-1])
            neigbors.append(X[:, :, 2:, 1:-1])
            neigbors.append(X[:, :, 1:-1, :-2])
            neigbors.append(X[:, :, 1:-1, 2:])
    return neigbors

def dlcam_to_label_neighbors(cam, cls_label, img_box=None, ignore_mid=False, cfg=None):
    b, c, h, w = cam.shape# 21
    pascal_E_joint = [0.9980886695132567, 0.9860495622941258, 0.9555923573386919, 0.9888166500147213,
     0.9861891841249937, 0.9894536447668415, 0.9928481727772317, 0.9899944432748093, 0.9939554498312092,
      0.9723734804881907, 0.9814734877790432, 0.992167780556063, 0.9921082109870506, 0.9904260573555301,
       0.9877020724598903, 0.9890704556150036, 0.9771896568239452, 0.9865771364049042, 0.9878175644560833,
        0.9948415228771397, 0.993315147276186]
    E = torch.tensor(pascal_E_joint).view((1, c, 1, 1)).to(cam.device)
    neighbors = torch.stack(get_neigbor_tensors(cam, n=4, entrophy=False)) #[4, n, c, h, w]
    k_neighbors, neigbor_idx = torch.topk(neighbors, k=1, axis=0)##[1, n, c, h, w]
    for neighbor in k_neighbors:
        beta = torch.exp(torch.tensor(-1 / 2))  # for more neigbors use neigbor_idx
        cam = cam + beta * neighbor - (torch.max(cam * neighbor, cam * E) * beta)

    cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
        [1, 1, h, w])
    # print(cls_label_rep)
    valid_cam = cam * cls_label_rep
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)#如果cam进行标签清洗则会最大值不对
    _pseudo_label[cam_value <= cfg.dlcam.low_thre] = cfg.dataset.ignore_index#cfg.dataset.ignore_index
    if img_box is None:
        return _pseudo_label

    # _pseudo_label = cam.argmax(dim=1)

    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return pseudo_label

def dlcam_to_label_dpa_neighbors(cam, cls_label, img_box=None, ignore_mid=False, cfg=None,n_iter=None,max_iters=None,max_margin=80,margin_width=40):
    b, c, h, w = cam.shape# 21
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
        [1, 1, h, w])

    _topk, _ = torch.topk(cam, 2, dim=1)
    _margin = _topk[:, 0, :, :] - _topk[:, 1, :, :]  # shape: batch_size, h, w
    mask = torch.ones(b,h,w).cuda()

    if img_box is None:
        mask = mask * cfg.dataset.ignore_index
        for idx, coord in enumerate(img_box):
            mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1
    percent_unreliable = margin_width * (1 - n_iter / max_iters)#随着训练越来越小
    drop_percent = max_margin - percent_unreliable
    # drop_percent = 60


    thresh = np.percentile(_margin[mask != 255].detach().cpu().numpy().flatten(), 100-drop_percent)

    b, c, h, w = cam.shape# 21
    pascal_E_joint = [0.9980886695132567, 0.9860495622941258, 0.9555923573386919, 0.9888166500147213,
     0.9861891841249937, 0.9894536447668415, 0.9928481727772317, 0.9899944432748093, 0.9939554498312092,
      0.9723734804881907, 0.9814734877790432, 0.992167780556063, 0.9921082109870506, 0.9904260573555301,
       0.9877020724598903, 0.9890704556150036, 0.9771896568239452, 0.9865771364049042, 0.9878175644560833,
        0.9948415228771397, 0.993315147276186]
    E = torch.tensor(pascal_E_joint).view((1, c, 1, 1)).to(cam.device)
    neighbors = torch.stack(get_neigbor_tensors(cam, n=4, entrophy=False)) #[4, n, c, h, w]
    k_neighbors, neigbor_idx = torch.topk(neighbors, k=1, axis=0)##[1, n, c, h, w]
    for neighbor in k_neighbors:
        beta = torch.exp(torch.tensor(-1 / 2))  # for more neigbors use neigbor_idx
        cam = cam + beta * neighbor - (torch.max(cam * neighbor, cam * E) * beta)

    topk, _ = torch.topk(cam, 2, dim=1)
    margin = topk[:, 0, :, :] - topk[:, 1, :, :]  # shape: batch_size, h, w
    thresh_mask = margin.le(thresh).bool() * (mask != 255).bool()
    valid_cam = cam * cls_label_rep
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)#如果cam进行标签清洗则会最大值不对
    _pseudo_label[thresh_mask] = cfg.dataset.ignore_index


    if img_box is None:
        return _pseudo_label

    # _pseudo_label = cam.argmax(dim=1)

    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]


    return pseudo_label

def dlcam_to_label_dpa(cam, cls_label, img_box=None, ignore_mid=False, cfg=None,n_iter=None,max_iters=None,max_margin=80,margin_width=40):
    b, c, h, w = cam.shape# 21
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
        [1, 1, h, w])
    # print(cls_label_rep)
    valid_cam = cam * cls_label_rep
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)#如果cam进行标签清洗则会最大值不对
    # _pseudo_label[cam_value <= cfg.dlcam.low_thre] = cfg.dataset.ignore_index#cfg.dataset.ignore_index

    # entropy = -torch.sum(cam * torch.log(cam + 1e-10), dim=1)  # shape: batch_size, h, w
    _topk, _ = torch.topk(cam, 2, dim=1)
    _margin = _topk[:, 0, :, :] - _topk[:, 1, :, :]  # shape: batch_size, h, w
    mask = torch.ones_like(_pseudo_label)
    # if img_box is not None:
    #     mask = mask * cfg.dataset.ignore_index
    #     for idx, coord in enumerate(img_box):
    #         mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1
    percent_unreliable = margin_width * (1 - n_iter / max_iters)#随着训练越来越小
    drop_percent = max_margin - percent_unreliable
    # drop_percent = 60

    thresh = np.percentile(_margin[mask != 255].detach().cpu().numpy().flatten(), 100-drop_percent)
    thresh_mask = _margin.le(thresh).bool() * (mask != 255).bool()
    _pseudo_label[thresh_mask] = cfg.dataset.ignore_index

    if img_box is None:
        return _pseudo_label

    # _pseudo_label = cam.argmax(dim=1)

    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]


    return pseudo_label

def dlcam_to_label_dpa2(cam, cls_label, img_box=None, ignore_mid=False, cfg=None,n_iter=None,max_iters=None,max_margin=70,min_margin=40):
    b, c, h, w = cam.shape# 21
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
        [1, 1, h, w])
    # print(cls_label_rep)
    valid_cam = cam * cls_label_rep
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)#如果cam进行标签清洗则会最大值不对
    # _pseudo_label[cam_value <= cfg.dlcam.low_thre] = cfg.dataset.ignore_index#cfg.dataset.ignore_index

    # entropy = -torch.sum(cam * torch.log(cam + 1e-10), dim=1)  # shape: batch_size, h, w
    _topk, _ = torch.topk(cam, 2, dim=1)
    _margin = _topk[:, 0, :, :] - _topk[:, 1, :, :]  # shape: batch_size, h, w
    mask = torch.ones_like(_pseudo_label)
    # if img_box is not None:
    #     mask = mask * cfg.dataset.ignore_index
    #     for idx, coord in enumerate(img_box):
    #         mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1
    if n_iter / max_iters <0.75:
        percent_unreliable = (max_margin-min_margin) * ( n_iter / (max_iters*0.75))#随着训练越来越小
        percent = max_margin - percent_unreliable

    else:
        percent = min_margin

    thresh = np.percentile(_margin[mask != 255].detach().cpu().numpy().flatten(), percent)
    thresh_mask = _margin.le(thresh).bool() * (mask != 255).bool()
    _pseudo_label[thresh_mask] = cfg.dataset.ignore_index

    if img_box is None:
        return _pseudo_label

    # _pseudo_label = cam.argmax(dim=1)

    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]


    return pseudo_label

def dlcam_to_label_dpa3(cam, cls_label, img_box=None, ignore_mid=False, cfg=None,n_iter=None,max_iters=None,margin=20):
    b, c, h, w = cam.shape# 21
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
        [1, 1, h, w])
    # print(cls_label_rep)
    valid_cam = cam * cls_label_rep
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)#如果cam进行标签清洗则会最大值不对
    # _pseudo_label[cam_value <= cfg.dlcam.low_thre] = cfg.dataset.ignore_index#cfg.dataset.ignore_index

    # entropy = -torch.sum(cam * torch.log(cam + 1e-10), dim=1)  # shape: batch_size, h, w
    _topk, _ = torch.topk(cam, 2, dim=1)
    _margin = _topk[:, 0, :, :] - _topk[:, 1, :, :]  # shape: batch_size, h, w
    mask = torch.ones_like(_pseudo_label)
    # if img_box is not None:
    #     mask = mask * cfg.dataset.ignore_index
    #     for idx, coord in enumerate(img_box):
    #         mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1
    # percent_unreliable = margin_width * ( n_iter / max_iters)#随着训练越来越小

    thresh = np.percentile(_margin[mask != 255].detach().cpu().numpy().flatten(), margin)
    thresh_mask = _margin.le(thresh).bool() * (mask != 255).bool()
    _pseudo_label[thresh_mask] = cfg.dataset.ignore_index

    if img_box is None:
        return _pseudo_label

    # _pseudo_label = cam.argmax(dim=1)

    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]


    return pseudo_label

def dlcam_to_label_dpa_marginmax_width(cam, cls_label, img_box=None, ignore_mid=False, cfg=None,n_iter=None,max_iters=None,max_margin=70,margin_width=50):
    b, c, h, w = cam.shape# 21
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
        [1, 1, h, w])
    # print(cls_label_rep)
    valid_cam = cam * cls_label_rep
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)#如果cam进行标签清洗则会最大值不对
    # _pseudo_label[cam_value <= cfg.dlcam.low_thre] = cfg.dataset.ignore_index#cfg.dataset.ignore_index

    # entropy = -torch.sum(cam * torch.log(cam + 1e-10), dim=1)  # shape: batch_size, h, w
    _topk, _ = torch.topk(cam, 2, dim=1)
    _margin = _topk[:, 0, :, :] - _topk[:, 1, :, :]  # shape: batch_size, h, w
    mask = torch.ones_like(_pseudo_label)
    # if img_box is not None:
    #     mask = mask * cfg.dataset.ignore_index
    #     for idx, coord in enumerate(img_box):
    #         mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1
    percent_unreliable = margin_width * ( n_iter / max_iters)#随着训练越来越小

    thresh = np.percentile(_margin[mask != 255].detach().cpu().numpy().flatten(), max_margin-percent_unreliable)
    thresh_mask = _margin.le(thresh).bool() * (mask != 255).bool()
    _pseudo_label[thresh_mask] = cfg.dataset.ignore_index

    if img_box is None:
        return _pseudo_label

    # _pseudo_label = cam.argmax(dim=1)

    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]


    return pseudo_label

def dlcrosscam_to_label(cam, cls_label, img_box=None, ignore_mid=False, cfg=None):
    b, c, h, w = cam.shape# 21
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
        [1, 1, h, w])
    # print(cls_label_rep)
    valid_cam = cam * cls_label_rep
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)#如果cam进行标签清洗则会最大值不对
    cls_thes,_ = valid_cam.reshape(b,c,-1).max(dim=-1)  #1 21
    # print(_pseudo_label.size())# n h w
    # print(cam_value.size())# n h w
    # 将n 21 变为n h w 21 然后用_pseudo_label n h w 挑选变为n h w
    # print(cfg.dlcam.cross_low_thre*cls_thes.unsqueeze(-1).unsqueeze(-1).size())
    # print(cam_value<cam_value)

    _pseudo_label[cam_value <= cfg.dlcam.cross_low_thre] = 21#cfg.dataset.ignore_index


    # _pseudo_label[cam_value <= cam_value] = cfg.dataset.ignore_index#最大的多少0.*倍

    #通过熵的大小球
    # entropy = -torch.sum(valid_cam * torch.log2(valid_cam + 1e-10),dim=1)
    # # print(entropy)
    # _pseudo_label[entropy >= cfg.dlcam.cross_low_thre] = cfg.dataset.ignore_index#cfg.dataset.ignore_index


    if img_box is None:
        return _pseudo_label

    # _pseudo_label = cam.argmax(dim=1)

    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return pseudo_label

def cam_to_label(cam, cls_label, img_box=None, ignore_mid=False, cfg=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    # print("cls_label_rep.size()",cls_label_rep.size())
    # print("cam.size()",cam.size())

    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value <= cfg.cam.bkg_score] = 0

    if ignore_mid:
        _pseudo_label[cam_value <= cfg.cam.high_thre] = cfg.dataset.ignore_index
        _pseudo_label[cam_value <= cfg.cam.low_thre] = 0

    if img_box is None:
        return _pseudo_label

    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return valid_cam, pseudo_label


def ignore_img_box(label, img_box, ignore_index):
    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label


def cam_to_fg_bg_label(imgs, cams, cls_label, bg_thre=0.3, fg_thre=0.6):
    scale = 2
    imgs = F.interpolate(imgs, size=(imgs.shape[2] // scale, imgs.shape[3] // scale), mode="bilinear",
                         align_corners=False)
    cams = F.interpolate(cams, size=imgs.shape[2:], mode="bilinear", align_corners=False)

    b, c, h, w = cams.shape
    # print("1",imgs.size())#4 3 256 256
    _imgs = denormalize_img(imgs=imgs)
    # print("2",_imgs.size())#4 3 256 256

    cam_label = torch.ones(size=(b, h, w), ).to(cams.device)
    bg_label = torch.ones(size=(b, 1), ).to(cams.device)
    _cls_label = torch.cat((bg_label, cls_label), dim=1)

    lt_pad = torch.ones(size=(1, h, w), ).to(cams.device) * bg_thre
    ht_pad = torch.ones(size=(1, h, w), ).to(cams.device) * fg_thre

    for i in range(b):
        keys = torch.nonzero(_cls_label[i, ...])[:, 0]
        # print(keys)
        n_keys = _cls_label[i, ...].cpu().numpy().sum().astype(np.uint8)
        valid_cams = cams[i, keys[1:] - 1, ...]

        lt_cam = torch.cat((lt_pad, valid_cams), dim=0)#加入背景之后的cam
        ht_cam = torch.cat((ht_pad, valid_cams), dim=0)
        # print("lt_cam.size()",lt_cam.size())#2 512 512
        _, cam_label_lt = lt_cam.max(dim=0)#获得最大值的索引 相当于argmax
        # print("cam_label_lt.size()",cam_label_lt.size())#
        _, cam_label_ht = ht_cam.max(dim=0)
        # print(_imgs[i,...].shape)
        _images = _imgs[i, ...].permute(1, 2, 0).cpu().numpy().astype(np.uint8)#对img进行变换
        # print("3",_images.shape)#256 256 3
        _cam_label_lt = cam_label_lt.cpu().numpy()
        _cam_label_ht = cam_label_ht.cpu().numpy()
        # print("_images.shape", _images.shape)#256 256 3
        # print("_cam_label_lt.shape", _cam_label_lt.shape)#256 256
        # print("n_keys.shape", n_keys.shape)
        _cam_label_lt_crf = crf_inference_label(_images, _cam_label_lt, n_labels=n_keys)#输入原图 预测像素级类别 分类类别
        _cam_label_lt_crf_ = keys[_cam_label_lt_crf]
        # print("_cam_label_lt_crf_.size()",_cam_label_lt_crf_.size())#256 256
        # print(_cam_label_lt_crf_)
        _cam_label_ht_crf = crf_inference_label(_images, _cam_label_ht, n_labels=n_keys)
        _cam_label_ht_crf_ = keys[_cam_label_ht_crf]#256 256
        # print("_cam_label_ht_crf_.size()",_cam_label_ht_crf_.size())
        # print(_cam_label_ht_crf_)
        # _cam_label_lt_crf = torch.from_numpy(_cam_label_lt_crf).to(cam_label.device)
        # _cam_label_ht_crf = torch.from_numpy(_cam_label_ht_crf).to(cam_label.device)

        cam_label[i, ...] = _cam_label_ht_crf_
        cam_label[i, _cam_label_ht_crf_ == 0] = 255
        cam_label[i, (_cam_label_ht_crf_ + _cam_label_lt_crf_) == 0] = 0
        # imageio.imsave("out.png", encode_cmap(cam_label[i,...].cpu().numpy()))
        # cam_label_lt

    return cam_label

def multi_scale_dlcam(model, inputs, scales,cls_labels,cfg,dl_aff):
    cam_list = []

    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        _cam = model(inputs_cat, dlcam_only=True,cfg=cfg,class_label=cls_labels,dl_aff=dl_aff,inputs_denorm=inputs_denorm)
        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        cls_label_rep = F.pad(cls_labels.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
            [1, 1, h, w])
        _cam = _cam * cls_label_rep
        cam_list = [F.relu(_cam)]
        n=len(scales)
        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)#[1, 3, 256, 256]
                inputs_denorm = imutils.denormalize_img2(inputs.clone())
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)
                _cam = model(inputs_cat, dlcam_only=True,cfg=cfg,class_label=cls_labels,dl_aff=dl_aff,inputs_denorm=inputs_denorm)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = _cam * cls_label_rep
                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam /= n
    return cam

def multi_scale_cam(model, inputs, scales):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=((h-1)//4+1, (w-1)//4+1), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        cam_list = [F.relu(_cam)]
        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=((h-1)//4+1, (w-1)//4+1), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam

def multi_scale_cam_orign(model, inputs, scales):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam

def upsize_multi_scale_cam(model, inputs, scales):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        cam_list = [F.relu(_cam)]
        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam

def multi_scale_cam_with_aff_mat(model, inputs, scales):
    cam_list, aff_mat = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam, _aff_mat = model(inputs_cat, cam_only=True)
        aff_mat.append(_aff_mat)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam, _aff_mat = model(inputs_cat, cam_only=True)
                aff_mat.append(_aff_mat)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

    max_aff_mat = aff_mat[np.argmax(scales)]
    return cam, max_aff_mat

def refine_dlcam_to_label(cam, ref_mod=None, images=None,cls_label=None, img_box=None, ignore_mid=False, cfg=None, down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)
    cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
        [1, 1, h, w])
    # print(cls_label_rep)
    valid_cam = cam * cls_label_rep
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)#如果cam进行标签清洗则会最大值不对

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cam.device)
    cls_labels = torch.cat((bkg_cls, cls_label), dim=1)

    refined_label_h = torch.ones(size=(b, h, w)) * cfg.dataset.ignore_index
    refined_label_h = refined_label_h.to(cam.device)

    _cams_with_bkg_h = F.interpolate(cam, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        refined_cams = ref_mod(_images[[idx], ...], valid_cams_h)
        refined_cams = F.interpolate(refined_cams, size=(h, w), mode="bilinear", align_corners=False)
        _refined_label_h = refined_cams.argmax(dim=1)
        _refined_label_h = valid_key[_refined_label_h]

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
    refined_label_h[cam_value <= cfg.dlcam.low_thre] = cfg.dataset.ignore_index#cfg.dataset.ignore_index

    return refined_label_h

def refine_dlcam_to_label_val(cam, ref_mod=None, images=None,cls_label=None, img_box=None, ignore_mid=False, cfg=None, down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)
    cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat(
        [1, 1, h, w])
    # print(cls_label_rep)
    valid_cam = cam * cls_label_rep
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)#如果cam进行标签清洗则会最大值不对

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cam.device)
    cls_labels = torch.cat((bkg_cls, cls_label), dim=1)

    refined_label_h = torch.ones(size=(b, h, w)) * cfg.dataset.ignore_index
    refined_label_h = refined_label_h.to(cam.device)

    _cams_with_bkg_h = F.interpolate(cam, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    for idx in range(b):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)


        refined_cams = ref_mod(_images[[idx], ...], valid_cams_h)
        refined_cams = F.interpolate(refined_cams, size=(h, w), mode="bilinear", align_corners=False)
        _refined_label_h = refined_cams.argmax(dim=1)
        _refined_label_h = valid_key[_refined_label_h]

        refined_label_h[idx,...] = _refined_label_h[0,...]

    return refined_label_h

def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None, img_box=None,
                            down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * cfg.cam.high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * cfg.cam.low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * cfg.dataset.ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
                                        valid_key=valid_key, orig_size=(h, w))
        # print(_refined_label_h.size())#1 512 512

        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = cfg.dataset.ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def refine_cams_with_bkg_v3(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None, img_box=None,
                            down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_l = torch.ones(size=(b, 1, h, w)) * cfg.cam.low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * cfg.dataset.ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_l = refined_label.clone()

    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    return refined_label_l


def refine_cams_with_bkg_v2_crf(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None, img_box=None,
                            down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)#对image下采样了两倍

    _imgs = F.interpolate(images, size=(h , w), mode="bilinear", align_corners=False)
    # print("1",_imgs.size())
    _crf_imgs = denormalize_img(imgs=_imgs)
    # print("2",_crf_imgs.size())

    bkg_h = torch.ones(size=(b, 1, h, w)) * cfg.cam.high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * cfg.cam.low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)  #分类标签

    refined_label = torch.ones(size=(b, h, w)) * cfg.dataset.ignore_index #先初始化为不确定
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        n_keys = cls_labels[idx, ...].cpu().numpy().sum().astype(np.uint8)
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _crf_imgs_idx = _crf_imgs[idx, ...].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # print("3",_crf_imgs_idx.shape)

        _refined_label_h = _refine_cams_wcrf(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
                                        valid_key=valid_key, orig_size=(h, w), dnormal_img=_crf_imgs_idx, n_labels=n_keys).unsqueeze(0)
        # print(_refined_label_h.size())

        _refined_label_l = _refine_cams_wcrf(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w), dnormal_img=_crf_imgs_idx, n_labels=n_keys).unsqueeze(0)

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = cfg.dataset.ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def refine_cams_with_bkg_val(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None, down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * cfg.cam.high_thre
    # bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * cfg.cam.low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * cfg.dataset.ignore_index
    refined_label = refined_label.to(cams.device)
    # refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    # cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    # _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
    #                                  align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    # print(_cams_with_bkg_h.size())# 1 21 h w
    for idx in range(b):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        # valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        # _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
        #                                 valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))

        # refined_label_h[idx,...] = _refined_label_h[0, ...]
        refined_label_l[idx, ...] = _refined_label_l[0, ...]

    # refined_label = refined_label_h.clone()
    # refined_label[refined_label_h == 0] = cfg.dataset.ignore_index
    # refined_label[(refined_label_h + refined_label_l) == 0] = 0
    # # print(refined_label.size()) # 1 h w

    return refined_label_l


def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label

def _refine_cams_wcrf(ref_mod, images, cams, valid_key, orig_size, dnormal_img, n_labels):
    refined_cams = ref_mod(images, cams)#image是下采样两倍的 cams是原图大小的
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    # print("refined_label",refined_label.size())#应该是512*512
    _refined_label = refined_label.squeeze(0).cpu().numpy()
    # print("dnormal_img.shape",dnormal_img.shape)#1 512 3
    # print("_refined_label.shape",_refined_label.shape)#512 512
    # print("n_labels.shape",n_labels.shape)
    _cam_label_lt_crf = crf_inference_label(dnormal_img, _refined_label, n_labels=n_labels)
    refined_label = valid_key[_cam_label_lt_crf]

    return refined_label


def refine_cams_with_cls_label(ref_mod=None, images=None, labels=None, cams=None, img_box=None):
    refined_cams = torch.zeros_like(cams)
    b = images.shape[0]

    # bg_label = torch.ones(size=(b, 1),).to(labels.device)
    cls_label = labels

    for idx, coord in enumerate(img_box):
        _images = images[[idx], :, coord[0]:coord[1], coord[2]:coord[3]]

        _, _, h, w = _images.shape
        _images_ = F.interpolate(_images, size=[h // 2, w // 2], mode="bilinear", align_corners=False)

        valid_key = torch.nonzero(cls_label[idx, ...])[:, 0]
        valid_cams = cams[[idx], :, coord[0]:coord[1], coord[2]:coord[3]][:, valid_key, ...]

        _refined_cams = ref_mod(_images_, valid_cams)
        _refined_cams = F.interpolate(_refined_cams, size=_images.shape[2:], mode="bilinear", align_corners=False)

        refined_cams[idx, valid_key, coord[0]:coord[1], coord[2]:coord[3]] = _refined_cams[0, ...]

    return refined_cams


def cams_to_affinity_label(cam_label, mask=None, ignore_index=255):
    b, h, w = cam_label.shape

    cam_label_resized = F.interpolate(cam_label.unsqueeze(1).type(torch.float32), size=[h // 16, w // 16],
                                      mode="nearest")

    _cam_label = cam_label_resized.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    # aff_label[(_cam_label_rep+_cam_label_rep_t) == 0] = ignore_index
    for i in range(b):

        if mask is not None:
            aff_label[i, mask == 0] = ignore_index

        aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index

    return aff_label


def propagte_aff_cam(cams, aff=None, mask=None):
    b, c, h, w = cams.shape
    n_pow = 2
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            aff[i, mask == 0] = 0

    # cams = F.interpolate(cams, size=[h//16, w//16], mode="bilinear", align_corners=False).detach()
    cams_rw = cams.clone()

    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-4)

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _cams = cams[i].reshape(c, -1)
        _aff = aff[i]
        _cams_rw = torch.matmul(_cams, _aff)
        cams_rw[i] = _cams_rw.reshape(cams_rw[i].shape)

    # cams_rw = F.interpolate(cams_rw, size=[h, w], mode="bilinear", align_corners=False)

    return cams_rw


def propagte_aff_cam_with_bkg(cams, aff=None, mask=None, cls_labels=None, bkg_score=None):
    b, _, h, w = cams.shape

    bkg = torch.ones(size=(b, 1, h, w)) * bkg_score
    bkg = bkg.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    cams_with_bkg = torch.cat((bkg, cams), dim=1)

    cams_rw = torch.zeros_like(cams_with_bkg)

    ##########

    b, c, h, w = cams_with_bkg.shape
    n_pow = 2
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            aff[i, mask == 0] = 0

    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-1)  ## avoid nan

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _cams = cams_with_bkg[i].reshape(c, -1)
        valid_key = torch.nonzero(cls_labels[i, ...])[:, 0]
        _cams = _cams[valid_key, ...]
        _cams = F.softmax(_cams, dim=0)
        _aff = aff[i]
        _cams_rw = torch.matmul(_cams, _aff)
        cams_rw[i, valid_key, :] = _cams_rw.reshape(-1, cams_rw.shape[2], cams_rw.shape[3])

    return cams_rw