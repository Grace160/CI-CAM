import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from scipy import misc,ndimage
from PIL import Image

def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16), :]


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


def tensorboard_image(imgs=None, cam=None, ):
    ## images

    _imgs = denormalize_img(imgs=imgs)
    grid_imgs = torchvision.utils.make_grid(tensor=_imgs, nrow=2)

    cam = F.interpolate(cam, size=_imgs.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.cpu()
    cam_max = cam.max(dim=1)[0]
    cam_heatmap = plt.get_cmap('jet')(cam_max.numpy())[:, :, :, 0:3] * 255
    cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
    cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
    grid_cam = torchvision.utils.make_grid(tensor=cam_img.type(torch.uint8), nrow=2)

    return grid_imgs, grid_cam


def tensorboard_edge(edge=None, n_row=2):
    ##
    edge = F.interpolate(edge, size=[224, 224], mode='bilinear', align_corners=False)[:, 0, ...]
    edge = edge.cpu()
    edge_heatmap = plt.get_cmap('viridis')(edge.numpy())[:, :, :, 0:3] * 255
    edge_cmap = torch.from_numpy(edge_heatmap).permute([0, 3, 1, 2])

    grid_edge = torchvision.utils.make_grid(tensor=edge_cmap.type(torch.uint8), nrow=n_row)

    return grid_edge


def tensorboard_attn(attns=None, size=[224, 224], n_pix=0, n_row=4):
    n = len(attns)
    imgs = []
    for idx, attn in enumerate(attns):

        b, hw, _ = attn.shape
        h = w = int(np.sqrt(hw))

        attn_ = attn.clone()  # - attn.min()
        # attn_ = attn_ / attn_.max()
        _n_pix = int(h * n_pix) * (w + 1)
        attn_ = attn_[:, _n_pix, :].reshape(b, 1, h, w)

        attn_ = F.interpolate(attn_, size=size, mode='bilinear', align_corners=True)

        attn_ = attn_.cpu()[:, 0, :, :]

        def minmax_norm(x):
            for i in range(x.shape[0]):
                x[i, ...] = x[i, ...] - x[i, ...].min()
                x[i, ...] = x[i, ...] / x[i, ...].max()
            return x

        attn_ = minmax_norm(attn_)

        attn_heatmap = plt.get_cmap('viridis')(attn_.numpy())[:, :, :, 0:3] * 255
        attn_heatmap = torch.from_numpy(attn_heatmap).permute([0, 3, 2, 1])
        imgs.append(attn_heatmap)
    attn_img = torch.cat(imgs, dim=0)

    grid_attn = torchvision.utils.make_grid(tensor=attn_img.type(torch.uint8), nrow=n_row).permute(0, 2, 1)

    return grid_attn


def tensorboard_attn2(attns=None, size=[224, 224], n_pixs=[0.0, 0.3, 0.6, 0.9], n_row=4, with_attn_pred=True):
    n = len(attns)
    attns_top_layers = []
    attns_last_layer = []
    grid_attns = []
    if with_attn_pred:
        _attns_top_layers = attns[:-3]
        _attns_last_layer = attns[-3:-1]
    else:
        _attns_top_layers = attns[:-2]
        _attns_last_layer = attns[-2:]

    attns_top_layers = [_attns_top_layers[i][:, 0, ...] for i in range(len(_attns_top_layers))]
    if with_attn_pred:
        attns_top_layers.append(attns[-1])
    grid_attn_top_case0 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[0], n_row=n_row)
    grid_attn_top_case1 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[1], n_row=n_row)
    grid_attn_top_case2 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[2], n_row=n_row)
    grid_attn_top_case3 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[3], n_row=n_row)
    grid_attns.append(grid_attn_top_case0)
    grid_attns.append(grid_attn_top_case1)
    grid_attns.append(grid_attn_top_case2)
    grid_attns.append(grid_attn_top_case3)

    for attn in _attns_last_layer:
        for i in range(attn.shape[1]):
            attns_last_layer.append(attn[:, i, :, :])
    grid_attn_last_case0 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[0], n_row=2 * n_row)
    grid_attn_last_case1 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[1], n_row=2 * n_row)
    grid_attn_last_case2 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[2], n_row=2 * n_row)
    grid_attn_last_case3 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[3], n_row=2 * n_row)
    grid_attns.append(grid_attn_last_case0)
    grid_attns.append(grid_attn_last_case1)
    grid_attns.append(grid_attn_last_case2)
    grid_attns.append(grid_attn_last_case3)

    return grid_attns


def tensorboard_label(labels=None):
    ## labels
    labels_cmap = encode_cmap(np.squeeze(labels))
    # labels_cmap = torch.from_numpy(labels_cmap).unsqueeze(0).permute([0, 3, 1, 2])
    labels_cmap = torch.from_numpy(labels_cmap).permute([0, 3, 1, 2])
    grid_labels = torchvision.utils.make_grid(tensor=labels_cmap, nrow=2)

    return grid_labels


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

import warnings
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def data_aug_rotation(origin_img, fg_img, fg_seg):
    ##input:  origin_img tensor(3,h,w)cuda, fg_img numpy(3,h,w), fg_seg numpy(h,w)
    ##output: tensor tensor(3,h,w)cuda
    bg_img = origin_img.clone()#只是复制值 不共享 且自动放在相同的cuda上 3 h w
    ########################旋转########################
    degree = random.randint(-90, 90)
    fg_img = np.transpose(fg_img, (1, 2, 0))#h w 3

    fg_img = ndimage.rotate(fg_img, degree)
    fg_seg = ndimage.rotate(fg_seg, degree)

    fg_height = fg_img.shape[0]
    fg_width = fg_img.shape[1]
    fg_aspect_ratio = fg_width / fg_height
    if bg_img.shape[1] * 0.7 < fg_height:
        fg_height = bg_img.shape[1] * 0.7
        fg_width = fg_height * fg_aspect_ratio
    if bg_img.shape[2] * 0.7 < fg_width:
        fg_width = bg_img.shape[2] * 0.7
        fg_height = fg_width / fg_aspect_ratio
    resize_shape = (int(fg_height), int(fg_width))
    fg_img = torch.tensor(fg_img).permute(2,0,1).cuda()#3 h w
    fg_img = resize(fg_img.unsqueeze(0), size=resize_shape, mode='bilinear', align_corners=False,
                 warning=False).squeeze(0)
    fg_seg = np.array(Image.fromarray(fg_seg).resize(resize_shape)).transpose(1,0)

    mask = fg_seg != 0
    row = random.randint(0, bg_img.shape[1] - fg_img.shape[1])
    col = random.randint(0, bg_img.shape[2] - fg_img.shape[2])

    bg_img_modify = bg_img[:, row:row + resize_shape[0], col:col + resize_shape[1]]
    bg_img_modify[:, mask] = fg_img[:, mask]#由于python中的等号是赋予的引用 所以相当于改变了bg_img
    aug_img = bg_img

    return aug_img

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def random_crop(images, cropsize, fills):
    if isinstance(images[0], Image.Image):
        imgsize = images[0].size[::-1]
    else:
        imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, fills):

        if isinstance(img, Image.Image):
            img = img.crop((box[6], box[4], box[7], box[5]))
            cont = Image.new(img.mode, (cropsize, cropsize))
            cont.paste(img, (box[2], box[0]))
            new_images.append(cont)

        else:
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            new_images.append(cont)

    return new_images

class RescaleNearest():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, npimg):
        import cv2
        return cv2.resize(npimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)