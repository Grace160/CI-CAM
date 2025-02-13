import numpy as np
from numpy.lib.utils import deprecate
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio
from . import transforms
import torchvision
from skimage.segmentation import slic,find_boundaries
from PIL import Image
import random
from scipy import misc,ndimage
from skimage.transform import resize

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list


def load_cls_label_list(name_list_dir):
    return np.load(os.path.join(name_list_dir, 'cls_labels_onehot.npy'), allow_pickle=True).item()


def data_aug_rotation(origin_img, fg_img, fg_seg):
    bg_img = np.empty_like(origin_img)
    bg_img[:, :, :] = origin_img[:, :, :]
    ########################旋转########################
    degree = random.randint(-90, 90)
    fg_img = ndimage.rotate(fg_img, degree)
    fg_seg = ndimage.rotate(fg_seg, degree)

    fg_height = fg_img.shape[0]
    fg_width = fg_img.shape[1]
    fg_aspect_ratio = fg_width / fg_height
    if bg_img.shape[0] * 0.7 < fg_height:
        fg_height = bg_img.shape[0] * 0.7
        fg_width = fg_height * fg_aspect_ratio
    if bg_img.shape[1] * 0.7 < fg_width:
        fg_width = bg_img.shape[1] * 0.7
        fg_height = fg_width / fg_aspect_ratio
    resize_shape = (int(fg_height), int(fg_width))
    fg_img = resize(fg_img, resize_shape)
    fg_seg = resize(fg_seg, resize_shape)

    mask = fg_seg != 0
    row = random.randint(0, bg_img.shape[0] - fg_img.shape[0])
    col = random.randint(0, bg_img.shape[1] - fg_img.shape[1])

    bg_img_modify = bg_img[row:row + resize_shape[0], col:col + resize_shape[1], :]
    fg_img = (fg_img * 255).astype(np.uint8)
    bg_img_modify[mask, :] = fg_img[mask, :]

    aug_img = bg_img

    return aug_img

class VOC12Dataset(Dataset):
    def __init__(
            self,
            root_dir=None,
            name_list_dir=None,
            split='train',
            stage='train',
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClassAug')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name + '.jpg')
        image = np.asarray(imageio.imread(img_name))
        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name + '.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name + '.png')
            label = np.asarray(imageio.imread(label_dir))
            # print(label.shape)#(230, 500)

        elif self.stage == "test":
            label = image[:, :, 0]

        return _img_name, image, label


class VOC12ClsDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image,label):
        img_box = None
        if self.aug:
            '''
            if self.resize_range: 
                image, label = transforms.random_resize(
                    image, label, size_range=self.resize_range)
            '''
            if self.rescale_range:
                image,label = transforms.random_scaling(
                    image,label=label,
                    scale_range=self.rescale_range)

            if self.img_fliplr:
                image,label = transforms.random_fliplr(image,label)
            image = self.color_jittor(image)
            if self.crop_size:
                image,label, img_box, mask = transforms.random_crop(
                    image,
                    label=label,
                    crop_size=self.crop_size,
                    mean_rgb=[0, 0, 0],  # [123.675, 116.28, 103.53],
                    ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        if self.aug:
            mode = np.random.randint(3)
            if mode == 1:
                image = transforms.add_gaussian_noise(image, noise_sigma=25)
            if mode == 0:
                mask2 = mask == 0
                row = np.min(mask2, axis=1)  # 获取首次出现前景位置
                col = np.min(mask2, axis=0)
                mn_row = np.argmin(row)
                mx_row = row.shape[0] - np.argmin(row[::-1]) - 1
                mn_col = np.argmin(col)
                mx_col = col.shape[0] - np.argmin(col[::-1]) - 1
                image[mn_row:mx_row + 1, mn_col:mx_col + 1] = self.color_jittor(image[mn_row:mx_row + 1, mn_col:mx_col + 1])
            if mode == 2:
                image = image

        image = transforms.normalize_img(image.astype(np.float32))
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, img_box,label

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        # label_onehot = F.one_hot(label, num_classes)

        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

        img_name, image, label = super().__getitem__(idx)

        image, img_box,label = self.__transforms(image=image,label=label)

        cls_label = self.label_list[img_name]

        if self.aug:
            return img_name, image, cls_label, img_box,label
        else:
            return img_name, image, cls_label


class VOC12SegDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            '''
            if self.resize_range: 
                image, label = transforms.random_resize(
                    image, label, size_range=self.resize_range)

            if self.rescale_range:
                image, label = transforms.random_scaling(
                    image,
                    label,
                    scale_range=self.rescale_range)
            '''
            if self.img_fliplr:
                image, label = transforms.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = transforms.random_crop(
                    image,
                    label,
                    crop_size=self.crop_size,
                    mean_rgb=[123.675, 116.28, 103.53],
                    ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        image = transforms.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)

        cls_label = self.label_list[img_name]
        # cls_label =image[:, :, 0]

        return img_name, image, label, cls_label


class VOC12ClsFgAugDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 cfg=None,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()
        self.seg_fg_dir=np.load('/amax/sungy/project/e2edl/work_dir_transformer_train/checkpoints/2024-02-28-02-56/cross_fg/seg_fg_dirs.npy', allow_pickle=True).item()
        self.sem_seg_out_aug_fg_dir='/amax/sungy/project/e2edl/work_dir_transformer_train/checkpoints/2024-02-28-02-56/cross_fg'
        # self.seg_fg_dir=np.load(os.path.join(cfg.sem_seg_out_aug_fg_dir,'seg_fg_dirs.npy'), allow_pickle=True).item()
        # self.sem_seg_out_aug_fg_dir = cfg.sem_seg_out_aug_fg_dir
        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image,label,fg_aug_img):
        img_box = None
        if self.aug:
            '''
            if self.resize_range:
                image, label = transforms.random_resize(
                    image, label, size_range=self.resize_range)
            '''
            if self.rescale_range:
                image,label,fg_aug_img = transforms.random_scaling_fgaug(
                    image,label=label,
                    scale_range=self.rescale_range,aug_img=fg_aug_img)

            if self.img_fliplr:
                image,label,fg_aug_img = transforms.random_fliplr_fgaug(image,label,aug_img=fg_aug_img)
            image = self.color_jittor(image)
            fg_aug_img = self.color_jittor(fg_aug_img)

            if self.crop_size:
                image,label, img_box, mask,fg_aug_img = transforms.random_crop_fgaug(
                    image,
                    label=label,
                    crop_size=self.crop_size,
                    mean_rgb=[0, 0, 0],  # [123.675, 116.28, 103.53],
                    ignore_index=self.ignore_index,aug_img=fg_aug_img)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        if self.aug:
            mode = np.random.randint(3)
            if mode == 1:
                image, fg_aug_img = transforms.add_gaussian_noise_fgaug(image, noise_sigma=25,aug_img=fg_aug_img)
            if mode == 0:
                mask2 = mask == 0
                row = np.min(mask2, axis=1)  # 获取首次出现前景位置
                col = np.min(mask2, axis=0)
                mn_row = np.argmin(row)
                mx_row = row.shape[0] - np.argmin(row[::-1]) - 1
                mn_col = np.argmin(col)
                mx_col = col.shape[0] - np.argmin(col[::-1]) - 1
                image[mn_row:mx_row + 1, mn_col:mx_col + 1] = self.color_jittor(image[mn_row:mx_row + 1, mn_col:mx_col + 1])
                fg_aug_img[mn_row:mx_row + 1, mn_col:mx_col + 1] = self.color_jittor(fg_aug_img[mn_row:mx_row + 1, mn_col:mx_col + 1])

            if mode == 2:
                image = image
                fg_aug_img = fg_aug_img

        image = transforms.normalize_img(image.astype(np.float32))
        fg_aug_img = transforms.normalize_img(fg_aug_img.astype(np.float32))

        ## to chw
        image = np.transpose(image, (2, 0, 1))
        fg_aug_img = np.transpose(fg_aug_img, (2, 0, 1))


        return image, img_box, label,fg_aug_img

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        # label_onehot = F.one_hot(label, num_classes)

        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

        img_name, image, label = super().__getitem__(idx)
        cls_label = self.label_list[img_name]
        label_index = np.arange(len(cls_label))
        label_index = label_index[cls_label == 0]
        fg_label = np.random.choice(label_index)
        fg_name = np.random.choice(self.seg_fg_dir[fg_label])
        fg_pack = np.load(os.path.join(self.sem_seg_out_aug_fg_dir, str(fg_label), fg_name),
                          allow_pickle=True).item()

        fg_img = fg_pack['img']
        fg_seg = fg_pack['seg']
        aug_img = data_aug_rotation(image, fg_img, fg_seg)
        aug_label_one_hot = np.zeros(len(cls_label))
        aug_label_one_hot[cls_label == 1] = 1
        aug_label_one_hot[fg_label] = 1
        # imageio.imsave('tmp.jpg', aug_img)
        image, img_box, label,fg_aug_img = self.__transforms(image=image,label=label,fg_aug_img=aug_img)


        if self.aug:
            return img_name, image, cls_label, img_box, label,fg_aug_img,aug_label_one_hot
        else:
            return img_name, image, cls_label,fg_aug_img,aug_label_one_hot

