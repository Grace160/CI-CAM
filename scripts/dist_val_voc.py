import argparse
import datetime
import logging
import os
import random
import sys
import warnings
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from datasets import transforms
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import pydensecrf.utils as utils
import pydensecrf.densecrf as dcrf
from datasets import voc
import imageio
from utils import evaluate, imutils
from PIL import Image
from utils.camutils import (multi_scale_cam, refine_cams_with_bkg_v2,refine_cams_with_bkg_val)
from wetr.model_attn_aff import WeTr
from wetr.PAR import PAR


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")

parser.add_argument("--high_thre", default=0.55, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.35, type=float, help="low_bkg_score")

parser.add_argument('--backend', default='nccl')
parser.add_argument('--checkpoint', help='checkpoint file')


def setup_logger(filename='test.log'):
    ## setup logger
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

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

def whole_inference(inputs, model, cfg, h, w, rescale=True):
    _, preds = model(inputs, )
    if rescale:
        preds = resize(
            preds,
            size=(h, w),
            mode='bilinear',
            align_corners=False,
            warning=False)
    return preds

def slide_inference(inputs, model, cfg, h, w, rescale=True):
    h_stride, w_stride = (480, 480)
    h_crop, w_crop = (512, 512)
    batch_size, _, h_img, w_img = inputs.size()
    num_classes = cfg.dataset.num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = inputs.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
    # print(h_grids)
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = inputs[:, :, y1:y2, x1:x2]
            crop_img = torch.cat([crop_img, crop_img.flip(-1)], dim=0)
            _, crop_seg_logit = model(crop_img, )
            crop_seg_logit = (crop_seg_logit[0, ...] + crop_seg_logit[1, ...].flip(-1)) / 2
            crop_seg_logit = crop_seg_logit.unsqueeze(0)
            preds += F.pad(crop_seg_logit,
                           (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    if torch.onnx.is_in_onnx_export():
        # cast count_mat to constant while exporting to ONNX
        count_mat = torch.from_numpy(
            count_mat.cpu().detach().numpy()).to(device=inputs.device)
    preds = preds / count_mat
    if rescale:
        preds = resize(
            preds,
            size=(h, w),
            mode='bilinear',
            align_corners=False,
            warning=False)
    return preds

# def DenseCRF( image, probmap, iter_max=10, pos_w=3, pos_xy_std=1, bi_w=4, bi_xy_std=67, bi_rgb_std=3):
def DenseCRF(image, probmap, iter_max=10, pos_w=3, pos_xy_std=3, bi_w=4, bi_xy_std=64, bi_rgb_std=5):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        dc = dcrf.DenseCRF2D(W, H, C)
        dc.setUnaryEnergy(U)
        dc.addPairwiseGaussian(sxy=pos_xy_std, compat=pos_w)
        dc.addPairwiseBilateral(
            sxy=bi_xy_std, srgb=bi_rgb_std, rgbim=image, compat=bi_w
        )

        Q = dc.inference(iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    cfg.cam.high_thre = args.high_thre
    cfg.cam.low_thre = args.low_thre

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    if args.local_rank == 0:
        setup_logger(filename=os.path.join(cfg.work_dir.dir, 'val_voc_'+ timestamp + '.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)


    num_workers = 2

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    wetr = WeTr(backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True,
                pooling=args.pooling, )
    # logging.info('\nNetwork config: \n%s' % (wetr))
    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24])
    par.to(device)
    wetr.to(device)
    model = DistributedDataParallel(wetr, device_ids=[args.local_rank], find_unused_parameters=True)
    temp_state_dict = torch.load(args.checkpoint)
    model.load_state_dict(temp_state_dict, strict=False)
    model.eval()
    # model = torch.load(args.checkpoint)

    if args.local_rank == 0:
        logging.info('Validating...')


    preds, gts, pseudo_labels = [], [], []
    scales = [1.0, 0.5, 0.75, 1.25, 1.5, 1.75]
    # scales = [1.0, 0.5, 0.75]

    # scales = [1.0]
    with torch.no_grad():

        for _, data in tqdm(enumerate(val_loader),
                            total=len(val_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs = np.asarray(imageio.imread(os.path.join(cfg.dataset.root_dir,'JPEGImages',name[0] + '.jpg')))
            inputs = transforms.normalize_img(inputs)
            inputs = np.transpose(inputs, (2, 0, 1))
            inputs = torch.from_numpy(inputs).unsqueeze(0).cuda()
            _,_,h,w = inputs.size()
            inputs_denorm = imutils.denormalize_img2(inputs.clone())
            _cams = multi_scale_cam(model, inputs, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            refined_pseudo_label = refine_cams_with_bkg_val(par, inputs_denorm, cams=resized_cam, cls_labels=cls_label,
                                                           cfg=cfg)
            n_refined_pseudo_label, _, _ = refined_pseudo_label.size()
            # for i in range(n_refined_pseudo_label):
            #     refined_pseudo_label[i, ...] = torch.from_numpy(
            #         fill_bg_with_fg(refined_pseudo_label[i, ...].detach().cpu().numpy().astype(np.int32))).cuda()

            seg_logit = slide_inference(inputs, model,cfg,h,w)#n,21,512,512
            # print(seg_logit.shape)
            seg_logit = F.softmax(seg_logit, dim=1)
            # print(seg_logit.size())
            # print(seg_logit)
            for i in range(1, len(scales)):
                img = resize(inputs,size=(int(h*scales[i]), int(w*scales[i])), mode='bilinear',align_corners=False, warning=False)
                cur_seg_logit = slide_inference(img, model,cfg,h,w)
                cur_seg_logit = F.softmax(cur_seg_logit, dim=1)
                seg_logit += cur_seg_logit
            seg_logit /= len(scales)
            ##crf
            if cfg.val.crf:
                ori_img = np.array(Image.open(os.path.join(cfg.dataset.root_dir, 'JPEGImages', name[0] + '.jpg')).convert("RGB"))
                seg_logit_ori = seg_logit[0].cpu().numpy()
                seg_logit = DenseCRF(ori_img, seg_logit_ori)
                seg_logit = seg_logit.reshape((1, seg_logit.shape[0], seg_logit.shape[1], seg_logit.shape[2]))
                seg_logit = torch.FloatTensor(seg_logit)

            seg_pred = seg_logit.argmax(dim=1)
            # print(seg_pred)
            seg_pred = seg_pred.cpu().numpy().astype(np.int16)

            # unravel batch dim
            preds += list(seg_pred)
            gts += list(labels.cpu().numpy().astype(np.int16))
            pseudo_labels += list(refined_pseudo_label.cpu().numpy().astype(np.int16))
    # print(preds)
    # print(np.shape(gts))
    # print(np.shape(preds))
    seg_score = evaluate.scores(gts, preds)
    pseudo_labels_score = evaluate.scores(gts, pseudo_labels)

    if args.local_rank == 0:
        logging.info("segs score:")
        logging.info(seg_score)
        logging.info("pseudo labels score:")
        logging.info(pseudo_labels_score)


