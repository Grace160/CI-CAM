import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from wetr import aff
from datasets import voc
from utils.losses import DenseEnergyLoss, get_energy_loss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from PIL import Image
import math
from utils import evaluate, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils import (cam_to_label, multi_scale_cam, multi_scale_dlcam, dlcam_to_label_dpa_marginmax_width,dlcam_to_label_dpa_neighbors)
from utils.optimizer import PolyWarmupAdamW
from wetr.model_attn_aff import WeTr

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

parser.add_argument("--high_thre", default=0.35, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.35, type=float, help="low_bkg_score")
parser.add_argument("--queue_len", default=500, type=int, help="queue_len")
parser.add_argument("--momentum", default=1.0, type=float, help="momentum")
parser.add_argument('--backend', default='nccl')
parser.add_argument("--margin_max", default=70, type=int, help="margin_max")
parser.add_argument("--margin_width", default=50, type=int, help="margin_width")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def validate(model=None, data_loader=None, cfg=None,dl_aff=None):
    preds, gts, cams, dl_pseudo_labels = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            _,_,h,w = inputs.size()
            cls_label_rep = F.pad(cls_label.unsqueeze(-1).unsqueeze(-1), (0, 0, 0, 0, 1, 0), 'constant', 1.0).repeat([1, 1, h, w])
            _cams = multi_scale_cam(model, inputs, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)
            cls, segs = model(inputs, class_label=cls_label, cfg=cfg)
            dl_cams = multi_scale_dlcam(model, inputs=inputs, scales=cfg.cam.scales, cls_labels=cls_label, cfg=cfg,
                                        dl_aff=dl_aff)  # [2, 20, 128, 128]
            dl_cams = dl_cams * cls_label_rep
            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            dl_pseudo_label = dl_cams.argmax(dim=1)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            dl_pseudo_labels += list(dl_pseudo_label.cpu().numpy().astype(np.int16))

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    dl_pseudo_labels_score = evaluate.scores(gts, dl_pseudo_labels)

    model.train()
    return cls_score, seg_score, cam_score, dl_pseudo_labels_score


def train(cfg):
    num_workers = 10

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              # shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              sampler=train_sampler,
                              prefetch_factor=4)

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
                pooling=args.pooling,
                queue_len=args.queue_len,
                momentum=args.momentum,)
    logging.info('\nNetwork config: \n%s' % (wetr))
    param_groups = wetr.get_param_groups()
    wetr.to(device)
    dl_aff = aff.AFF(num_iter=10, dilations=[1,2])
    dl_aff.to(device)


    if args.local_rank == 0:
        writer = SummaryWriter(cfg.work_dir.tb_logger_dir)
        dummy_input = torch.rand(1, 3, 384, 384).cuda(0)
        # writer.add_graph(wetr, dummy_input)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,  ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    wetr = DistributedDataParallel(wetr, device_ids=[args.local_rank], find_unused_parameters=True,broadcast_buffers=False)
    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    bkg_cls = torch.ones(size=(cfg.train.samples_per_gpu, 1))

    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs, cls_labels, img_box,label = next(train_loader_iter)
            # img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        except:
            train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box,label  = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)

        cls_labels = cls_labels.to(device, non_blocking=True)

        dl_cams = multi_scale_dlcam(wetr, inputs=inputs, scales=cfg.cam.scales, cls_labels=cls_labels, cfg=cfg,
                                    dl_aff=dl_aff)  # [2, 20, 128, 128]
        #动态置信度阈值
        pseudo_label = dlcam_to_label_dpa_marginmax_width(dl_cams, cls_label=cls_labels, img_box=img_box, ignore_mid=True,
                                               cfg=cfg,n_iter=n_iter,max_iters=cfg.train.max_iters,max_margin=args.margin_max,margin_width=args.margin_width)#img_box

        cls, segs = wetr(inputs, seg_detach=args.seg_detach, class_label=cls_labels, cfg=cfg)  # feature:n,256,128,128

        reg_loss = get_energy_loss(img=inputs, logit=segs, label=pseudo_label, img_box=img_box, loss_layer=loss_layer)

        seg_loss = F.cross_entropy(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)

        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * cls_loss + 0.0 * seg_loss + 0.0 * reg_loss
        else:
            loss = 1.0 * cls_loss + 0.1 * seg_loss + 0.01 * reg_loss

        avg_meter.add({'cls_loss': cls_loss.item(), 'seg_loss': seg_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs, dim=1, ).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)
            lbs = label.numpy().astype(np.int16)

            seg_mAcc_dlcam = (lbs == gts).sum() / preds.size


            if args.local_rank == 0:
                logging.info(
                    "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, pseudo_seg_loss %.4f, pseudo_seg_mAcc_mAccdlcam: %.4f" % (
                    n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('seg_loss'), seg_mAcc_dlcam))

        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "wetr_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                torch.save(wetr.state_dict(), ckpt_name)
            cls_score, seg_score, cam_score, dl_pseudo_labels_score = validate(model=wetr, data_loader=val_loader, cfg=cfg, dl_aff=dl_aff)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (cls_score))
                logging.info("cams score:")
                logging.info(cam_score)
                logging.info("segs score:")
                logging.info(seg_score)
                logging.info("dl pseudo labels score:")
                logging.info(dl_pseudo_labels_score)

    return True


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
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)

    if args.local_rank == 0:
        setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp + '.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)

    ## fix random seed
    setup_seed(42)
    train(cfg=cfg)
