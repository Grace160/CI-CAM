import torch.nn.init as init
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SCG
from .segformer_head import SegFormerHead
from . import mix_transformer
import numpy as np
from utils.camutils import cam_to_label
class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MLP(nn.Module):
        """
        Linear Embedding
        """

        def __init__(self, input_dim=2048, embed_dim=768):
            super().__init__()
            self.proj = nn.Linear(input_dim, embed_dim)

        def forward(self, x):
            x = x.flatten(2).transpose(1, 2)
            x = self.proj(x)
            return x

class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None,queue_len=300,momentum=0.9, ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride


        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims
        self.momentum = momentum
        self.memory_items = nn.Parameter(torch.Tensor(num_classes,768))
        init.kaiming_uniform_(self.memory_items, a=math.sqrt(5))  # Kaiming initialization
        self.attention4 = SELayer(self.in_channels[3])
        self.attention3 = SELayer(self.in_channels[2])
        self.attention2 = SELayer(self.in_channels[1])
        self.attention1 = SELayer(self.in_channels[0])
        self.linear_c42 = MLP(input_dim=self.in_channels[3], embed_dim=256)
        self.linear_c32 = MLP(input_dim=self.in_channels[2], embed_dim=256)
        self.linear_c22 = MLP(input_dim=self.in_channels[1], embed_dim=128)
        self.linear_c12 = MLP(input_dim=self.in_channels[0], embed_dim=128)
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        for param in list(self.attention4.parameters()):
            param_groups[3].append(param)
        for param in list(self.attention3.parameters()):
            param_groups[3].append(param)
        for param in list(self.attention2.parameters()):
            param_groups[3].append(param)
        for param in list(self.attention1.parameters()):
            param_groups[3].append(param)

        return param_groups

    def get_pixel_aff(self,n, h, w, images, dl_aff, beta):  # beta: n21,nhw     cross #(n+1) * c，nhw
        nhw = n * h * w
        d, _ = beta.size()#cross中 d=(n+1) * c
        beta = beta.reshape(d, n, h, w).permute(1, 0, 2, 3)  # n d h w
        for j in range(n):
            _images = F.interpolate(images[j].unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
            beta[j] = dl_aff(_images.clone(), beta[j].unsqueeze(0).clone())
        # print("beta.size()",beta.size())#torch.Size([2, 42, 128, 128])  现在是torch.Size([2, 42, 32, 32])
        beta = beta.permute(1, 0, 2, 3).reshape(d, nhw)

        return beta

    def get_dl_logit_aff_cross(self,hie_fea,pseudo_label,dl_aff,inputs_denorm):
        n,dim,h,w=hie_fea.size()
        seeds = torch.nn.functional.one_hot(pseudo_label, num_classes=21).permute(0,3,1,2)#生成每个类别包括 2 21 128 128
        crop_feature = seeds.unsqueeze(2) * hie_fea.unsqueeze(1)#seed本身是n 21 h w->n 21 1 h w    hie_fea:n,dim,h,w->n 1 dim h w    crop_feature:n 21 dim h w
        prototypes = F.adaptive_avg_pool2d(crop_feature.reshape(-1, dim, h, w), (1, 1)).view(n, 21, dim)
        _,c,_=prototypes.size()

        # Compute similarity between features and memory_items
        features_norm = F.normalize(prototypes.view(-1, dim), p=2, dim=1)  # shape: (n*c, dim)
        memory_items_norm = F.normalize(self.memory_items, p=2, dim=1)  # shape: (num_items, dim)
        similarity = torch.matmul(features_norm, memory_items_norm.t())  # shape: (n*c, num_items)
        # Update memory items
        best_memory_items_indices = similarity.argmax(dim=1)  # shape: (n*c,)
        non_zero_channels = prototypes.view(-1, dim).sum(dim=1) != 0  # shape: (n*c,)
        self.memory_items.data[best_memory_items_indices[non_zero_channels]] = self.momentum * self.memory_items.data[best_memory_items_indices[non_zero_channels]] + (1 - self.momentum) * prototypes.view(-1, dim)[non_zero_channels]
        # Concatenate features and memory_items
        local_global_prototypes = torch.cat([self.memory_items.data.unsqueeze(0), prototypes], dim=0)  # shape: (n+1, c, dim)

        # prototypes=F.normalize(prototypes,dim=2)
        hie_fea=F.normalize(hie_fea, dim=1)# n dim h w
        X = local_global_prototypes.permute(2,0,1).view(dim,-1)#dim （n+1)21
        hie_fea = hie_fea.permute(0,2,3,1).reshape(-1,dim)  #nhw, dim
        I = torch.eye((n+1) * c).cuda()
        lam = 0.0045
        P = X.t().mm(X) +lam * I  # FT*F + lambda*I  (n+1) * c，(n+1) * c
        P = torch.inverse(P).mm(X.t())  #(n+1) * c，dim
        beta = P.mm(hie_fea.t())#(n+1) * c，nhw
        #跨像素上下文
        beta = self.get_pixel_aff(n, h, w, inputs_denorm, dl_aff, beta.clone())
        beta = beta.view(n+1,c, n*h*w).permute(1,0,2)  ## C*(n+1)
        code_x = torch.zeros((c, dim, n*h*w)).cuda()
        X = X.t().view(n+1, c, dim).permute(1,0,2)
        for cls in range(c):
            code_x[cls] = X[cls].t().mm(beta[cls])
        err = hie_fea.t() - code_x #c, dim, n*h*w   # 21 768 nhw
        dist = torch.sum(err ** 2, dim=1)  # C*n    self.num_cls, n*h*w   21 nhw
        Q_logits = dist.t()#nhw 21
        Q_logits = -torch.log(Q_logits)
        Q_logits = torch.softmax(Q_logits / 0.1, dim=1)#nb, n_classes
        Q_logits = Q_logits.reshape(n,h,w,c).permute(0,3,1,2)

        return Q_logits

    # def get_dl_logit(self,hie_fea,pseudo_label):
    #     n,dim,h,w=hie_fea.size()
    #     seeds = torch.nn.functional.one_hot(pseudo_label, num_classes=21).permute(0,3,1,2)#生成每个类别包括 2 21 128 128
    #     crop_feature = seeds.unsqueeze(2) * hie_fea.unsqueeze(1)#seed本身是n 21 h w->n 21 1 h w    hie_fea:n,dim,h,w->n 1 dim h w    crop_feature:n 21 dim h w
    #     prototypes = F.adaptive_avg_pool2d(crop_feature.reshape(-1, dim, h, w), (1, 1)).view(n, 21, dim)
    #     _,c,_=prototypes.size()
    #
    #     # prototypes=F.normalize(prototypes,dim=2)
    #     hie_fea=F.normalize(hie_fea, dim=1)# n dim h w
    #     X = prototypes.permute(2,0,1).view(dim,-1)#dim n21
    #     hie_fea = hie_fea.permute(0,2,3,1).reshape(-1,dim)
    #     I = torch.eye(n * c).cuda()
    #     lam = 0.0045
    #     P = X.t().mm(X) +lam * I  # FT*F + lambda*I
    #     P = torch.inverse(P).mm(X.t())
    #     beta = P.mm(hie_fea.t()).view(n,c, n*h*w).permute(1,0,2)  ## C*n
    #     code_x = torch.zeros((c, dim, n*h*w)).cuda()
    #     X = X.t().view(n, c, dim).permute(1,0,2)
    #     for cls in range(c):
    #         code_x[cls] = X[cls].t().mm(beta[cls])
    #     err = hie_fea.t() - code_x #c, dim, n*h*w   # 21 768 nhw
    #     dist = torch.sum(err ** 2, dim=1)  # C*n    self.num_cls, n*h*w   21 nhw
    #     Q_logits = dist.t()#nhw 21
    #
    #     Q_logits = -torch.log(Q_logits)
    #     Q_logits = torch.softmax(Q_logits / 0.1, dim=1)#nb, n_classes   还是上采样后在softmax
    #     Q_logits = Q_logits.reshape(n,h,w,c).permute(0,3,1,2)
    #
    #     return Q_logits

    def fusion_feature(self,x):
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape
        _c42 = self.linear_c42(c4.detach()).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c42 = F.interpolate(_c42 / (torch.norm(_c42, dim=1, keepdim=True) + 1e-5), size=c3.size()[2:],mode='bilinear',align_corners=False)

        _c32 = self.linear_c32(c3.detach()).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c32 = F.interpolate(_c32 / (torch.norm(_c32, dim=1, keepdim=True) + 1e-5), size=c3.size()[2:],mode='bilinear',align_corners=False)

        _c22 = self.linear_c22(c2.detach()).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c22 = F.interpolate(_c22 / (torch.norm(_c22, dim=1, keepdim=True) + 1e-5), size=c3.size()[2:],mode='bilinear',align_corners=False)

        _c12 = self.linear_c12(c1.detach()).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c12 = F.interpolate(_c12 / (torch.norm(_c12, dim=1, keepdim=True) + 1e-5), size=c3.size()[2:],mode='bilinear',align_corners=False)

        x12 = torch.cat([_c42, _c32, _c22, _c12], dim=1)

        return x12

    def forward(self, x, spx=None, cam_only=False, seg_detach=True, class_label=None, iter=None, pseudo_label=None,dlcam_only=False,cfg=None,dl_aff=None,inputs_denorm=None):
        _, _, nh,nw = x.size()
        _x, _attns = self.encoder(x)
        # _x1, _x2, _x3, _x4 = _x
        _x1_ori, _x2_ori, _x3_ori, _x4_ori = _x
        _x4 = self.attention4(_x4_ori)+_x4_ori

        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4
        if dlcam_only:
            #层级特征
            feature = self.fusion_feature(_x)
            hie_feature = feature.detach()
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            b,_,h,w = cam_s4.size()
            _cam = F.relu(torch.max(cam_s4[:int(b/2), ...], cam_s4[int(b/2):, ...].flip(-1)))#  1size是128
            _cam = _cam + F.adaptive_max_pool2d(-_cam, (1, 1))#最小值变为0
            _cam /= F.adaptive_max_pool2d(_cam, (1, 1)) + 1e-5#放大到0-1之间
            pseudo_label_raw = cam_to_label(_cam, cls_label=class_label, img_box=None, ignore_mid=False,
                                                   cfg=cfg)#img_box
            hie_feature = hie_feature[:int(b/2), ...]#+feature[int(b/2):, ...]
            # Q_logits = self.get_dl_logit(hie_feature, pseudo_label_raw)
            # return Q_logits
            #基于跨图上下文的类激活图
            Q_logits_aff = self.get_dl_logit_aff_cross(hie_feature, pseudo_label_raw,dl_aff,inputs_denorm)  # feature:[2, 256, 128, 128]  pseudo_label: 2, 128, 128]
            return Q_logits_aff
        #通道交互
        _x3 = self.attention3(_x3_ori) + _x3_ori
        _x2 = self.attention2(_x2_ori) + _x2_ori
        _x1 = self.attention1(_x1_ori) + _x1_ori

        _x = [_x1, _x2, _x3, _x4]

        seg = self.decoder(_x)
        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        seg = F.interpolate(seg, size=x.shape[2:], mode='bilinear', align_corners=False)

        return cls_x4, seg


if __name__ == "__main__":
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeTr('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    wetr._param_groups()
    dummy_input = torch.rand(2, 3, 512, 512)
    wetr(dummy_input)