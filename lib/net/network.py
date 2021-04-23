import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import (res32_cifar, res50, res10)
from modules import GAP, FCNorm, Identity, LWS, cRT
import copy
import numpy as np
import cv2
import os

class Network(nn.Module):
    def __init__(self, cfg, mode="train", num_classes=1000):
        super(Network, self).__init__()
        pretrain = (
            True
            if mode == "train"
            and cfg.RESUME_MODEL == ""
            and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.num_classes = num_classes
        self.cfg = cfg

        self.backbone = eval(self.cfg.BACKBONE.TYPE)(
            self.cfg,
            pretrain=pretrain,
            pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
            last_layer_stride=2,
        )
        self.module = self._get_module()
        self.classifier = self._get_classifer()

        if cfg.NETWORK.PRETRAINED and os.path.isfile(cfg.NETWORK.PRETRAINED_MODEL):
            try:
                self.load_model(cfg.NETWORK.PRETRAINED_MODEL)
            except:
                raise ValueError('network pretrained model error')

    def forward(self, x, **kwargs):
        if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
            return self.extract_feature(x, **kwargs)
        elif "classifier_flag" in kwargs:
            return self.classifier(x)
        elif 'feature_maps_flag' in kwargs:
            return self.extract_feature_maps(x)
        elif 'layer' in kwargs and 'index' in kwargs:
            if kwargs['layer'] in ['layer1', 'layer2', 'layer3']:
                x = self.backbone.forward(x, index=kwargs['index'], layer=kwargs['layer'], coef=kwargs['coef'])
            else:
                x = self.backbone(x)
            x = self.module(x)
            if kwargs['layer'] == 'pool':
                x = kwargs['coef']*x+(1-kwargs['coef'])*x[kwargs['index']]
            x = x.view(x.shape[0], -1)
            x = self.classifier(x)
            if kwargs['layer'] == 'fc':
                x = kwargs['coef']*x + (1-kwargs['coef'])*x[kwargs['index']]
            return x

        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def get_backbone_layer_info(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            layers = 3
            blocks_info = [5, 5, 5]
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            layers = 4
            blocks_info = [1, 1, 1, 1]
        else:
            layers = 4
            blocks_info = [3, 4, 6, 3]
        return layers, blocks_info

    def extract_feature(self, x, **kwargs):
        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        return x

    def extract_feature_maps(self, x):
        x = self.backbone(x)
        return x

    def freeze_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False


    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone model has been loaded...")


    def load_model(self, model_path, tau_norm=False, tau=1):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if  k == 'classifier.weight':
                if tau_norm:
                    print('*-*'*30)
                    print('Using tau-normalization')
                    print('*-*'*30)
                    v = v / torch.pow(torch.norm(v, 2, 1, keepdim=True), tau)
                # if self.cfg.CLASSIFIER.TYPE == "cRT":
                #
                # elif self.cfg.CLASSIFIER.TYPE == "LWS"
            new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("All model has been loaded...")


    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            num_features = 512
        else:
            num_features = 2048
        return num_features


    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module= Identity()
        else:
            raise NotImplementedError

        return module


    def _get_classifer(self):
        bias_flag = self.cfg.CLASSIFIER.BIAS

        num_features = self.get_feature_length()
        if self.cfg.CLASSIFIER.TYPE == "FCNorm":
            classifier = FCNorm(num_features, self.num_classes)
        elif self.cfg.CLASSIFIER.TYPE == "FC":
            classifier = nn.Linear(num_features, self.num_classes, bias=bias_flag)
        elif self.cfg.CLASSIFIER.TYPE == "cRT":
            classifier = cRT(num_features, self.num_classes)
        elif self.cfg.CLASSIFIER.TYPE == "LWS":
            classifier = LWS(num_features, self.num_classes)
        else:
            raise NotImplementedError

        return classifier


    def cam_params_reset(self):
        self.classifier_weights = np.squeeze(list(self.classifier.parameters())[0].detach().cpu().numpy())

    def get_CAM_with_groundtruth(self, image_idxs, dataset, label_list, size):
        ret_cam = []
        size_upsample = size
        for i in range(len(image_idxs)):
            idx = image_idxs[i]
            label = label_list[idx]
            self.eval()
            with torch.no_grad():
                img = dataset._get_trans_image(idx)
                feature_conv = self.forward(img.to('cuda'), feature_maps_flag=True).detach().cpu().numpy()
            b, c, h, w = feature_conv.shape
            assert b == 1
            feature_conv = feature_conv.reshape(c, h*w)
            cam = self.classifier_weights[label].dot(feature_conv)
            del img
            del feature_conv
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255*cam_img)
            ret_cam.append(cv2.resize(cam_img, size_upsample))
        return ret_cam
