import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from ..utils import get_one_hot
from ..loss_base import CrossEntropy



class ClassBalanceCE(CrossEntropy):
    r"""
    Reference:
    Cui et al., Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.

        Equation: Loss(x, c) = \frac{1-\beta}{1-\beta^{n_c}} * CrossEntropy(x, c)

    Class-balanced loss considers the real volumes, named effective numbers, of each class, \
    rather than nominal numeber of images provided by original datasets.

    Args:
        beta(float, double) : hyper-parameter for class balanced loss to control the cost-sensitive weights.
    """
    def __init__(self, para_dict= None):
        super(ClassBalanceCE, self).__init__(para_dict)
        self.beta = self.para_dict['cfg'].LOSS.ClassBalanceCE.BETA
        self.class_balanced_weight = np.array([(1-self.beta)/(1- self.beta ** N) for N in self.num_class_list])
        self.class_balanced_weight = torch.FloatTensor(self.class_balanced_weight / np.sum(self.class_balanced_weight) * self.num_classes).to(self.device)

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = self.class_balanced_weight
        else:
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.class_balanced_weight


class ClassBalanceFocal(CrossEntropy):
    r"""
    Reference:
    Li et al., Focal Loss for Dense Object Detection. ICCV 2017.
    Cui et al., Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.

        Equation: Loss(x, class) = \frac{1-\beta}{1-\beta^{n_c}} * FocalLoss(x, c)

    Class-balanced loss considers the real volumes, named effective numbers, of each class, \
    rather than nominal numeber of images provided by original datasets.

    Args:
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                               putting more focus on hard, misclassiﬁed examples
        beta(float, double): hyper-parameter for class balanced loss to control the cost-sensitive weights.
    """
    def __init__(self, para_dict=None):
        super(ClassBalanceFocal, self).__init__(para_dict)
        self.beta = self.para_dict['cfg'].LOSS.ClassBalanceFocal.BETA
        self.gamma = self.para_dict['cfg'].LOSS.ClassBalanceFocal.GAMMA
        self.class_balanced_weight = np.array([(1-self.beta)/(1- self.beta ** N) for N in self.num_class_list])
        self.class_balanced_weight = torch.FloatTensor(self.class_balanced_weight / np.sum(self.class_balanced_weight) * self.num_classes).to(self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        weight = (self.weight_list[targets]).to(targets.device)
        label = get_one_hot(targets, self.num_classes)
        p = self.sigmoid(inputs)
        focal_weights = torch.pow((1-p)*label + p * (1-label), self.gamma)
        loss = F.binary_cross_entropy_with_logits(inputs, label, reduction = 'none') * focal_weights
        loss = (loss * weight.view(-1, 1)).sum() / inputs.shape[0]
        return loss

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = self.class_balanced_weight
        else:
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.class_balanced_weight
            else:
                self.weight_list = torch.ones(self.class_balanced_weight.shape).to(self.device)
