import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from ..utils import get_one_hot
from ..loss_base import CrossEntropy

class FocalLoss(CrossEntropy):
    r"""
    Reference:
    Li et al., Focal Loss for Dense Object Detection. ICCV 2017.

        Equation: Loss(x, class) = - (1-sigmoid(p^t))^gamma \log(p^t)

    Focal loss tries to make neural networks to pay more attentions on difficult samples.

    Args:
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                               putting more focus on hard, misclassiﬁed examples
    """
    def __init__(self, para_dict=None):
        super(FocalLoss, self).__init__(para_dict)
        self.gamma = self.para_dict['cfg'].LOSS.FocalLoss.GAMMA #hyper-parameter
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        weight = (self.weight_list[targets]).to(targets.device) \
            if self.weight_list is not None else \
            torch.FloatTensor(torch.ones(targets.shape[0])).to(targets.device)
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
            self.weight_list = torch.FloatTensor(np.array([1 for _ in self.num_class_list])).to(self.device)
        else:
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = torch.FloatTensor(np.array([min(self.num_class_list) / N for N in self.num_class_list])).to(self.device)
            else:
                self.weight_list = torch.FloatTensor(np.array([1 for _ in self.num_class_list])).to(self.device)
