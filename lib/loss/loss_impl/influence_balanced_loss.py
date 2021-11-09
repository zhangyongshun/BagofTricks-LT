import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from ..loss_base import CrossEntropy

class InfluenceBalancedLoss(CrossEntropy):
    r"""
    References:
    Seulki et al., Influence-Balanced Loss for Imbalanced Visual Classification, ICCV 2021.
    """

    def __init__(self, para_dict=None):
        super(InfluenceBalancedLoss, self).__init__(para_dict)

        ib_weight = 1.0 / np.array(self.num_class_list)
        ib_weight = ib_weight / np.sum(ib_weight) * self.num_classes
        self.ib_weight = torch.FloatTensor(ib_weight).to(self.device)
        self.use_vanilla_ce = False
        self.alpha = self.para_dict['cfg'].LOSS.InfluenceBalancedLoss.ALPHA

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """

        if self.use_vanilla_ce:
            return super().forward(inputs, targets)

        assert 'feature' in kwargs, 'Feature is required in InfluenceBalancedLoss. \
                You should feed the features from lib/core/combiner.py, \
                and can see \
                    https://github.com/pseulki/IB-Loss/blob/751cd39e43dee4f6cb9fff2d3fb24acd633a22c3/models/resnet_cifar.py#L130 \
                for more details'

        feature = torch.sum(torch.abs(kwargs['feature']), 1).reshape(-1, 1)
        grads = torch.sum(torch.abs(F.softmax(inputs, dim=1) - F.one_hot(targets, self.num_classes)), 1)
        ib = grads * feature.reshape(-1)
        ib = self.alpha / (ib + 1e-3)
        ib_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight_list)*ib
        return ib_loss.mean()


    def update(self, epoch):
        """
        Args:
            epoch: int
        """
        if not self.drw:
            self.weight_list = self.ib_weight
        else:
            self.weight_list = torch.ones(self.ib_weight.shape).to(self.device)
            start = (epoch-1) // self.drw_start_epoch
            self.use_vanilla_ce = True
            if start:
                self.use_vanilla_ce = False
                self.weight_list = self.ib_weight