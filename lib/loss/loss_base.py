import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()
        self.para_dict = para_dict
        self.num_classes = self.para_dict["num_classes"]
        self.num_class_list = self.para_dict['num_class_list']
        self.device = self.para_dict['device']

        self.weight_list = None
        #setting about defferred re-balancing by re-weighting (DRW)
        self.drw = self.para_dict['cfg'].TRAIN.TWO_STAGE.DRW
        self.drw_start_epoch = self.para_dict['cfg'].TRAIN.TWO_STAGE.START_EPOCH #start from 1

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        loss = F.cross_entropy(inputs, targets, weight=self.weight_list)
        return loss

    def update(self, epoch):
        """
        Adopt cost-sensitive cross-entropy as the default
        Args:
            epoch: int. starting from 1.
        """
        start = (epoch-1) // self.drw_start_epoch
        if start and self.drw:
            self.weight_list = torch.FloatTensor(np.array([min(self.num_class_list) / N for N in self.num_class_list])).to(self.device)
