import torch
import numpy as np

from ..loss_base import CrossEntropy

class CostSensitiveCE(CrossEntropy):
    r"""
        Equation: Loss(z, c) = - (\frac{N_min}{N_c})^\gamma * CrossEntropy(z, c),
        where gamma is a hyper-parameter to control the weights,
            N_min is the number of images in the smallest class,
            and N_c is the number of images in the class c.

    The representative re-weighting methods, which assigns class-dependent weights to the loss function

    Args:
        gamma (float or double): to control the loss weights: (N_min/N_i)^gamma
    """
    def __init__(self, para_dict=None):
        super(CostSensitiveCE, self).__init__(para_dict)
        gamma = self.para_dict['cfg'].LOSS.CostSensitiveCE.GAMMA
        self.csce_weight = torch.FloatTensor(np.array([(min(self.num_class_list) / N)**gamma for N in self.num_class_list])).to(self.device)

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = self.csce_weight
        else:
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.csce_weight
