import torch
import torch.nn as nn
from torch.nn import functional as F

from ..loss_base import CrossEntropy
from loss import *


class DiVEKLD(CrossEntropy):
    """
    Knowledge Distillation

    References:
        Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
        Yin-Yin He et al., Distilling Virtual Examples for Long-tailed Recognition. ICCV 2021

    Equation:
        loss = (1-alpha) * ce(logits_s, label) + alpha * kld(logits_s, logits_t)

    """

    def __init__(self, para_dict=None):
        super(DiVEKLD, self).__init__(para_dict)

        self.power = para_dict["cfg"].LOSS.DiVEKLD.POWER if para_dict["cfg"].LOSS.DiVEKLD.POWER_NORM else 1.0
        self.T = para_dict["cfg"].LOSS.DiVEKLD.TEMPERATURE
        self.alpha = para_dict["cfg"].LOSS.DiVEKLD.ALPHA
        self.base_loss = eval(para_dict["cfg"].LOSS.DiVEKLD.BASELOSS)(para_dict)

    def forward(self, inputs_s, inputs_t, targets, **kwargs):

        logp_s = F.log_softmax(inputs_s / self.T, dim=1)
        soft_t = (F.softmax(inputs_t / self.T, dim=1))**self.power
        soft_t /= soft_t.sum(1, keepdim=True)
        soft_t.detach_()
        kl_loss = (self.T**2)*F.kl_div(logp_s,soft_t, reduction='batchmean')
        loss = self.alpha * kl_loss + (1 - self.alpha) * self.base_loss(inputs_s, targets)

        return loss

    def update(self, epoch):
        self.base_loss.update(epoch)

