import torch
from torch.nn import functional as F

from ..loss_base import CrossEntropy

class CDT(CrossEntropy):
    r"""
    References:
    Class-Dependent Temperatures (CDT) Loss, Ye et al., Identifying and Compensating for Feature Deviation in Imbalanced Deep Learning, arXiv 2020.

    Equation:  Loss(x, c) = - log(\frac{exp(x_c / a_c)}{sum_i(exp(x_i / a_i))}), and a_j = (N_max/n_j)^\gamma,
                where gamma is a hyper-parameter, N_max is the number of images in the largest class,
                and n_j is the number of image in class j.
    Args:
        gamma (float or double): to control the punishment to feature deviation.  For CIFAR-10, γ ∈ [0.0, 0.4]. For CIFAR-100
        and Tiny-ImageNet, γ ∈ [0.0, 0.2]. For iNaturalist, γ ∈ [0.0, 0.1]. We then select γ from several
        uniformly placed grid values in the range
    """
    def __init__(self, para_dict=None):
        super(CDT, self).__init__(para_dict)
        self.gamma = self.para_dict['cfg'].LOSS.CDT.GAMMA
        self.cdt_weight = torch.FloatTensor([(max(self.num_class_list) / i) ** self.gamma for i in self.num_class_list]).to(self.device)

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        inputs = inputs / self.weight_list
        loss = F.cross_entropy(inputs, targets)
        return loss

    def update(self, epoch):
        """
        Args:
            epoch: int
        """
        if not self.drw:
            self.weight_list = self.cdt_weight
        else:
            self.weight_list = torch.ones(self.cdt_weight.shape).to(self.device)
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.cdt_weight
