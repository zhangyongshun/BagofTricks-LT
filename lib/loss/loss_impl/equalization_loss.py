import torch

from ..loss_base import CrossEntropy
from ..utils import get_one_hot
from torch.nn import functional as F

class SEQL(CrossEntropy):
    r"""
    Reference:
    Tan et al., Equalization Loss for Long-Tailed Object Recognition. CVPR 2020.

    SEQL means softmax equalization loss

    Equation:
        Eq.1: loss = - sum_j (y_j * log p_j),
        Eq.2: p_j = exp(z_j) / sum_k(w_k * exp(z_k),
        Eq.3: w_k = 1 - beta * T_lambda(n_k) * (1 - y_k),
        where y: one-hot label
              beta: a random variable with a probability of gamma (hyper-parameter) to be 1 and 1-gamma to be 0,
              n_k: the number of images in class k
              T_lambda: a threshold function which outputs 1 when n_k < lambda (hyper-parameter), otherwise, 0

    Args:
        gamma (float or double): a probability in beta, which is used to maintain the gradient of negative samples
        lambda (float or double): a threshold in T_lambda function

        For detailed ablation study of SEQL, see original paper's Table 7.
    """
    def __init__(self, para_dict=None):
        super(SEQL, self).__init__(para_dict)
        num_class_list = torch.FloatTensor(self.num_class_list)
        self.t_lambda = ((num_class_list/num_class_list.sum()) < self.para_dict['cfg'].LOSS.SEQL.LAMBDA).to(self.device)
        self.gamma = self.para_dict['cfg'].LOSS.SEQL.GAMMA

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        onehot_label = get_one_hot(targets, self.num_classes)
        #Eq.3
        beta = (torch.rand(onehot_label.size()) < self.gamma).to(self.device)
        w = 1 - beta * self.t_lambda * (1 - onehot_label)
        #Eq.2
        p = torch.exp(inputs) / ((w*torch.exp(inputs)).sum(axis=1, keepdims=True))
        #Eq.1
        loss = F.nll_loss(torch.log(p), targets)
        return loss
