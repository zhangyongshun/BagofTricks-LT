import torch
from torch.nn import functional as F
import numpy as np
from ..loss_base import CrossEntropy

class CrossEntropyLabelAwareSmooth(CrossEntropy):
    r"""Cross entropy loss with label-aware smoothing regularizer.

    Reference:
        Zhong et al. Improving Calibration for Long-Tailed Recognition. CVPR 2021. https://arxiv.org/abs/2104.00466

    For more details of label-aware smoothing, you can see Section 3.2 in the above paper.

    Args:
        shape (str): the manner of how to get the params of label-aware smoothing.
        smooth_head (float): the largest  label smoothing factor
        smooth_tail (float): the smallest label smoothing factor
    """
    def __init__(self, para_dict=None):
        super(CrossEntropyLabelAwareSmooth, self).__init__(para_dict)

        smooth_head = self.para_dict['cfg'].LOSS.CrossEntropyLabelAwareSmooth.SMOOTH_HEAD
        smooth_tail = self.para_dict['cfg'].LOSS.CrossEntropyLabelAwareSmooth.SMOOTH_TAIL
        shape = self.para_dict['cfg'].LOSS.CrossEntropyLabelAwareSmooth.SHAPE

        n_1 = max(self.num_class_list)
        n_K = min(self.num_class_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(self.num_class_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(self.num_class_list) - n_K) / (n_1 - n_K)
        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(self.num_class_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        else:
            raise AttributeError

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, inputs, targets, **kwargs):
        smoothing = self.smooth[targets]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
