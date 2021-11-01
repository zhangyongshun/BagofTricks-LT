import torch
import torch.nn as nn
from ..loss_base import CrossEntropy

class CrossEntropyLabelSmooth(CrossEntropy):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

        Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight of label smooth.
    """
    def __init__(self, para_dict=None):
        super(CrossEntropyLabelSmooth, self).__init__(para_dict)
        self.epsilon = self.para_dict['cfg'].LOSS.CrossEntropyLabelSmooth.EPSILON #hyper-parameter in label smooth
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
