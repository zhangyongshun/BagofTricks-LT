import torch
import torch.nn as nn
import torch.nn.functional as F

# for LDAM Loss
class FCNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.linear(F.normalize(x), F.normalize(self.weight))
        return out


class LWS(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super(LWS, self).__init__()
        self.fc = nn.Linear(num_features, num_classes, bias=bias)
        self.scales = nn.Parameter(torch.ones(num_classes))
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        x *= self.scales
        return x
