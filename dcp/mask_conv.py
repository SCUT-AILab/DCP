import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F

__all__ = ["MaskConv2d"]


class MaskConv2d(nn.Conv2d):
    """
    Custom convolutional layers for channel pruning.
    Here we use mask to indicate the selected/unselected channels. 1 denotes selected and 0 denotes unselected.
    We store the original weight in weight and the pruned weight in pruned_weight.
    Before channel selection, the pruned weight is initialized to zero.
    In greedy algorithm for channel selection, we initialize the pruned weight with the original weight and
    then optimized pruned weight w.r.t. the selected channels by minimizing the problem (9).
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super(MaskConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        self.register_buffer("d", torch.ones(self.weight.data.size(1)))
        self.pruned_weight = Parameter(self.weight.clone())

    def forward(self, input):
        return F.conv2d(input, self.pruned_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
