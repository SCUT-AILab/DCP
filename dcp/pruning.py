import logging

import torch.nn as nn

from dcp.mask_conv import MaskConv2d

__all__ = ['ResModelPrune', 'get_select_channels']

logger = logging.getLogger('channel_selection')


def get_select_channels(d):
    """
    Get select channels
    """

    select_channels = (d > 0).nonzero().squeeze()
    return select_channels


def get_thin_params(layer, select_channels, dim=0):
    """
    Get params from layers after pruning
    """

    if isinstance(layer, (nn.Conv2d, MaskConv2d)):
        # if isinstance(layer, MaskConv2d) and dim == 1:
        #     layer.pruned_weight.data.mul_(layer.d.data.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        if isinstance(layer, MaskConv2d):
            layer.weight.data = layer.pruned_weight.clone().data
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        if layer.bias is not None:
            if dim == 0:
                thin_bias = layer.bias.data.index_select(dim, select_channels)
            else:
                thin_bias = layer.bias.data
        else:
            thin_bias = None

    elif isinstance(layer, nn.BatchNorm2d):
        assert dim == 0, "invalid dimension for bn_layer"

        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_mean = layer.running_mean.index_select(dim, select_channels)
        thin_var = layer.running_var.index_select(dim, select_channels)
        if layer.bias is not None:
            thin_bias = layer.bias.data.index_select(dim, select_channels)
        else:
            thin_bias = None
        return (thin_weight, thin_mean), (thin_bias, thin_var)
    elif isinstance(layer, nn.PReLU):
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_bias = None

    return thin_weight, thin_bias


def replace_layer(old_layer, init_weight, init_bias=None, keeping=False):
    """
    Replace specific layer of model
    :params layer: original layer
    :params init_weight: thin_weight
    :params init_bias: thin_bias
    :params keeping: whether to keep MaskConv2d
    """

    if hasattr(old_layer, "bias") and old_layer.bias is not None:
        bias_flag = True
    else:
        bias_flag = False
    if isinstance(old_layer, MaskConv2d) and keeping:
        new_layer = MaskConv2d(
            init_weight.size(1),
            init_weight.size(0),
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=bias_flag)

        new_layer.pruned_weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)
        new_layer.d.copy_(old_layer.d)
        new_layer.float_weight.data.copy_(old_layer.d)

    elif isinstance(old_layer, (nn.Conv2d, MaskConv2d)):
        if old_layer.groups != 1:
            new_groups = init_weight.size(0)
            in_channels = init_weight.size(0)
            out_channels = init_weight.size(0)
        else:
            new_groups = 1
            in_channels = init_weight.size(1)
            out_channels = init_weight.size(0)

        new_layer = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=old_layer.kernel_size,
                              stride=old_layer.stride,
                              padding=old_layer.padding,
                              bias=bias_flag,
                              groups=new_groups)

        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, nn.BatchNorm2d):
        weight = init_weight[0]
        mean_ = init_weight[1]
        bias = init_bias[0]
        var_ = init_bias[1]
        new_layer = nn.BatchNorm2d(weight.size(0))
        new_layer.weight.data.copy_(weight)
        assert init_bias is not None, "batch normalization needs bias"
        new_layer.bias.data.copy_(bias)
        new_layer.running_mean.copy_(mean_)
        new_layer.running_var.copy_(var_)
    elif isinstance(old_layer, nn.PReLU):
        new_layer = nn.PReLU(init_weight.size(0))
        new_layer.weight.data.copy_(init_weight)

    else:
        assert False, "unsupport layer type:" + \
                      str(type(old_layer))
    return new_layer


# resnet only ---------------------------------------------------------------------------
class ResBlockPrune(object):
    """
    Residual block pruning
    """

    def __init__(self, block, block_type):
        self.block = block
        self.block_type = block_type
        self.select_channels = None

    def pruning(self):
        """
        Perform pruning
        """

        # prune pre-resnet on cifar
        if self.block_type in ["preresnet"]:
            if self.block.conv2.d.sum() == 0:
                self.block = self.block.downsample
                logger.info("remove whole block")
                return None
            # compute selected channels
            select_channels = get_select_channels(self.block.conv2.d)
            self.select_channels = select_channels

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

            # prune and replace bn2
            thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
            self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)
            # self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias, keeping=True)
            self.block.cuda()

        # prune shallow resnet on imagenet
        elif self.block_type == "resnet_basic":
            if self.block.conv2.d.sum() == 0:
                self.block = self.block.downsample
                logger.info("remove whole block")
                return None

            # compute selected channels
            select_channels = get_select_channels(self.block.conv2.d)
            self.select_channels = select_channels

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(self.block.bn1, select_channels, 0)
            self.block.bn1 = replace_layer(self.block.bn1, thin_weight, thin_bias)

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

        # prune deep resnet on imagenet
        elif self.block_type == "resnet_bottleneck":
            if (self.block.conv2.d.sum() == 0
                    or self.block.conv3.d.sum() == 0):
                self.block = self.block.downsample
                logger.info("remove whole block")
                return None

            # compute selected channels of conv2
            select_channels = get_select_channels(self.block.conv2.d)
            self.select_channels = select_channels

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(self.block.bn1, select_channels, 0)
            self.block.bn1 = replace_layer(self.block.bn1, thin_weight, thin_bias)

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)  # , keeping=True)

            self.block.cuda()
            # compute selected channels of conv3
            select_channels = get_select_channels(self.block.conv3.d)

            # prune and replace conv3
            thin_weight, thin_bias = get_thin_params(self.block.conv3, select_channels, 1)
            self.block.conv3 = replace_layer(self.block.conv3, thin_weight, thin_bias)

            # prune and replace bn2
            thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
            self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 0)
            self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)
            self.block.cuda()

        else:
            assert False, "invalid block type: " + self.block_type


class ResSeqPrune(object):
    """
    Sequantial pruning
    """

    def __init__(self, sequential, seq_type):
        self.sequential = sequential
        self.sequential_length = len(list(self.sequential))
        self.res_block_prune = []
        self.select_channels = None

        for i in range(self.sequential_length):
            self.res_block_prune.append(
                ResBlockPrune(self.sequential[i], block_type=seq_type))

    def pruning(self):
        """
        Perform pruning
        """

        for i in range(self.sequential_length):
            self.res_block_prune[i].pruning()

        temp_seq = []
        for i in range(self.sequential_length):
            if self.res_block_prune[i].block is not None:
                temp_seq.append(self.res_block_prune[i].block)
        self.sequential = nn.Sequential(*temp_seq)
        self.select_channels = self.res_block_prune[0].select_channels


class ResModelPrune(object):
    """
    Prune residual networks
    """

    def __init__(self, model, net_type, depth):
        self.model = model
        if net_type == "resnet":
            if depth >= 50:
                self.net_type = "resnet_bottleneck"
            else:
                self.net_type = "resnet_basic"
        else:
            self.net_type = net_type
        logger.info("|===>Init ResModelPrune")

    def run(self):
        """
        Perform pruning
        """

        if self.net_type in ["resnet_basic", "resnet_bottleneck"]:
            res_seq_prune = [
                ResSeqPrune(self.model.layer1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer2, seq_type=self.net_type),
                ResSeqPrune(self.model.layer3, seq_type=self.net_type),
                ResSeqPrune(self.model.layer4, seq_type=self.net_type)
            ]

            for i in range(4):
                res_seq_prune[i].pruning()
            self.model.layer1 = res_seq_prune[0].sequential
            self.model.layer2 = res_seq_prune[1].sequential
            self.model.layer3 = res_seq_prune[2].sequential
            self.model.layer4 = res_seq_prune[3].sequential
            self.model.cuda()
            logger.info(self.model)

        elif self.net_type in ["preresnet"]:
            res_seq_prune = [
                ResSeqPrune(self.model.layer1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer2, seq_type=self.net_type),
                ResSeqPrune(self.model.layer3, seq_type=self.net_type)
            ]
            for i in range(3):
                res_seq_prune[i].pruning()

            self.model.layer1 = res_seq_prune[0].sequential
            self.model.layer2 = res_seq_prune[1].sequential
            self.model.layer3 = res_seq_prune[2].sequential
            logger.info(self.model)
            self.model.cuda()
