import torch

block_num = {'preresnet56': 27, 'resnet18': 8, 'resnet50': 16}

def concat_gpu_data(data):
    """
    Concat gpu data from different gpu.
    """

    data_cat = data["0"]
    for i in range(1, len(data)):
        data_cat = torch.cat((data_cat, data[str(i)].cuda(0)))
    return data_cat


def cal_pivot(n_losses, net_type, depth, logger):
    """
    Calculate the inserted layer for additional loss
    """

    num_segments = n_losses + 1
    network_block_num = block_num[net_type + str(depth)]
    num_block_per_segment = (network_block_num // num_segments) + 1
    pivot_set = []
    for i in range(num_segments - 1):
        pivot_set.append(min(num_block_per_segment * (i + 1), network_block_num - 1))
    logger.info("pivot set: {}".format(pivot_set))
    return num_segments, pivot_set
