import math

import torch

__all__ = ["compute_singlecrop_error", "AverageMeter"]


def compute_singlecrop_error(outputs, labels, loss, top5_flag=False):
    """
    Compute singlecrop top-1 and top-5 error
    :param outputs: the output of the model
    :param labels: the ground truth of the data
    :param loss: the loss value of current batch
    :param top5_flag: whether to calculate the top-5 error
    :return: top-1 error list, loss list and top-5 error list
    """

    with torch.no_grad():
        if isinstance(outputs, list):
            top1_loss = []
            top1_error = []
            top5_error = []
            for i in range(len(outputs)):
                top1_accuracy, top5_accuracy = accuracy(outputs[i], labels, topk=(1, 5))
                top1_error.append(100 - top1_accuracy)
                top5_error.append(100 - top5_accuracy)
                top1_loss.append(loss[i].item())
        else:
            top1_accuracy, top5_accuracy = accuracy(outputs, labels, topk=(1, 5))
            top1_error = 100 - top1_accuracy
            top5_error = 100 - top5_accuracy
            top1_loss = loss.item()

        if top5_flag:
            return top1_error, top1_loss, top5_error
        else:
            return top1_error, top1_loss


def accuracy(outputs, labels, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    :param outputs: the outputs of the model
    :param labels: the ground truth of the data
    :param topk: the list of k in top-k
    :return: accuracy
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset all parameters
        """

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update parameters
        """

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def minus_inputs(inputs):
    outputs = []
    for i in range(len(inputs)):
        outputs.append(-inputs[i])
    return outputs


def compute_inner_product(inputs_a, inputs_b):
    cum_sum = 0
    for i in range(len(inputs_a)):
        cum_sum += ((inputs_a[i].data.mul(inputs_b[i].data)).sum()).item()
        # cum_sum += ((inputs_a[i] * inputs_b[i]).sum()).item()
    return cum_sum


def compute_cosine(inputs_a, inputs_b):
    a_b_inner_product = compute_inner_product(inputs_a, inputs_b)
    inputs_a_norm = math.sqrt(compute_inner_product(inputs_a, inputs_a))
    inputs_b_norm = math.sqrt(compute_inner_product(inputs_b, inputs_b))
    a_b_cosine = a_b_inner_product / (inputs_a_norm * inputs_b_norm)
    return a_b_cosine, a_b_inner_product, inputs_a_norm, inputs_b_norm
