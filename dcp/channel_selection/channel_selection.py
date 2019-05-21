import datetime
import math
import os
import time

import torch
import torch.nn as nn

import dcp.utils as utils
from dcp.mask_conv import MaskConv2d
from dcp.utils.others import concat_gpu_data
from dcp.utils.write_log import write_log


class LayerChannelSelection(object):
    """
    Discrimination-aware channel selection
    """

    def __init__(self, trainer, train_loader, val_loader, settings, checkpoint, logger, tensorboard_logger):
        self.segment_wise_trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.settings = settings
        self.checkpoint = checkpoint

        self.logger = logger
        self.tensorboard_logger = tensorboard_logger

        self.feature_cache_original_input = {}
        self.feature_cache_original_output = {}
        self.feature_cache_pruned_input = {}
        self.feature_cache_pruned_output = {}

        self.criterion_mse = nn.MSELoss().cuda()
        self.criterion_softmax = nn.CrossEntropyLoss().cuda()

        self.logger_counter = 0

        self.record_time = utils.AverageMeter()
        self.record_selection_mse_loss = utils.AverageMeter()
        self.record_selection_softmax_loss = utils.AverageMeter()
        self.record_selection_loss = utils.AverageMeter()
        self.record_sub_problem_softmax_loss = utils.AverageMeter()
        self.record_sub_problem_mse_loss = utils.AverageMeter()
        self.record_sub_problem_loss = utils.AverageMeter()
        self.record_sub_problem_top1_error = utils.AverageMeter()
        self.record_sub_problem_top5_error = utils.AverageMeter()

    def split_segment_into_three_parts(self, original_segment, pruned_segment, block_count):
        """
        Split the segment into three parts:
            segment_before_pruned_module, pruned_module, segment_after_pruned_module.
        In this way, we can store the input of the pruned module.
        """

        original_segment_list = utils.model2list(original_segment)
        pruned_segment_list = utils.model2list(pruned_segment)

        original_segment_before_pruned_module = []
        pruned_segment_before_pruned_module = []
        pruned_segment_after_pruned_module = []
        for i in range(len(pruned_segment)):
            if i < block_count:
                original_segment_before_pruned_module.append(original_segment_list[i])
                pruned_segment_before_pruned_module.append(pruned_segment_list[i])
            if i > block_count:
                pruned_segment_after_pruned_module.append(pruned_segment_list[i])
        self.original_segment_before_pruned_module = nn.Sequential(*original_segment_before_pruned_module)
        self.pruned_segment_before_pruned_module = nn.Sequential(*pruned_segment_before_pruned_module)
        self.pruned_segment_after_pruned_module = nn.Sequential(*pruned_segment_after_pruned_module)

    @staticmethod
    def _replace_layer(net, layer, layer_index):
        assert isinstance(net, nn.Sequential), "only support nn.Sequential"
        new_net = None

        count = 0
        for origin_layer in net:
            if count == layer_index:
                if new_net is None:
                    new_net = nn.Sequential(layer)
                else:
                    new_net.add_module(str(len(new_net)), layer)
            else:
                if new_net is None:
                    new_net = nn.Sequential(origin_layer)
                else:
                    new_net.add_module(str(len(new_net)), origin_layer)
            count += 1
        return new_net

    def replace_layer_with_mask_conv(self, pruned_segment, module, layer_name, block_count):
        """
        Replace the pruned layer with mask convolution
        """

        if layer_name == "conv2":
            layer = module.conv2
        elif layer_name == "conv3":
            layer = module.conv3
        elif layer_name == "conv":
            assert self.settings.net_type in ["vgg"], "only support vgg"
            layer = module
        else:
            assert False, "unsupport layer: {}".format(layer_name)

        if not isinstance(layer, MaskConv2d):
            temp_conv = MaskConv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=(layer.bias is not None))
            temp_conv.weight.data.copy_(layer.weight.data)

            if layer.bias is not None:
                temp_conv.bias.data.copy_(layer.bias.data)
            temp_conv.pruned_weight.data.fill_(0)
            temp_conv.d.fill_(0)

            if layer_name == "conv2":
                module.conv2 = temp_conv
            elif layer_name == "conv3":
                module.conv3 = temp_conv
            elif layer_name == "conv":
                pruned_segment = self._replace_layer(net=pruned_segment,
                                                     layer=temp_conv,
                                                     layer_index=block_count)
            layer = temp_conv
        return pruned_segment, layer

    def _hook_origin_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        self.feature_cache_original_input[gpu_id] = input[0]
        self.feature_cache_original_output[gpu_id] = output

    def _hook_pruned_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        self.feature_cache_pruned_input[gpu_id] = input[0]
        self.feature_cache_pruned_output[gpu_id] = output

    def register_layer_hook(self, original_segment, pruned_segment, module, layer_name, block_count):
        """
        In order to get the input and the output of the intermediate layer, we register
        the forward hook for the pruned layer
        """

        if layer_name == "conv2":
            self.hook_origin = original_segment[block_count].conv2.register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = module.conv2.register_forward_hook(self._hook_pruned_feature)
        elif layer_name == "conv3":
            self.hook_origin = original_segment[block_count].conv3.register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = module.conv3.register_forward_hook(self._hook_pruned_feature)
        elif layer_name == "conv":
            self.hook_origin = original_segment[block_count].register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = pruned_segment[block_count].register_forward_hook(self._hook_pruned_feature)

    def segment_parallelism(self, original_segment, pruned_segment):
        """
        Parallel setting for segment
        """

        self.original_segment_parallel = utils.data_parallel(original_segment, self.settings.n_gpus)
        self.pruned_segment_parallel = utils.data_parallel(pruned_segment, self.settings.n_gpus)

    def reset_average_meter(self):
        self.record_time.reset()
        self.record_selection_mse_loss.reset()
        self.record_selection_softmax_loss.reset()
        self.record_selection_loss.reset()
        self.record_sub_problem_softmax_loss.reset()
        self.record_sub_problem_mse_loss.reset()
        self.record_sub_problem_loss.reset()
        self.record_sub_problem_top1_error.reset()
        self.record_sub_problem_top5_error.reset()

    def prepare_channel_selection(self, original_segment, pruned_segment, module, aux_fc, layer_name, block_count):
        """
        Prepare for channel selection
        1. Split the segment into three parts.
        2. Replace the pruned layer with mask convolution.
        3. Store the input feature map of the pruned layer in advance to accelerate channel selection.
        """

        self.split_segment_into_three_parts(original_segment, pruned_segment, block_count)
        pruned_segment, layer = self.replace_layer_with_mask_conv(pruned_segment, module, layer_name, block_count)
        self.register_layer_hook(original_segment, pruned_segment, module, layer_name, block_count)

        # parallel setting
        self.segment_parallelism(original_segment, pruned_segment)

        # turn gradient off
        # avoid computing the gradient
        for params in self.original_segment_parallel.parameters():
            params.requires_grad = False
        for params in self.pruned_segment_parallel.parameters():
            params.requires_grad = False

        # freeze the Batch Normalization
        self.original_segment_parallel.eval()
        self.pruned_segment_parallel.eval()
        aux_fc.eval()

        self.num_batch = len(self.train_loader)

        # turn on the gradient with respect to the pruned layer
        layer.pruned_weight.requires_grad = True
        aux_fc.cuda()

        self.logger_counter = 0
        return pruned_segment, layer

    def get_batch_data(self, train_dataloader_iter):
        images, labels = train_dataloader_iter.next()
        images = images.cuda()
        labels = labels.cuda()
        original_module_input_feature = self.original_segment_before_pruned_module(images)
        pruned_module_input_feature = self.pruned_segment_before_pruned_module(images)
        return original_module_input_feature, pruned_module_input_feature, labels

    def compute_loss_error(self, original_segment, pruned_segment, block_count, aux_fc,
                           original_module_input_feature, pruned_module_input_feature, labels):
        """
        Compute the total loss, softmax_loss, mse_loss, top1_error and top5_error
        """

        # forward propagation
        original_segment[block_count](original_module_input_feature)
        pruned_output = pruned_segment[block_count](pruned_module_input_feature)
        fc_output = aux_fc(self.pruned_segment_after_pruned_module(pruned_output))

        # get the output feature of the pruned layer
        origin_output = concat_gpu_data(self.feature_cache_original_output)
        pruned_output = concat_gpu_data(self.feature_cache_pruned_output)

        # compute loss
        softmax_loss = self.criterion_softmax(fc_output, labels)
        mse_loss = self.criterion_mse(pruned_output, origin_output.detach())
        loss = mse_loss * self.settings.mse_weight + softmax_loss * self.settings.softmax_weight

        top1_error, _, top5_error = utils.compute_singlecrop_error(
            outputs=fc_output, labels=labels,
            loss=softmax_loss, top5_flag=True)
        return loss, self.settings.softmax_weight * softmax_loss, self.settings.mse_weight * mse_loss, \
               top1_error, top5_error

    def find_maximum_grad_fnorm(self, grad_fnorm, layer):
        """
        Find the channel index with maximum gradient frobenius norm,
        and initialize the pruned_weight w.r.t. the selected channel.
        """

        grad_fnorm.data.mul_(1 - layer.d).sub_(layer.d)
        _, max_index = torch.topk(grad_fnorm, 1)
        layer.d[max_index] = 1
        # warm-started from the pre-trained model
        if self.settings.warmstart:
            layer.pruned_weight.data[:, max_index, :, :] = layer.weight[:, max_index, :, :].data.clone()

    def find_most_violated(self, original_segment, pruned_segment, aux_fc, layer, block_count):
        """
        Find the channel with maximum gradient frobenius norm.
        :param original_segment: original network segments
        :param pruned_segment: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param layer: the layer to be pruned
        :param block_count: current block no.
        """

        layer.pruned_weight.grad = None
        train_dataloader_iter = iter(self.train_loader)

        for j in range(int(self.num_batch)):
            # get data
            original_module_input_feature, pruned_module_input_feature, labels = \
                self.get_batch_data(train_dataloader_iter)

            loss, softmax_loss, mse_loss, top1_error, top5_error = \
                self.compute_loss_error(original_segment, pruned_segment, block_count, aux_fc,
                                        original_module_input_feature, pruned_module_input_feature, labels)

            loss.backward()

            self.record_selection_loss.update(loss.item(), labels.size(0))
            self.record_selection_mse_loss.update(mse_loss.item(), labels.size(0))
            self.record_selection_softmax_loss.update(softmax_loss.item(), labels.size(0))

        cum_grad = layer.pruned_weight.grad.data.clone()
        layer.pruned_weight.grad = None

        # calculate F norm of gradient
        grad_fnorm = cum_grad.mul(cum_grad).sum((2, 3)).sqrt().sum(0)

        # find grad_fnorm with maximum absolute gradient
        self.find_maximum_grad_fnorm(grad_fnorm, layer)

    def set_layer_wise_optimizer(self, layer):
        params_list = []
        params_list.append({"params": layer.pruned_weight, "lr": self.settings.layer_wise_lr})
        if layer.bias is not None:
            layer.bias.requires_grad = True
            params_list.append({"params": layer.bias, "lr": self.settings.layer_wise_lr})

        optimizer = torch.optim.SGD(params=params_list,
                                    weight_decay=self.settings.weight_decay,
                                    momentum=self.settings.momentum,
                                    nesterov=True)

        # self.logger.info(optimizer)
        # assert False
        return optimizer

    def record_epoch_loss_tensorboard_logger(self, loss, softmax_loss, mse_loss,
                                             top1_error, top5_error, lr, block_count, layer_name):
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_epoch_SoftmaxLoss".format(block_count, layer_name),
            value=softmax_loss,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_epoch_Loss".format(block_count, layer_name),
            value=loss,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_epoch_MSELoss".format(block_count, layer_name),
            value=mse_loss,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_epoch_Top1Error".format(block_count, layer_name),
            value=top1_error,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_epoch_Top5Error".format(block_count, layer_name),
            value=top5_error,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_epoch_Lr".format(block_count, layer_name),
            value=lr,
            step=self.logger_counter)

    def solve_sub_problem(self, original_segment, pruned_segment, aux_fc, layer, block_count, layer_name):
        """
        We optimize W w.r.t. the selected channels by minimizing the problem (8)
        :param original_segment: original network segments
        :param pruned_segment: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param layer: the layer to be pruned
        :param block_count: current block no.
        """

        optimizer = self.set_layer_wise_optimizer(layer)
        train_dataloader_iter = iter(self.train_loader)

        for j in range(self.num_batch):
            # get data
            original_module_input_feature, pruned_module_input_feature, labels = \
                self.get_batch_data(train_dataloader_iter)

            loss, softmax_loss, mse_loss, top1_error, top5_error = \
                self.compute_loss_error(original_segment, pruned_segment, block_count, aux_fc,
                                        original_module_input_feature, pruned_module_input_feature, labels)

            optimizer.zero_grad()
            # compute gradient
            loss.backward()
            # we only optimize W with respect to the selected channel
            layer.pruned_weight.grad.data.mul_(
                layer.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(layer.pruned_weight))
            optimizer.step()

            # update record info
            self.record_sub_problem_softmax_loss.update(softmax_loss.item(), labels.size(0))
            self.record_sub_problem_mse_loss.update(mse_loss.item(), labels.size(0))
            self.record_sub_problem_loss.update(loss.item(), labels.size(0))
            self.record_sub_problem_top1_error.update(top1_error, labels.size(0))
            self.record_sub_problem_top5_error.update(top5_error, labels.size(0))

        layer.pruned_weight.grad = None
        if layer.bias is not None:
            layer.bias.grad = None
        if layer.bias is not None:
            layer.bias.requires_grad = False

    def write_log(self, layer, block_count, layer_name):
        self.write_tensorboard_log(block_count, layer_name)
        self.write_log2file(layer, block_count, layer_name)

    def write_tensorboard_log(self, block_count, layer_name):
        self.tensorboard_logger.scalar_summary(
            tag="Selection-block-{}_{}_LossAll".format(block_count, layer_name),
            value=self.record_selection_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Selection-block-{}_{}_MSELoss".format(block_count, layer_name),
            value=self.record_selection_mse_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Selection-block-{}_{}_SoftmaxLoss".format(block_count, layer_name),
            value=self.record_selection_softmax_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_SoftmaxLoss".format(block_count, layer_name),
            value=self.record_sub_problem_softmax_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_Loss".format(block_count, layer_name),
            value=self.record_sub_problem_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_MSELoss".format(block_count, layer_name),
            value=self.record_sub_problem_mse_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_Top1Error".format(block_count, layer_name),
            value=self.record_sub_problem_top1_error.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_Top5Error".format(block_count, layer_name),
            value=self.record_sub_problem_top5_error.avg,
            step=self.logger_counter)
        self.logger_counter += 1

    def write_log2file(self, layer, block_count, layer_name):
        write_log(
            dir_name=os.path.join(self.settings.save_path, "log"),
            file_name="log_block-{:0>2d}_{}.txt".format(block_count, layer_name),
            log_str="{:d}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t\n".format(
                int(layer.d.sum()),
                self.record_selection_loss.avg,
                self.record_selection_mse_loss.avg,
                self.record_selection_softmax_loss.avg,
                self.record_sub_problem_loss.avg,
                self.record_sub_problem_mse_loss.avg,
                self.record_sub_problem_softmax_loss.avg,
                self.record_sub_problem_top1_error.avg,
                self.record_sub_problem_top5_error.avg))
        log_str = "Block-{:0>2d}-{}  #channels: [{:0>4d}|{:0>4d}]  ".format(
            block_count, layer_name,
            int(layer.d.sum()), layer.d.size(0))
        log_str += "[selection]loss: {:4f}  mseloss: {:4f}  softmaxloss: {:4f}  ".format(
            self.record_selection_loss.avg,
            self.record_selection_mse_loss.avg,
            self.record_selection_softmax_loss.avg)
        log_str += "[subproblem]loss: {:4f}  mseloss: {:4f}  softmaxloss: {:4f}  ".format(
            self.record_sub_problem_loss.avg,
            self.record_sub_problem_mse_loss.avg,
            self.record_sub_problem_softmax_loss.avg)
        log_str += "top1error: {:4f}  top5error: {:4f}  ".format(
            self.record_sub_problem_top1_error.avg,
            self.record_sub_problem_top5_error.avg)
        self.logger.info(log_str)

    def remove_layer_hook(self):
        self.hook_origin.remove()
        self.hook_pruned.remove()
        self.logger.info("|===>remove hook")

    def channel_selection_for_one_layer(self, original_segment, pruned_segment,
                                        aux_fc, module, block_count, layer_name="conv2"):
        """
        Conduct channel selection for one layer in a module
        :param original_segment: original network segments
        :param pruned_segment: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param module: the module need to be pruned
        :param block_count: current block no.
        :param layer_name: the name of layer need to be pruned
        """

        # layer-wise channel selection
        self.logger.info("|===>layer-wise channel selection: block-{}-{}".format(block_count, layer_name))
        pruned_segment, layer = self.prepare_channel_selection(original_segment, pruned_segment, module,
                                                               aux_fc, layer_name, block_count)

        for channel in range(layer.in_channels):
            if layer.d.eq(0).sum() <= math.floor(layer.in_channels * self.settings.pruning_rate):
                break

            self.reset_average_meter()

            time_start = time.time()
            # find the channel with the maximum gradient norm
            self.find_most_violated(original_segment, pruned_segment, aux_fc, layer, block_count)
            # solve problem (8) w.r.t. the selected channels
            self.solve_sub_problem(original_segment, pruned_segment, aux_fc, layer, block_count, layer_name)
            time_interval = time.time() - time_start

            self.write_log(layer, block_count, layer_name)
            self.record_time.update(time_interval)

        self.tensorboard_logger.scalar_summary(
            tag="Channel_num",
            value=layer.d.eq(1).sum(),
            step=block_count)
        log_str = "|===>Select channel from block-{:d}_{}: time_total:{} time_avg: {}".format(
            block_count, layer_name,
            str(datetime.timedelta(seconds=self.record_time.sum)),
            str(datetime.timedelta(seconds=self.record_time.avg)))
        self.logger.info(log_str)

        # turn requires_grad on
        for params in self.original_segment_parallel.parameters():
            params.requires_grad = True
        for params in self.pruned_segment_parallel.parameters():
            params.requires_grad = True
        self.remove_layer_hook()
        return pruned_segment
