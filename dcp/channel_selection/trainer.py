import math
import time

import torch.autograd
import torch.nn as nn

import dcp.utils as utils
from dcp.mask_conv import MaskConv2d
from dcp.aux_classifier import AuxClassifier
from dcp.models.preresnet import PreBasicBlock
from dcp.models.resnet import BasicBlock, Bottleneck


class View(nn.Module):
    """
    Reshape data from 4 dimension to 2 dimension
    """

    def forward(self, x):
        assert x.dim() == 2 or x.dim() == 4, "invalid dimension of input {:d}".format(x.dim())
        if x.dim() == 4:
            out = x.view(x.size(0), -1)
        else:
            out = x
        return out


class SegmentWiseTrainer(object):
    """
        Segment-wise trainer for channel selection
    """

    def __init__(self, original_model, pruned_model, train_loader, val_loader, settings, logger, tensorboard_logger,
                 run_count=0):

        self.original_model = original_model
        self.pruned_model = pruned_model
        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.lr = self.settings.lr
        self.ori_segments = []
        self.pruned_segments = []
        self.seg_optimizer = []
        self.aux_fc = []
        self.fc_optimizer = []
        self.run_count = run_count

        # run pre-processing
        self.insert_additional_losses()

    def create_segments(self):
        net_origin = None
        net_pruned = None

        if self.settings.net_type in ["preresnet", "resnet"]:
            if self.settings.net_type == "preresnet":
                net_origin = nn.Sequential(self.original_model.conv)
                net_pruned = nn.Sequential(self.pruned_model.conv)
            elif self.settings.net_type == "resnet":
                net_head = nn.Sequential(
                    self.original_model.conv1,
                    self.original_model.bn1,
                    self.original_model.relu,
                    self.original_model.maxpool)
                net_origin = nn.Sequential(net_head)
                net_head = nn.Sequential(
                    self.pruned_model.conv1,
                    self.pruned_model.bn1,
                    self.pruned_model.relu,
                    self.pruned_model.maxpool)
                net_pruned = nn.Sequential(net_head)
        else:
            self.logger.info("unsupported net_type: {}".format(self.settings.net_type))
            assert False, "unsupported net_type: {}".format(self.settings.net_type)
        self.logger.info("init shallow head done!")

        block_count = 0
        if self.settings.net_type in ["resnet", "preresnet"]:
            for ori_module, pruned_module in zip(self.original_model.modules(), self.pruned_model.modules()):
                if isinstance(ori_module, (PreBasicBlock, Bottleneck, BasicBlock)):
                    self.logger.info("enter block: {}".format(type(ori_module)))
                    if net_origin is not None:
                        net_origin.add_module(str(len(net_origin)), ori_module)
                    else:
                        net_origin = nn.Sequential(ori_module)

                    if net_pruned is not None:
                        net_pruned.add_module(str(len(net_pruned)), pruned_module)
                    else:
                        net_pruned = nn.Sequential(pruned_module)
                    block_count += 1

                    # if block_count is equals to pivot_num, then create new segment
                    if block_count in self.settings.pivot_set:
                        self.ori_segments.append(net_origin)
                        self.pruned_segments.append(net_pruned)
                        net_origin = None
                        net_pruned = None

        self.final_block_count = block_count
        self.ori_segments.append(net_origin)
        self.pruned_segments.append(net_pruned)

    def create_auxiliary_classifiers(self):
        # create auxiliary classifier
        num_classes = self.settings.n_classes
        in_channels = 0
        for i in range(len(self.pruned_segments) - 1):
            if isinstance(self.pruned_segments[i][-1], (PreBasicBlock, BasicBlock)):
                in_channels = self.pruned_segments[i][-1].conv2.out_channels
            elif isinstance(self.pruned_segments[i][-1], Bottleneck):
                in_channels = self.pruned_segments[i][-1].conv3.out_channels
            else:
                self.logger.error("Nonsupport layer type!")
                assert False, "Nonsupport layer type!"
            assert in_channels != 0, "in_channels is zero"

            self.aux_fc.append(AuxClassifier(in_channels=in_channels, num_classes=num_classes))

        pruned_final_fc = None
        if self.settings.net_type == "preresnet":
            pruned_final_fc = nn.Sequential(*[
                self.pruned_model.bn,
                self.pruned_model.relu,
                self.pruned_model.avg_pool,
                View(),
                self.pruned_model.fc])
        elif self.settings.net_type == "resnet":
            pruned_final_fc = nn.Sequential(*[
                self.pruned_model.avgpool,
                View(),
                self.pruned_model.fc])
        else:
            self.logger.error("Nonsupport net type: {}!".format(self.settings.net_type))
            assert False, "Nonsupport net type: {}!".format(self.settings.net_type)
        self.aux_fc.append(pruned_final_fc)

    def model_parallelism(self):
        self.ori_segments = utils.data_parallel(model=self.ori_segments, n_gpus=self.settings.n_gpus)
        self.pruned_segments = utils.data_parallel(model=self.pruned_segments, n_gpus=self.settings.n_gpus)
        self.aux_fc = utils.data_parallel(model=self.aux_fc, n_gpus=1)

    def create_optimizers(self):
        # create optimizers
        for i in range(len(self.pruned_segments)):
            temp_optim = []
            # add parameters in segmenets into optimizer
            # from the i-th optimizer contains [0:i] segments
            for j in range(i + 1):
                temp_optim.append({'params': self.pruned_segments[j].parameters(),
                                   'lr': self.settings.lr})

            # optimizer for segments and fc
            temp_seg_optim = torch.optim.SGD(
                temp_optim,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weight_decay,
                nesterov=True)

            temp_fc_optim = torch.optim.SGD(
                params=self.aux_fc[i].parameters(),
                lr=self.settings.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weight_decay,
                nesterov=True)

            self.seg_optimizer.append(temp_seg_optim)
            self.fc_optimizer.append(temp_fc_optim)

    def insert_additional_losses(self):
        """"
            1. Split the network into several segments with pre-define pivot set
            2. Create auxiliary classifiers
            3. Create optimizers for network segments and fcs
        """

        self.create_segments()
        self.create_auxiliary_classifiers()

        # parallel setting for segments and auxiliary classifiers
        self.model_parallelism()

        self.create_optimizers()

    @staticmethod
    def _convert_results(top1_error, top1_loss, top5_error):
        """
        Convert tensor list to float list
        :param top1_error: top1_error tensor list
        :param top1_loss:  top1_loss tensor list
        :param top5_error:  top5_error tensor list
        """

        assert isinstance(top1_error, list), "input should be a list"
        length = len(top1_error)
        top1_error_list = []
        top5_error_list = []
        top1_loss_list = []
        for i in range(length):
            top1_error_list.append(top1_error[i].avg)
            top5_error_list.append(top5_error[i].avg)
            top1_loss_list.append(top1_loss[i].avg)
        return top1_error_list, top1_loss_list, top5_error_list

    def update_model(self, original_model, pruned_model, aux_fc_state=None, aux_fc_opt_state=None, seg_opt_state=None):
        """
        Update model parameter and optimizer state
        :param original_model: baseline model
        :param pruned_model: pruned model
        :param aux_fc_state: state dict of auxiliary fully-connected layer
        :param aux_fc_opt_state: optimizer's state dict of auxiliary fully-connected layer
        :param seg_opt_state: optimizer's state dict of segment
        """

        self.ori_segments = []
        self.pruned_segments = []
        self.seg_optimizer = []
        self.aux_fc = []
        self.fc_optimizer = []

        self.original_model = original_model
        self.pruned_model = pruned_model
        self.insert_additional_losses()
        if aux_fc_state is not None:
            self.update_aux_fc(aux_fc_state, aux_fc_opt_state, seg_opt_state)

    def update_aux_fc(self, aux_fc_state, aux_fc_opt_state=None, seg_opt_state=None):
        """
        Update auxiliary classifier parameter and optimizer state
        :param aux_fc_state: state dict of auxiliary fully-connected layer
        :param aux_fc_opt_state: optimizer's state dict of auxiliary fully-connected layer
        :param seg_opt_state: optimizer's state dict of segment
        """

        if len(self.aux_fc) == 1:
            if isinstance(self.aux_fc[0], nn.DataParallel):
                self.aux_fc[0].module.load_state_dict(aux_fc_state[-1])
            else:
                self.aux_fc[0].load_state_dict(aux_fc_state[-1])
            if aux_fc_opt_state is not None:
                self.fc_optimizer[0].load_state_dict(aux_fc_opt_state[-1])
            if seg_opt_state is not None:
                self.seg_optimizer[0].load_state_dict(seg_opt_state[-1])

        elif len(self.aux_fc) == len(aux_fc_state):
            for i in range(len(aux_fc_state)):
                if isinstance(self.aux_fc[i], nn.DataParallel):
                    self.aux_fc[i].module.load_state_dict(aux_fc_state[i])
                else:
                    self.aux_fc[i].load_state_dict(aux_fc_state[i])

                if aux_fc_opt_state is not None:
                    self.fc_optimizer[i].load_state_dict(aux_fc_opt_state[i])
                if seg_opt_state is not None:
                    self.seg_optimizer[i].load_state_dict(seg_opt_state[i])
        else:
            assert False, "size not match! len(self.aux_fc)={:d}, len(aux_fc_state)={:d}".format(
                len(self.aux_fc), len(aux_fc_state))

    def auxnet_forward(self, images, labels=None):
        """
        Forward propagation
        """

        outputs = []
        temp_input = images
        losses = []
        for i in range(len(self.pruned_segments)):
            # forward
            temp_output = self.pruned_segments[i](temp_input)
            fcs_output = self.aux_fc[i](temp_output)
            outputs.append(fcs_output)
            if labels is not None:
                losses.append(self.criterion(fcs_output, labels))
            temp_input = temp_output
        return outputs, losses

    @staticmethod
    def _correct_nan(grad):
        """
        Fix nan
        :param grad: gradient input
        """

        grad.masked_fill_(grad.ne(grad), 0)
        return grad

    def auxnet_backward_for_loss_i(self, loss, i):
        """
        Backward propagation for the i-th loss
        :param loss: the i-th loss
        :param i: which one to perform backward propagation
        """

        self.seg_optimizer[i].zero_grad()
        self.fc_optimizer[i].zero_grad()

        loss.backward(retain_graph=True)
        # set the gradient of unselected channel to zero
        for j in range(len(self.pruned_segments)):
            if isinstance(self.pruned_segments[i], nn.DataParallel):
                for module in self.pruned_segments[j].module.modules():
                    if isinstance(module, MaskConv2d) and module.pruned_weight.grad is not None:
                        module.pruned_weight.grad.data.mul_(
                            module.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(module.pruned_weight))
            else:
                for module in self.pruned_segments[j].modules():
                    if isinstance(module, MaskConv2d) and module.pruned_weight.grad is not None:
                        module.pruned_weight.grad.data.mul_(
                            module.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(module.pruned_weight))

        self.fc_optimizer[i].step()
        self.seg_optimizer[i].step()

    def update_lr(self, epoch):
        """
        Update learning rate of optimizers
        :param epoch: index of epoch
        """

        gamma = 0
        for step in self.settings.step:
            if epoch + 1.0 > int(step):
                gamma += 1
        lr = self.settings.lr * math.pow(0.1, gamma)
        self.lr = lr

        for i in range(len(self.seg_optimizer)):
            for param_group in self.seg_optimizer[i].param_groups:
                param_group['lr'] = lr

            for param_group in self.fc_optimizer[i].param_groups:
                param_group['lr'] = lr

    def val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """

        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.pruned_segments)
        for i in range(num_segments):
            self.pruned_segments[i].eval()
            self.aux_fc[i].eval()
            top1_error.append(utils.AverageMeter())
            top5_error.append(utils.AverageMeter())
            top1_loss.append(utils.AverageMeter())

        iters = len(self.val_loader)

        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                start_time = time.time()
                data_time = start_time - end_time

                if self.settings.n_gpus == 1:
                    images = images.cuda()
                labels = labels.cuda()

                outputs, losses = self.auxnet_forward(images, labels)

                # compute loss and error rate
                single_error, single_loss, single5_error = utils.compute_singlecrop_error(
                    outputs=outputs, labels=labels,
                    loss=losses, top5_flag=True)

                for j in range(num_segments):
                    top1_error[j].update(single_error[j], images.size(0))
                    top5_error[j].update(single5_error[j], images.size(0))
                    top1_loss[j].update(single_loss[j], images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

                utils.print_result(epoch, 1, i + 1,
                                   iters, self.lr, data_time, iter_time,
                                   single_error,
                                   single_loss,
                                   mode="Validation",
                                   logger=self.logger)

        top1_error_list, top1_loss_list, top5_error_list = self._convert_results(
            top1_error=top1_error, top1_loss=top1_loss, top5_error=top5_error)

        if self.logger is not None:
            for i in range(num_segments):
                self.tensorboard_logger.scalar_summary(
                    "segment_wise_fine_tune_val_top1_error_{:d}".format(i), top1_error[i].avg, self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "segment_wise_fine_tune_val_top5_error_{:d}".format(i), top5_error[i].avg, self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "segment_wise_fine_tune_val_loss_{:d}".format(i), top1_loss[i].avg, self.run_count)
        self.run_count += 1

        self.logger.info("|===>Validation Error: {:4f}/{:4f}, Loss: {:4f}".format(
            top1_error[-1].avg, top5_error[-1].avg, top1_loss[-1].avg))
        return top1_error_list, top1_loss_list, top5_error_list