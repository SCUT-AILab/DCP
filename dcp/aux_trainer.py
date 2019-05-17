import math
import time

import torch.autograd
import torch.nn as nn

import dcp.utils as utils
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


class AuxTrainer(object):
    """
    Trainer for auxnet
    """

    def __init__(self, model, train_loader, val_loader, settings, logger, tensorboard_logger, run_count=0):

        self.model = model
        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.lr = self.settings.lr
        self.segments = []
        self.aux_fc = []
        self.seg_optimizer = []
        self.fc_optimizer = []
        self.run_count = run_count

        # run pre-processing
        self.set_loss_weight()
        self.insert_additional_losses()

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

    def set_loss_weight(self):
        """
        The weight of the k-th auxiliary loss: gamma_k = \max(0.01, (\frac{L_k}{L_K})^2)
        More details can be found in Section 3.2 in "The Shallow End: Empowering Shallower Deep-Convolutional Networks
        through Auxiliary Outputs": https://arxiv.org/abs/1611.01773.
        """

        base_weight = 0
        self.lr_weight = torch.zeros(len(self.settings.pivot_set)).cuda()
        self.pivot_weight = self.lr_weight.clone()
        num_layers = 1
        if self.settings.net_type == "preresnet" or \
                (self.settings.net_type == "resnet" and self.settings.depth < 50):
            num_layers = 2
        elif self.settings.net_type == "resnet" and self.settings.depth >= 50:
            num_layers = 3
        for i in range(len(self.settings.pivot_set) - 1, -1, -1):
            temp_weight = max(pow(float(self.settings.pivot_set[i] * num_layers + 1) /
                                  (self.settings.pivot_set[-1] * num_layers + 1), 2), 0.01)
            base_weight += temp_weight
            self.pivot_weight[i] = temp_weight
            self.lr_weight[i] = base_weight

    def create_segments(self):
        """
        Split the network into several segments with pre-define pivot set
        """

        shallow_model = None

        if self.settings.net_type in ["preresnet", "resnet"]:
            if self.settings.net_type == "preresnet":
                shallow_model = nn.Sequential(self.model.conv)
            elif self.settings.net_type == "resnet":
                net_head = nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.relu,
                    self.model.maxpool)
                shallow_model = nn.Sequential(net_head)
        else:
            self.logger.info("unsupported net_type: {}".format(self.settings.net_type))
            assert False, "unsupported net_type: {}".format(self.settings.net_type)
        self.logger.info("init shallow head done!")

        block_count = 0
        if self.settings.net_type in ["resnet", "preresnet"]:
            for module in self.model.modules():
                if isinstance(module, (PreBasicBlock, Bottleneck, BasicBlock)):
                    self.logger.info("enter block: {}".format(type(module)))
                    if shallow_model is not None:
                        shallow_model.add_module(str(len(shallow_model)), module)
                    else:
                        shallow_model = nn.Sequential(module)
                    block_count += 1

                    # if block_count is equals to pivot_num, then create new segment
                    if block_count in self.settings.pivot_set:
                        self.segments.append(shallow_model)
                        shallow_model = None

        self.final_block_count = block_count
        self.logger.info(self.final_block_count)
        self.segments.append(shallow_model)

    def create_auxiliary_classifiers(self):
        """
        We insert the auxiliary classifiers after the convolutional layer.
        """

        num_classes = self.settings.n_classes
        in_channels = 0
        for i in range(len(self.segments) - 1):
            if isinstance(self.segments[i][-1], (PreBasicBlock, BasicBlock)):
                in_channels = self.segments[i][-1].conv2.out_channels
            elif isinstance(self.segments[i][-1], Bottleneck):
                in_channels = self.segments[i][-1].conv3.out_channels
            else:
                self.logger.error("Nonsupport layer type!")
                assert False, "Nonsupport layer type!"
            assert in_channels != 0, "in_channels is zero"

            self.aux_fc.append(AuxClassifier(in_channels=in_channels, num_classes=num_classes))

        final_fc = None
        if self.settings.net_type == "preresnet":
            final_fc = nn.Sequential(*[
                self.model.bn,
                self.model.relu,
                self.model.avg_pool,
                View(),
                self.model.fc])
        elif self.settings.net_type == "resnet":
            final_fc = nn.Sequential(*[
                self.model.avgpool,
                View(),
                self.model.fc])
        else:
            self.logger.error("Nonsupport net type: {}!".format(self.settings.net_type))
            assert False, "Nonsupport net type: {}!".format(self.settings.net_type)
        self.aux_fc.append(final_fc)

    def model_parallelism(self):
        self.segments = utils.data_parallel(model=self.segments, n_gpus=self.settings.n_gpus)
        self.aux_fc = utils.data_parallel(model=self.aux_fc, n_gpus=1)

    def create_optimizers(self):
        """
        Create optimizers for network segments and fcs
        """

        for i in range(len(self.segments)):
            temp_optim = []
            # add parameters in segmenets into optimizer
            # from the i-th optimizer contains [0:i] segments
            for j in range(i + 1):
                temp_optim.append({'params': self.segments[j].parameters(),
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

    def update_model(self, model, aux_fc_state=None, aux_fc_opt_state=None, seg_opt_state=None):
        """
        Update model parameter and optimizer state
        :param model: model
        :param aux_fc_state: state dict of auxiliary fully-connected layer
        :param aux_fc_opt_state: optimizer's state dict of auxiliary fully-connected layer
        :param seg_opt_state: optimizer's state dict of segment
        """

        self.segments = []
        self.aux_fc = []
        self.seg_optimizer = []
        self.fc_optimizer = []

        self.model = model
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

    @staticmethod
    def adjustweight(lr_weight=1.0):
        """
        Adjust weight according to loss
        :param lr_weight: weight of the learning rate
        """

        return 1.0 / lr_weight

    def auxnet_forward(self, images, labels=None):
        """
        Forward propagation fot auxnet
        """

        outputs = []
        temp_input = images
        losses = []
        for i in range(len(self.segments)):
            # forward
            temp_output = self.segments[i](temp_input)
            fcs_output = self.aux_fc[i](temp_output)
            outputs.append(fcs_output)
            if labels is not None:
                losses.append(self.criterion(fcs_output, labels))
            temp_input = temp_output
        return outputs, losses

    def auxnet_backward_for_loss_i(self, loss, i):
        """
        Backward propagation for the i-th loss
        :param loss: the i-th loss
        :param i: which one to perform backward propagation
        """

        self.seg_optimizer[i].zero_grad()
        self.fc_optimizer[i].zero_grad()

        # lr = lr * (pivot_weight / lr_weight)
        if i < len(self.seg_optimizer) - 1:
            for param_group in self.seg_optimizer[i].param_groups:
                # param_group['lr'] = self.lr * self.adjustweight(self.lr_weight[i].item()) * self.pivot_weight[i]
                param_group['lr'] = self.lr * self.adjustweight(self.lr_weight[i].item())

            loss.backward(retain_graph=True)
            for param_group in self.seg_optimizer[i].param_groups:
                for p in param_group['params']:
                    if p.grad is None:
                        continue
                    p.grad.data.mul_(p.new([self.pivot_weight[i]]))
        else:
            loss.backward(retain_graph=True)

        self.fc_optimizer[i].step()
        self.seg_optimizer[i].step()

    def update_lr(self, epoch):
        """
        Update learning rate of optimizers
        :param epoch: index of epoch
        """

        if hasattr(self.settings, 'warmup_n_epochs'):
            if epoch < self.settings.warmup_n_epochs:
                lr = self.settings.warmup_lr + (self.settings.lr / self.settings.warmup_n_epochs) * epoch
                self.lr = lr
            else:
                gamma = 0
                for step in self.settings.step:
                    if epoch + 1.0 > int(step):
                        gamma += 1
                lr = self.settings.lr * math.pow(0.1, gamma)
                self.lr = lr
        else:
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

    def train(self, epoch):
        """
        Train one epoch for auxnet
        :param epoch: index of epoch
        """

        iters = len(self.train_loader)
        self.update_lr(epoch)

        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.segments)
        for i in range(num_segments):
            self.segments[i].train()
            self.aux_fc[i].train()
            top1_error.append(utils.AverageMeter())
            top5_error.append(utils.AverageMeter())
            top1_loss.append(utils.AverageMeter())

        start_time = time.time()
        end_time = start_time
        for i, (images, labels) in enumerate(self.train_loader):
            start_time = time.time()
            data_time = start_time - end_time

            if self.settings.n_gpus == 1:
                images = images.cuda()
            labels = labels.cuda()

            # forward
            outputs, losses = self.auxnet_forward(images, labels)
            # backward
            for j in range(len(self.seg_optimizer)):
                self.auxnet_backward_for_loss_i(losses[j], j)

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

            if i % self.settings.print_frequency == 0:
                utils.print_result(epoch, self.settings.n_epochs, i + 1,
                                   iters, self.lr, data_time, iter_time,
                                   single_error,
                                   single_loss,
                                   mode="Train",
                                   logger=self.logger)

        top1_error_list, top1_loss_list, top5_error_list = self._convert_results(
            top1_error=top1_error, top1_loss=top1_loss, top5_error=top5_error)
        if self.logger is not None:
            for i in range(num_segments):
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_top1_error_{:d}".format(i), top1_error[i].avg,
                    self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_top5_error_{:d}".format(i), top5_error[i].avg,
                    self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_loss_{:d}".format(i), top1_loss[i].avg, self.run_count)
            self.tensorboard_logger.scalar_summary("lr", self.lr, self.run_count)

        self.logger.info("|===>Training Error: {:4f}/{:4f}, Loss: {:4f}".format(
            top1_error[-1].avg, top5_error[-1].avg, top1_loss[-1].avg))
        return top1_error_list, top1_loss_list, top5_error_list

    def val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """

        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.segments)
        for i in range(num_segments):
            self.segments[i].eval()
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

                if i % self.settings.print_frequency == 0:
                    utils.print_result(epoch, self.settings.n_epochs, i + 1,
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
                    "auxnet_val_top1_error_{:d}".format(i), top1_error[i].avg, self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_val_top5_error_{:d}".format(i), top5_error[i].avg, self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_val_loss_{:d}".format(i), top1_loss[i].avg, self.run_count)
        self.run_count += 1

        self.logger.info("|===>Validation Error: {:4f}/{:4f}, Loss: {:4f}".format(
            top1_error[-1].avg, top5_error[-1].avg, top1_loss[-1].avg))
        return top1_error_list, top1_loss_list, top5_error_list
