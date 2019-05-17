import time

import math
import torch.autograd
import torch.nn as nn

import dcp.utils as utils


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


class Trainer(object):
    """
    Trainer for auxnet
    """

    def __init__(self, model, train_loader, val_loader, settings, logger,
                 tensorboard_logger, optimizer_state=None, run_count=0):
        self.settings = settings

        self.model = utils.data_parallel(model=model, n_gpus=self.settings.n_gpus)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.lr = self.settings.lr
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.settings.lr,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weight_decay,
            nesterov=True)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.run_count = run_count

    def forward(self, images, labels=None):
        """
        forward propagation
        """
        # forward and backward and optimize
        output = self.model(images)

        if labels is not None:
            loss = self.criterion(output, labels)
            return output, loss
        else:
            return output, None

    def backward(self, loss):
        """
        backward propagation
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, epoch):
        """
        Train one epoch for auxnet
        :param epoch: index of epoch
        """

        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        iters = len(self.train_loader)
        self.update_lr(epoch)
        # Switch to train mode
        self.model.train()

        start_time = time.time()
        end_time = start_time

        for i, (images, labels) in enumerate(self.train_loader):
            start_time = time.time()
            data_time = start_time - end_time

            if self.settings.n_gpus == 1:
                images = images.cuda()
            labels = labels.cuda()

            # forward
            output, loss = self.forward(images, labels)
            self.backward(loss)

            # compute loss and error rate
            single_error, single_loss, single5_error = utils.compute_singlecrop_error(
                outputs=output, labels=labels,
                loss=loss, top5_flag=True)

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))

            end_time = time.time()
            iter_time = end_time - start_time

            if i % self.settings.print_frequency == 0:
                utils.print_result(epoch, self.settings.n_epochs, i + 1,
                                   iters, self.lr, data_time, iter_time,
                                   single_error,
                                   single_loss,
                                   mode="Train",
                                   logger=self.logger)

        if self.tensorboard_logger is not None:
            self.tensorboard_logger.scalar_summary('train_top1_error', top1_error.avg, self.run_count)
            self.tensorboard_logger.scalar_summary('train_top5_error', top5_error.avg, self.run_count)
            self.tensorboard_logger.scalar_summary('train_loss', top1_loss.avg, self.run_count)
            self.tensorboard_logger.scalar_summary("lr", self.lr, self.run_count)

        self.logger.info("|===>Training Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}"
                         .format(top1_error.avg, top1_loss.avg, top5_error.avg))
        return top1_error.avg, top1_loss.avg, top5_error.avg

    def val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """

        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model.eval()

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

                output, loss = self.forward(images, labels)

                # compute loss and error rate
                single_error, single_loss, single5_error = utils.compute_singlecrop_error(
                    outputs=output, labels=labels,
                    loss=loss, top5_flag=True)

                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

                if i % self.settings.print_frequency == 0:
                    utils.print_result(epoch, self.settings.n_epochs, i + 1,
                                       iters, self.lr, data_time, iter_time,
                                       single_error,
                                       single_loss,
                                       mode="Validation",
                                       logger=self.logger)

        if self.tensorboard_logger is not None:
            self.tensorboard_logger.scalar_summary("val_top1_error", top1_error.avg, self.run_count)
            self.tensorboard_logger.scalar_summary("val_top5_error", top5_error.avg, self.run_count)
            self.tensorboard_logger.scalar_summary("val_loss", top1_loss.avg, self.run_count)

        self.run_count += 1
        self.logger.info("|===>Testing Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}"
                         .format(top1_error.avg, top1_loss.avg, top5_error.avg))
        return top1_error.avg, top1_loss.avg, top5_error.avg
