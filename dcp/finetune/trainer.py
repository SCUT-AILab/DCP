import math
import time

import torch.autograd
import torch.nn as nn

import dcp.utils as utils


class NetworkWiseTrainer(object):
    """
        Network-wise trainer for fine tuning after channel selection
    """

    def __init__(self, pruned_model, train_loader, val_loader, settings, logger, tensorboard_logger, run_count=0):
        self.pruned_model = utils.data_parallel(pruned_model, settings.n_gpus)
        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = torch.optim.SGD(
            params=self.pruned_model.parameters(),
            lr=self.settings.lr,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weight_decay,
            nesterov=True)

        self.run_count = run_count
        self.lr = self.settings.lr
        self.scalar_info = {}

    def update_model(self, pruned_model, optimizer_state=None):
        """
        Update pruned model parameter
        :param pruned_model: pruned model
        """

        self.optimizer = None
        self.pruned_model = utils.data_parallel(pruned_model, self.settings.n_gpus)
        self.optimizer = torch.optim.SGD(
            params=self.pruned_model.parameters(),
            lr=self.settings.lr,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weight_decay,
            nesterov=True)
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

    def update_lr(self, epoch):
        """
        Update learning rate of optimizers
        :param epoch: current training epoch
        """
        gamma = 0
        for step in self.settings.step:
            if epoch + 1.0 > int(step):
                gamma += 1
        lr = self.settings.lr * math.pow(0.1, gamma)
        self.lr = lr
        # update learning rate of model optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def forward(self, images, labels=None):
        """
        Forward propagation
        """

        # forward and backward and optimize
        output = self.pruned_model(images)

        if labels is not None:
            loss = self.criterion(output, labels)
            return output, loss
        else:
            return output, None

    def backward(self, loss):
        """
        Backward propagation
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, epoch):
        """
        Training
        """

        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        iters = len(self.train_loader)
        self.update_lr(epoch)
        # switch to train mode
        self.pruned_model.train()

        start_time = time.time()
        end_time = start_time

        for i, (images, labels) in enumerate(self.train_loader):
            start_time = time.time()
            data_time = start_time - end_time

            if self.settings.n_gpus == 1:
                images = images.cuda()
            labels = labels.cuda()

            output, loss = self.forward(images, labels)
            self.backward(loss)

            single_error, single_loss, single5_error = utils.compute_singlecrop_error(
                outputs=output, labels=labels,
                loss=loss, top5_flag=True)

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))

            end_time = time.time()
            iter_time = end_time - start_time

            if i % self.settings.print_frequency == 0:
                utils.print_result(
                    epoch, self.settings.n_epochs, i + 1,
                    iters, self.lr, data_time, iter_time,
                    single_error,
                    single_loss, top5error=single5_error,
                    mode="Train",
                    logger=self.logger)

        self.scalar_info['network_wise_fine_tune_train_top1_error'] = top1_error.avg
        self.scalar_info['network_wise_fine_tune_train_top5_error'] = top5_error.avg
        self.scalar_info['network_wise_fine_tune_train_loss'] = top1_loss.avg
        self.scalar_info['network_wise_fine_tune_lr'] = self.lr

        if self.tensorboard_logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        self.logger.info(
            "|===>Training Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(top1_error.avg, top1_loss.avg,
                                                                                  top5_error.avg))
        return top1_error.avg, top1_loss.avg, top5_error.avg

    def val(self, epoch):
        """
        Validation
        """

        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.pruned_model.eval()

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

                single_error, single_loss, single5_error = utils.compute_singlecrop_error(
                    outputs=output, loss=loss,
                    labels=labels, top5_flag=True)

                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

                if i % self.settings.print_frequency == 0:
                    utils.print_result(
                        epoch, self.settings.n_epochs, i + 1,
                        iters, self.lr, data_time, iter_time,
                        single_error, single_loss,
                        top5error=single5_error,
                        mode="Validation",
                        logger=self.logger)

        self.scalar_info['network_wise_fine_tune_val_top1_error'] = top1_error.avg
        self.scalar_info['network_wise_fine_tune_val_top5_error'] = top5_error.avg
        self.scalar_info['network_wise_fine_tune_val_loss'] = top1_loss.avg
        if self.tensorboard_logger is not None:
            for tag, value in self.scalar_info.items():
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}
        self.run_count += 1
        self.logger.info(
            "|===>Validation Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(top1_error.avg, top1_loss.avg,
                                                                                    top5_error.avg))
        return top1_error.avg, top1_loss.avg, top5_error.avg
