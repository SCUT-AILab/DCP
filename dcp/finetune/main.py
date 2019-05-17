import argparse

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from option import Option
from dcp.pruning import ResModelPrune
from torch import nn

from dcp.aux_trainer import AuxTrainer
from dcp.aux_checkpoint import AuxCheckPoint
from dcp.dataloader import *
from dcp.mask_conv import MaskConv2d
from dcp.model_builder import get_model
from dcp.models.preresnet import PreBasicBlock
from dcp.models.resnet import BasicBlock, Bottleneck
from dcp.utils.logger import get_logger
from dcp.utils.model_analyse import ModelAnalyse
from dcp.utils.tensorboard_logger import TensorboardLogger
from dcp.utils.write_log import write_log, write_settings
from dcp.utils import cal_pivot


class Experiment(object):
    """
    Run experiments with pre-defined pipeline
    """

    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)
        self.checkpoint = None
        self.train_loader = None
        self.val_loader = None
        self.pruned_model = None
        self.network_wise_trainer = None
        self.optimizer_state = None
        self.aux_fc_state = None

        os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.gpu

        self.settings.set_save_path()
        write_settings(self.settings)
        self.logger = get_logger(self.settings.save_path, "finetune")
        self.tensorboard_logger = TensorboardLogger(self.settings.save_path)
        self.logger.info("|===>Result will be saved at {}".format(self.settings.save_path))
        self.epoch = 0
        self.test_input = None

        self.prepare()

    def write_settings(self):
        """
        Save expriment settings to a file
        """

        with open(os.path.join(self.settings.save_path, "settings.log"), "w") as f:
            for k, v in self.settings.__dict__.items():
                f.write(str(k) + ": " + str(v) + "\n")

    def prepare(self):
        """
        Preparing experiments
        """

        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._cal_pivot()
        self._set_checkpoint()
        self._set_trainier()

    def _set_gpu(self):
        """
        Initialize the seed of random number generator
        """

        # set torch seed
        # init random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(0)
        cudnn.benchmark = True

    def _set_dataloader(self):
        """
        Create train loader and validation loader for channel pruning
        """

        if 'cifar' in self.settings.dataset:
            self.train_loader, self.val_loader = get_cifar_dataloader(self.settings.dataset,
                                                                      self.settings.batch_size,
                                                                      self.settings.n_threads,
                                                                      self.settings.data_path,
                                                                      self.logger)
        elif self.settings.dataset == 'imagenet':
            self.train_loader, self.val_loader = get_imagenet_dataloader(self.settings.dataset,
                                                                         self.settings.batch_size,
                                                                         self.settings.n_threads,
                                                                         self.settings.data_path,
                                                                         self.logger)

    def replace_layer_with_mask_conv_resnet(self):
        """
        Replace the conv layer in resnet with mask_conv for ResNet
        """

        for module in self.pruned_model.modules():
            if isinstance(module, (PreBasicBlock, BasicBlock, Bottleneck)):
                # replace conv2
                temp_conv = MaskConv2d(
                    in_channels=module.conv2.in_channels,
                    out_channels=module.conv2.out_channels,
                    kernel_size=module.conv2.kernel_size,
                    stride=module.conv2.stride,
                    padding=module.conv2.padding,
                    bias=(module.conv2.bias is not None))

                temp_conv.weight.data.copy_(module.conv2.weight.data)
                if module.conv2.bias is not None:
                    temp_conv.bias.data.copy_(module.conv2.bias.data)
                module.conv2 = temp_conv

                if isinstance(module, Bottleneck):
                    # replace conv3
                    temp_conv = MaskConv2d(
                        in_channels=module.conv3.in_channels,
                        out_channels=module.conv3.out_channels,
                        kernel_size=module.conv3.kernel_size,
                        stride=module.conv3.stride,
                        padding=module.conv3.padding,
                        bias=(module.conv3.bias is not None))

                    temp_conv.weight.data.copy_(module.conv3.weight.data)
                    if module.conv3.bias is not None:
                        temp_conv.bias.data.copy_(module.conv3.bias.data)
                    module.conv3 = temp_conv

    def replace_layer_with_mask_conv(self):
        """
        Replace the conv layer in resnet with mask_conv
        """

        if self.settings.net_type in ["preresnet", "resnet"]:
            self.replace_layer_with_mask_conv_resnet()

    def _set_model(self):
        """
        Get model
        """

        self.pruned_model, self.test_input = get_model(self.settings.dataset,
                                                       self.settings.net_type,
                                                       self.settings.depth,
                                                       self.settings.n_classes)
        self.replace_layer_with_mask_conv()

    def _set_checkpoint(self):
        """
        Load pre-trained model or resume checkpoint
        """

        assert self.pruned_model is not None, "please create model first"

        self.checkpoint = AuxCheckPoint(self.settings.save_path, self.logger)
        self._load_pretrained()
        # self._load_resume()

    def _load_pretrained(self):
        """
        Load pre-trained model
        """

        if self.settings.pretrained is not None:
            check_point_params = torch.load(self.settings.pretrained)
            model_state = check_point_params["pruned_model"]
            self.aux_fc_state = check_point_params["aux_fc"]
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, model_state)
            self.logger.info("|===>load restrain file: {}".format(self.settings.pretrained))

    def _load_resume(self):
        """
        Load resume checkpoint
        """

        if self.settings.resume is not None:
            check_point_params = torch.load(self.settings.resume)
            pruned_model_state = check_point_params["model"]
            # self.optimizer_state = check_point_params["optimizer"]
            self.epoch = check_point_params["epoch"]
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, pruned_model_state)
            self.logger.info("|===>load resume file: {}".format(self.settings.resume))
            self.network_wise_trainer.run_count = self.epoch

    def _set_trainier(self):
        """
        Initialize network-wise trainer
        """

        self.network_wise_trainer = AuxTrainer(model=self.pruned_model,
                                               train_loader=self.train_loader,
                                               val_loader=self.val_loader,
                                               settings=self.settings,
                                               logger=self.logger,
                                               tensorboard_logger=self.tensorboard_logger,
                                               run_count=self.epoch)
        if self.aux_fc_state is not None:
            self.network_wise_trainer.update_aux_fc(aux_fc_state=self.aux_fc_state)

    def _cal_pivot(self):
        """
        Calculate the block index for additional loss
        """

        self.num_segments, self.settings.pivot_set = cal_pivot(self.settings.n_losses, self.settings.net_type,
                                                               self.settings.depth, self.logger)

    def pruning(self):
        """
        Prune channels
        """
        self.logger.info("Before pruning:")
        self.logger.info(self.pruned_model)
        # self.network_wise_trainer.val(0)
        model_analyse = ModelAnalyse(self.pruned_model, self.logger)
        params_num = model_analyse.params_count()
        zero_num = model_analyse.zero_count()
        zero_rate = zero_num * 1.0 / params_num
        self.logger.info("zero rate is: {}".format(zero_rate))
        model_analyse.madds_compute(self.test_input)

        if self.settings.net_type in ["preresnet", "resnet"]:
            model_prune = ResModelPrune(model=self.pruned_model,
                                        net_type=self.settings.net_type,
                                        depth=self.settings.depth)
        else:
            assert False, "unsupport net_type: {}".format(self.settings.net_type)

        model_prune.run()
        self.network_wise_trainer.update_model(model_prune.model, self.optimizer_state)

        self.logger.info("After pruning:")
        self.logger.info(self.pruned_model)
        # self.network_wise_trainer.val(0)
        model_analyse = ModelAnalyse(self.pruned_model, self.logger)
        params_num = model_analyse.params_count()
        zero_num = model_analyse.zero_count()
        zero_rate = zero_num * 1.0 / params_num
        self.logger.info("zero rate is: {}".format(zero_rate))
        model_analyse.madds_compute(self.test_input)

    def fine_tuning(self):
        """
        Conduct network-wise fine-tuning after channel selection
        """

        best_top1 = 100
        best_top5 = 100

        start_epoch = 0
        if self.epoch != 0:
            start_epoch = self.epoch + 1
            self.epoch = 0

        for epoch in range(start_epoch, self.settings.n_epochs):
            train_error, train_loss, train5_error = self.network_wise_trainer.train(epoch)
            val_error, val_loss, val5_error = self.network_wise_trainer.val(epoch)

            # write log
            log_str = "{:d}\t".format(epoch)
            for i in range(len(train_error)):
                log_str += "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(
                    train_error[i], train_loss[i], val_error[i],
                    val_loss[i], train5_error[i], val5_error[i])
            write_log(self.settings.save_path, 'log.txt', log_str)

            # save model and checkpoint
            best_flag = False
            if best_top1 >= val_error[-1]:
                best_top1 = val_error[-1]
                best_top5 = val5_error[-1]
                best_flag = True

            if best_flag:
                self.checkpoint.save_aux_model(self.network_wise_trainer.model, self.network_wise_trainer.aux_fc)

            self.logger.info("|===>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}\n".format(best_top1, best_top5))
            self.logger.info("|==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}\n".format(100 - best_top1,
                                                                                                     100 - best_top5))

            if "imagenet" in self.settings.dataset:
                self.checkpoint.save_aux_checkpoint(self.network_wise_trainer.model,
                                                    self.network_wise_trainer.seg_optimizer,
                                                    self.network_wise_trainer.fc_optimizer,
                                                    self.network_wise_trainer.aux_fc, epoch, epoch + 1)
            else:
                self.checkpoint.save_aux_checkpoint(self.network_wise_trainer.model,
                                                    self.network_wise_trainer.seg_optimizer,
                                                    self.network_wise_trainer.fc_optimizer,
                                                    self.network_wise_trainer.aux_fc, epoch)


def main():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='input batch size for training (default: 64)')
    parser.add_argument('id', type=int, metavar='experiment_id',
                        help='Experiment ID')
    args = parser.parse_args()

    option = Option(args.conf_path)
    option.manualSeed = args.id + 1
    option.experiment_id = option.experiment_id + "{:0>2d}".format(args.id + 1)

    experiment = Experiment(option)
    experiment.pruning()
    experiment._load_resume()
    experiment.fine_tuning()


if __name__ == '__main__':
    main()
