import argparse

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from option import Option
from dcp.pruning import ResModelPrune
from torch import nn

from dcp.trainer import Trainer
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
from dcp.models.pruned_preresnet import PrunedPreResNet
from dcp.models.pruned_resnet import PrunedResNet
from dcp.models.pruned_vgg import PrunedVGGCIFAR


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

    def _set_model(self):
        """
        Get model
        """

        if self.settings.dataset in ["cifar10", "cifar100"]:
            if self.settings.net_type == "preresnet":
                self.pruned_model = PrunedPreResNet(depth=self.settings.depth,
                                                    pruning_rate=self.settings.pruning_rate,
                                                    num_classes=self.settings.n_classes)
            elif self.settings.net_type == "vgg":
                self.pruned_model = PrunedVGGCIFAR(pruning_rate=self.settings.pruning_rate,
                                                   num_classes=self.settings.n_classes)
            else:
                assert False, "use {} data while network is {}".format(self.settings.dataset, self.settings.net_type)

        elif self.settings.dataset in ["imagenet", "imagenet_mio"]:
            if self.settings.net_type == "resnet":
                self.pruned_model = PrunedResNet(
                    depth=self.settings.depth,
                    pruning_rate=self.settings.pruning_rate,
                    num_classes=self.settings.n_classes)
            else:
                assert False, "use {} data while network is {}".format(self.settings.dataset, self.settings.net_type)

        else:
            assert False, "unsupported data set: {}".format(self.settings.dataset)

    def _set_checkpoint(self):
        """
        Load pre-trained model or resume checkpoint
        """

        assert self.pruned_model is not None, "please create model first"

        self.checkpoint = AuxCheckPoint(self.settings.save_path, self.logger)
        self._load_pretrained()

    def _load_pretrained(self):
        """
        Load pre-trained model
        """

        if self.settings.pretrained is not None:
            check_point_params = torch.load(self.settings.pretrained)
            model_state = check_point_params
            # self.aux_fc_state = check_point_params["aux_fc"]
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, model_state)
            self.logger.info("|===>load restrain file: {}".format(self.settings.pretrained))

    def _set_trainier(self):
        """
        Initialize network-wise trainer
        """

        self.trainer = Trainer(model=self.pruned_model,
                               train_loader=self.train_loader,
                               val_loader=self.val_loader,
                               settings=self.settings,
                               logger=self.logger,
                               tensorboard_logger=self.tensorboard_logger,
                               run_count=self.epoch)

    def evaluation(self):
        """
        Conduct network-wise fine-tuning after channel selection
        """

        self.trainer.val(0)


def main():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='input batch size for training (default: 64)')
    args = parser.parse_args()

    option = Option(args.conf_path)

    experiment = Experiment(option)
    experiment.evaluation()


if __name__ == '__main__':
    main()
