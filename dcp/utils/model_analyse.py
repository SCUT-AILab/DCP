import numpy as np
import torch.nn as nn
from prettytable import PrettyTable

from . import model_transform as mt

__all__ = ["ModelAnalyse"]


class ModelAnalyse(object):
    def __init__(self, model, logger):
        self.model = mt.list2sequential(model)
        self.logger = logger
        self.flops = []
        self.madds = []
        self.weight_shapes = []
        self.bias_shapes = []
        self.output_shapes = []

    def params_count(self):
        params_num_list = []

        output = PrettyTable()
        output.field_names = ["Param name", "Shape", "Dim"]

        self.logger.info("------------------------number of parameters------------------------\n")
        for name, param in self.model.named_parameters():
            if 'pruned_weight' in name:
                continue
            param_num = param.numel()
            param_shape = [shape for shape in param.shape]
            params_num_list.append(param_num)
            output.add_row([name, param_shape, param_num])
        self.logger.info(output)

        params_num_list = np.array(params_num_list)
        params_num = params_num_list.sum()
        self.logger.info("|===>Number of parameters is: {}".format(params_num))
        return params_num

    def zero_count(self):
        weights_zero_list = []

        output = PrettyTable()
        output.field_names = ["Param name", "Zero Num"]

        self.logger.info("------------------------number of zeros in parameters------------------------\n")
        for name, param in self.model.named_parameters():
            weight_zero = param.data.eq(0).sum().item()
            weights_zero_list.append(weight_zero)
            output.add_row([name, weight_zero])
        self.logger.info(output)

        weights_zero_list = np.array(weights_zero_list)
        zero_num = weights_zero_list.sum()
        self.logger.info("|===>Number of zeros is: {}".format(zero_num))
        return zero_num

    def _flops_conv_hook(self, layer, x, out):
        # compute number of multiply-add
        if layer.bias is not None:
            layer_flops = out.size(2) * out.size(3) * \
                          (2. * layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)) \
                          * layer.weight.size(0)
        else:
            layer_flops = out.size(2) * out.size(3) * \
                          (2. * layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3) - 1.) \
                          * layer.weight.size(0)
        self.flops.append(layer_flops)
        # if we only care about multipy operation, use following equation instead
        """
        layer_flops = out.size(2)*out.size(3)*layer.weight.size(1)*layer.weight.size(2)*layer.weight.size(0)
        """

    def _flops_linear_hook(self, layer, x, out):
        # compute number of flops
        if layer.bias is not None:
            layer_flops = (2 * layer.weight.size(1)) * layer.weight.size(0)
        else:
            layer_flops = (2 * layer.weight.size(1) - 1) * layer.weight.size(0)
        # if we only care about multipy operation, use following equation instead
        """
        layer_flops = layer.weight.size(1)*layer.weight.size(0)
        """
        self.flops.append(layer_flops)

    def _madds_conv_hook(self, layer, x, out):
        input = x[0]
        batch_size = input.shape[0]
        output_height, output_width = out.shape[2:]

        kernel_height, kernel_width = layer.kernel_size
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        groups = layer.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

        active_elements_count = batch_size * output_height * output_width

        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0
        if layer.bias is not None:
            bias_flops = out_channels * active_elements_count

        overall_flops = overall_conv_flops + bias_flops
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(out.shape))
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.madds.append(overall_flops)

    def _madds_linear_hook(self, layer, x, out):
        # compute number of multiply-add
        # layer_madds = layer.weight.size(0) * layer.weight.size(1)
        # if layer.bias is not None:
        #     layer_madds += layer.weight.size(0)
        input = x[0]
        batch_size = input.shape[0]
        overall_flops = int(batch_size * input.shape[1] * out.shape[1])

        bias_flops = 0
        if layer.bias is not None:
            bias_flops = out.shape[1]
        overall_flops = overall_flops + bias_flops
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(out.shape))
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.madds.append(overall_flops)

    def madds_compute(self, x):
        """
        Compute number of multiply-adds of the model
        """

        hook_list = []
        self.madds = []
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._madds_conv_hook))
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._madds_linear_hook))
        # run forward for computing FLOPs
        self.model.eval()
        self.model(x)

        madds_np = np.array(self.madds)
        madds_sum = float(madds_np.sum())
        percentage = madds_np / madds_sum

        output = PrettyTable()
        output.field_names = ["Layer", "Madds", "Weight Shape", "Bias Shape", "Output Shape", "Percentage"]

        self.logger.info("------------------------FLOPs------------------------\n")
        for i in range(len(self.madds)):
            output.add_row([i, madds_np[i], self.weight_shapes[i],
                            self.bias_shapes[i], self.output_shapes[i], percentage[i]])
        self.logger.info(output)
        repo_str = "|===>Total MAdds: {:e}".format(madds_sum)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()

        return madds_np

    def flops_compute(self, x):
        """
        Compute number of flops of the model
        """

        hook_list = []
        self.flops = []
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._flops_conv_hook))
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._flops_linear_hook))

        # run forward for computing FLOPs
        self.model.eval()
        self.model(x)

        flops_np = np.array(self.flops)
        flops_sum = float(flops_np.sum())
        percentage = flops_np / flops_sum
        for i in range(len(self.flops)):
            repo_str = "|===>FLOPs of layer [{:d}]: {:e}, {:f}".format(i, flops_np[i], percentage[i])
            self.logger.info(repo_str)
        repo_str = "### Total FLOPs: {:e}".format(flops_sum)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()

        return flops_np
