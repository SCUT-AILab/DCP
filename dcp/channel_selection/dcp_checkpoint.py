import os

import torch
import torch.nn as nn

import dcp.utils as utils
from dcp.checkpoint import CheckPoint


class DCPCheckPoint(CheckPoint):
    """
    Save model state to file
    check_point_params: original_model, pruned_model, aux_fc, current_pivot, index, block_count
    """

    def __init__(self, save_path, logger):
        super(DCPCheckPoint, self).__init__(save_path, logger)

    def save_dcp_model(self, original_model, pruned_model, aux_fc=None, current_pivot=None, index=0, block_count=0):
        check_point_params = {}

        original_model = utils.list2sequential(original_model)
        if isinstance(original_model, nn.DataParallel):
            check_point_params["original_model"] = original_model.module.state_dict()
        else:
            check_point_params["original_model"] = original_model.state_dict()

        pruned_model = utils.list2sequential(pruned_model)
        if isinstance(pruned_model, nn.DataParallel):
            check_point_params["pruned_model"] = pruned_model.module.state_dict()
        else:
            check_point_params["pruned_model"] = pruned_model.state_dict()

        aux_fc_state = []
        if aux_fc:
            for i in range(len(aux_fc)):
                if isinstance(aux_fc[i], nn.DataParallel):
                    aux_fc_state.append(aux_fc[i].module.state_dict())
                else:
                    aux_fc_state.append(aux_fc[i].state_dict())

        check_point_params["aux_fc"] = aux_fc_state
        check_point_params["current_pivot"] = current_pivot
        check_point_params["segment_num"] = index
        check_point_params["block_num"] = block_count
        model_save_name = "model_{:0>3d}_cs_{:0>3d}.pth".format(index, block_count)
        torch.save(check_point_params, os.path.join(self.save_path, model_save_name))

    def save_dcp_checkpoint(self, original_model, pruned_model, aux_fc=None, aux_fc_opt=None, seg_opt=None,
                            current_pivot=None, index=0, block_count=0):
        # save state of the network
        check_point_params = {}

        original_model = utils.list2sequential(original_model)
        if isinstance(original_model, nn.DataParallel):
            check_point_params["original_model"] = original_model.module.state_dict()
        else:
            check_point_params["original_model"] = original_model.state_dict()

        pruned_model = utils.list2sequential(pruned_model)
        if isinstance(pruned_model, nn.DataParallel):
            check_point_params["pruned_model"] = pruned_model.module.state_dict()
        else:
            check_point_params["pruned_model"] = pruned_model.state_dict()

        aux_fc_state = []
        aux_fc_opt_state = []
        seg_opt_state = []
        if aux_fc:
            for i in range(len(aux_fc)):
                if isinstance(aux_fc[i], nn.DataParallel):
                    temp_state = aux_fc[i].module.state_dict()
                else:
                    temp_state = aux_fc[i].state_dict()
                aux_fc_state.append(temp_state)
                if aux_fc_opt:
                    aux_fc_opt_state.append(aux_fc_opt[i].state_dict())
                if seg_opt:
                    seg_opt_state.append(seg_opt[i].state_dict())

        check_point_params["aux_fc"] = aux_fc_state
        check_point_params["aux_fc_opt"] = aux_fc_opt_state
        check_point_params["seg_opt"] = seg_opt_state
        check_point_params["current_pivot"] = current_pivot
        check_point_params["segment_num"] = index
        check_point_params["block_num"] = block_count
        checkpoint_save_name = "checkpoint_{:0>3d}_cs_{:0>3d}.pth".format(index, block_count)
        torch.save(check_point_params, os.path.join(self.save_path, checkpoint_save_name))
