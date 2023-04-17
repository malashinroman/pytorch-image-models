""" extra functions for dynamically configurable systems
    modular net support
"""
import numpy as np
import glob
from typing import Any, Dict, Optional, Union
from timm.models._pretrained import PretrainedCfg
from timm.models._factory import create_model
import torch
import os

def get_net_files(checkpoint_path):
    # FIXME: fix for arbitrary number of subdirectories
    # list all files with .npy extention in checkpoint_path, uses glob
    # return glob.glob(checkpoint_path + '/*best*.pth.tar', recursive=True)
    return glob.glob(os.path.join(checkpoint_path, '*/*/*model_best*'), recursive=True)


class DycsNet(torch.nn.Module):
    """ modular net for dycs
    """
    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.num_classes = 1000
        for i in range(len(nets)):
            self.nets[i] = self.nets[i].cuda()
            # self.nets[i] = self.nets[i].to('cuda:0')

    def forward(self, x):
        # forward pass through all networks
        ys = []
        # x = x.to('cuda:0')
        for net in self.nets:
            y = net(x)
            ys.append(y)

        # concatenate all outputs
        # take only 100 first elements of output vector for each networks
        ys = [y[:, :100] for y in ys]
        y = torch.cat(ys, dim=1)
        return y


def create_model_dycs(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg: Optional[Union[str,
                                   Dict[str, Any], PretrainedCfg]] = None,
    pretrained_cfg_overlay:  Optional[Dict[str, Any]] = None,
    checkpoint_path: str = '',
    scriptable: Optional[bool] = None,
    exportable: Optional[bool] = None,
    no_jit: Optional[bool] = None,
    **kwargs,
):
    if not 'dycs' in model_name:
        # if checkpoint_path is directory than do not use it
        if checkpoint_path.endswith('/'):
            checkpoint_path = ''
        return create_model(model_name, pretrained,
                            pretrained_cfg, pretrained_cfg_overlay,
                            checkpoint_path, scriptable, exportable,
                            no_jit, **kwargs)

    timm_model_name = model_name.replace('dycs_', '')
    pretrained = False
    network_files = get_net_files(checkpoint_path)
    if len(network_files) == 0:
        raise ValueError('No networks found in checkpoint path')
    if len(network_files) < 10:
        raise ValueError('Not enough networks found in checkpoint path')

    network_files=sorted(network_files, key=lambda x: int(x.split('class_map')[1][0]))
    nets = []

    # load all networks with torch
    for file in network_files:
        net = create_model(timm_model_name, pretrained, pretrained_cfg,
                           pretrained_cfg_overlay, file, scriptable,
                           exportable, no_jit, **kwargs)
        nets.append(net)

    return DycsNet(nets)
