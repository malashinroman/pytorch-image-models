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
from dycs.models.Dycs import DycsNet


def get_net_files(checkpoint_path):
    # FIXME: fix for arbitrary number of subdirectories
    # list all files with .npy extention in checkpoint_path, uses glob
    # return glob.glob(checkpoint_path + '/*best*.pth.tar', recursive=True)
    return glob.glob(os.path.join(checkpoint_path, '*/*/*model_best*'), recursive=True)


def create_model_dycs(
    args: Any,
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

    name_parts = model_name.split('dycs_')
    if len(name_parts) == 2:
        timm_model_name = name_parts[1]
        dycs_model_name = name_parts[0][:-1]
    else:
        raise ValueError('Invalid model name')

    # timm_model_name = model_name.replace('dycs_', '')
    pretrained = False
    network_files = get_net_files(checkpoint_path)
    if len(network_files) == 0:
        raise ValueError('No networks found in checkpoint path')
    if len(network_files) not in [5,10]:
        raise ValueError('Not enough networks found in checkpoint path')

    network_files = sorted(
        network_files, key=lambda x: int(x.split('class_map')[1][0]))
    nets = []

    # load all networks with torch
    for file in network_files:
        net = create_model(timm_model_name, pretrained, pretrained_cfg,
                           pretrained_cfg_overlay, file, scriptable,
                           exportable, no_jit, **kwargs)
        nets.append(net)

    masternet = create_model(timm_model_name, True, pretrained_cfg,
                       pretrained_cfg_overlay, file, scriptable,
                       exportable, no_jit, **kwargs)


    return DycsNet(args, nets, master_net=masternet)
