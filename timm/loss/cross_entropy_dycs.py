""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from yaml import warnings


def convert2raw_classification(x: torch.Tensor, target: torch.Tensor, args: Namespace):
    """Convert classification into raw classification for dycs
    """
    new_x = torch.zeros(size=(x.shape[0], 5), dtype=x.type())
    group_size = args.dycs_classes_per_group
    new_target = (target / group_size).int()
    """
    0-199:   0
    200-399: 1
    400-599: 2
    600-799: 3
    800-999: 4
    """
    n_superclasses = 1000 // group_size

    if args.dycs_fine2raw == 'max':
        for i in range(n_superclasses):
            new_x[:,i] = x[:,i*group_size:(i+1)*group_size].max(dim=1)
    elif args.dycs_fine2raw == 'sum':
        for i in range(n_superclasses):
            new_x[:,i] = x[:,i*group_size:(i+1)*group_size].sum(dim=1)
    elif args.dycs_fine2raw == 'mean':
        for i in range(n_superclasses):
            new_x[:,i] = x[:,i*group_size:(i+1)*group_size].mean(dim=1)
    else:
        raise Exception(f'unknonwn args.fin2raw: {args.dycs_fine2raw}')
    
    return new_x, new_target






class LabelSmoothingCrossEntropyRaw(nn.Module):
    """ NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1, args=None):
        super(LabelSmoothingCrossEntropyRaw, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.args = args

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x,target = convert2raw_classification(x, target, self.args)

        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
