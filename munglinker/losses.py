"""This file defines the pytorch fit() wrapper.
"""
from __future__ import print_function, unicode_literals, division

import torch
from torch.nn.modules.loss import _WeightedLoss

# This one is a really external dependency

torch.set_default_tensor_type('torch.FloatTensor')


class FocalLossElementwise(_WeightedLoss):
    """Elementwise Focal Loss implementation for arbitrary tensors that computes
    the loss element-wise, e.g. for semantic segmentation. Alpha-balancing
    is implemented for entire minibatches instead of per-channel.

    See: https://arxiv.org/pdf/1708.02002.pdf  -- RetinaNet
    """
    def __init__(self, weight=None, size_average=True, gamma=0, alpha_balance=False):
        super(FocalLossElementwise, self).__init__(weight, size_average)
        self.gamma = gamma
        self.alpha_balance = alpha_balance

    def forward(self, input, target):
        # Assumes true is a binary tensor.
        # All operations are elementwise, unless stated otherwise.
        # ELementwise cross-entropy
        # xent = true * torch.log(pred) + (1 - true) * torch.log(1 - pred)

        # p_t
        p_t = input * target   # p   if y == 1    -- where y is 0, this produces a 0
        p_t += (1 - input) * (1 - target)     # 1 - p   if y == 0      -- where y = 1, this adds 0

        fl_coef = -1 * ((1 - p_t) ** self.gamma)

        # Still elementwise...
        fl = fl_coef * torch.log(p_t)

        if self.alpha_balance:
            alpha = target.sum() / target.numel()
            alpha_t = alpha * input + (1 - alpha) * (1 - input)
            fl *= alpha_t

        # Now we would aggregate this
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

