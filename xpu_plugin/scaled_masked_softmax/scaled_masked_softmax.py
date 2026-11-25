"""
Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
This module provides megatron-core-xpu plugin.
"""

from torch_xmlir.nn import softmax_with_mask


def forward(input, mask, scale):
    """
    forward
    """
    return softmax_with_mask.SoftmaxWithMaskFunction.apply(input, mask, scale)
