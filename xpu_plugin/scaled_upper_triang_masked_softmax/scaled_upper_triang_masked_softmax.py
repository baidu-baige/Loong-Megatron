"""
Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
This module provides megatron-core-xpu plugin.
"""

from torch_xmlir.nn import scaled_upper_triang_masked_softmax


def forward(input, scale):
    """
    forward
    """
    return scaled_upper_triang_masked_softmax.ScaledUpperTriangMaskedSoftmaxFunction.apply(
        input, scale
    )
