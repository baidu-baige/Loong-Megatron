"""
Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
This module provides megatron-core-xpu plugin.
"""

from . import scaled_masked_softmax

forward = scaled_masked_softmax.forward
