"""
Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
This module provides megatron-core-xpu plugin.
"""

import importlib
import os
import sys

from . import (
    _hydra,
    empty_package,
    gated_deltanet,
    mock_megatron,
    multi_latent_attention,
    scaled_masked_softmax,
    scaled_softmax,
    scaled_upper_triang_masked_softmax,
)

__all__ = [
    "empty_package",
    "scaled_upper_triang_masked_softmax",
    "scaled_masked_softmax",
    "scaled_softmax",
    "multi_latent_attention",
]


import_module_list = [
    "megatron",
    "megatron.training",
    "megatron.core.fusions",
    "megatron.core.transformer.moe.token_dispatcher",
    "megatron.core.transformer.moe.moe_layer",
    "megatron.core.transformer.multi_latent_attention",
    "megatron.core.fusions.linear_cross_entropy.generic.entry",
]


def import_modules(verbose=True):
    """
    import_modules
    """
    for m in import_module_list:
        try:
            importlib.import_module(m)
        except Exception as e:
            if verbose:
                print(f'[WARN] Import Module {m} failed: {e}')


def replace_module(source: str, target: str, verbose=True):
    """
    replace_module
    """
    try:
        dst_module = importlib.import_module(target)
        sys.modules[source] = dst_module
        if verbose:
            print(f'[INFO] Module {source} replaced with {target}')
    except Exception as e:
        if verbose:
            print(f'[WARN] Module {source} replaced with {target} failed: {e}')


def replace_symbol(symbol_path, target, record_origin=True, verbose=True):
    """
    replace_symbol
    """
    parts = symbol_path.split('.')
    module_name = parts[0]

    if module_name not in sys.modules:
        try:
            importlib.import_module(module_name)
        except Exception as e:
            if verbose:
                print(f'[WARN] Failed to import {module_name}: {e}')
            return

    obj = sys.modules[module_name]
    target_name = getattr(target, '__name__', repr(target))
    try:
        for part in parts[1:-1]:
            obj = getattr(obj, part)
        origin = getattr(obj, parts[-1], None)
        setattr(obj, parts[-1], target)
        if record_origin and origin is not None:
            setattr(obj, f'__origin_{parts[-1]}', origin)
        if verbose:
            print(f'[INFO] Symbol {symbol_path} replaced with {target_name}')
    except Exception as e:
        if verbose:
            print(f'[WARN] Symbol {symbol_path} replaced with {target_name} failed: {e}')


def init_megatron_core_xpu_plugin():
    """
    init_megatron_core_xpu_plugin
    """
    import_modules()

    # replace_module
    replace_module("fused_layer_norm_cuda", "xpu_plugin.empty_package")
    replace_module(
        "scaled_upper_triang_masked_softmax_cuda", "xpu_plugin.scaled_upper_triang_masked_softmax"
    )
    replace_module("scaled_masked_softmax_cuda", "xpu_plugin.scaled_masked_softmax")
    replace_module("scaled_softmax_cuda", "xpu_plugin.scaled_softmax")

    # replace_symbol
    replace_symbol(
        "transformer_engine_torch.thd_get_partitioned_indices",
        mock_megatron.mock_thd_get_partitioned_indices,
    )
    replace_symbol(
        "megatron.legacy.model.fused_layer_norm.MixedFusedLayerNorm",
        mock_megatron.MockMixedFusedLayerNorm,
    )
    replace_symbol("megatron.legacy.model.LayerNorm", mock_megatron.MockMixedFusedLayerNorm)
    replace_symbol("megatron.legacy.fused_kernels.load", empty_package.empty_function)

    replace_symbol(
        "megatron.core.transformer.mlp.weighted_bias_swiglu_impl",
        mock_megatron.mock_weighted_bias_swiglu_impl,
    )
    replace_symbol(
        "megatron.core.transformer.moe.experts.weighted_bias_swiglu_impl",
        mock_megatron.mock_weighted_bias_swiglu_impl,
    )
    replace_symbol(
        "megatron.core.tensor_parallel.mappings._gather_along_first_dim",
        mock_megatron.mock_gather_along_first_dim,
    )
    replace_symbol(
        "megatron.core.tensor_parallel.mappings._reduce_scatter_along_first_dim",
        mock_megatron.mock_reduce_scatter_along_first_dim,
    )
    replace_symbol(
        "megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available",
        mock_megatron.mock_is_kernel_available,
    )
    replace_symbol(
        "megatron.core.fp8_utils._modify_underlying_storage_impl", empty_package.empty_function
    )
    replace_symbol(
        "megatron.core.fp8_utils._quantize_param_shard_impl", empty_package.empty_function
    )
    replace_symbol(
        "megatron.core.transformer.multi_latent_attention.MLASelfAttention",
        multi_latent_attention.MLASelfAttention
    )
    if os.getenv("XMLIR_USE_ROPE_FUSION") in ("true", "1", "True"):
        replace_symbol(
            "megatron.core.models.common.embeddings.rope_utils.apply_rotary_emb_flash", None
        )
        replace_symbol(
            "megatron.core.models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd",
            mock_megatron.mock_apply_rotary_pos_emb_bshd,
        )
        replace_symbol(
            "megatron.core.models.common.embeddings.rotary_pos_embedding._apply_rotary_pos_emb_bshd",
            mock_megatron.mock_apply_rotary_pos_emb_bshd,
        )

    if os.getenv("XMLIR_MOCK_ASYNC_LINEAR") not in ("false", "0", "False"):
        replace_symbol(
            "megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication",
            _hydra.LinearWithGradAccumulationAndAsyncCommunication,
        )
        replace_symbol(
            "megatron.core.tensor_parallel.layers.LinearWithFrozenWeight",
            _hydra.LinearWithFrozenWeight,
        )

    if os.getenv("XMLIR_MOCK_GATED_DELTANET") not in ("false", "0", "False"):
        gated_deltanet._install_gdn_mock_hook()

    # Mock fused linear cross-entropy backward for XPU.
    if os.getenv("XMLIR_MOCK_LCE") not in ("false", "0", "False"):
        replace_symbol(
            "megatron.core.fusions.linear_cross_entropy.generic.entry.backward",
            mock_megatron.mock_lce_backward,
        )
