"""
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
    is_dsa_skip_topk_layer,
    source_dsa_compute_layer,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def validate_dsa_index_share_pipeline_split(config: TransformerConfig, local_layer_ids) -> None:
    """Ensure DSA top-k sharing does not require top-k indices from another PP stage.

    Cross-layer index sharing propagates top-k indices through a per-forward carrier inside a
    single transformer block. Pipeline layout APIs provide local decoder layer ids as a contiguous,
    ascending range, so only the first layer can depend on a previous pipeline stage.
    """
    topk_freq = config.dsa_indexer_topk_freq
    if config.experimental_attention_variant != "dsa" or topk_freq <= 1 or not local_layer_ids:
        return

    layer_number = local_layer_ids[0] + 1
    skip_topk_offset = config.dsa_indexer_skip_topk_offset
    if not is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq):
        return

    source_layer_number = source_dsa_compute_layer(layer_number, skip_topk_offset, topk_freq)
    raise RuntimeError(
        "DSA index-share pipeline split is invalid: local layer "
        f"{layer_number} reuses top-k indices from computing layer "
        f"{source_layer_number}, but that source layer is not earlier in this "
        "pipeline stage. Cross-layer top-k sharing does not cross PP boundaries. "
        "Choose a pipeline layout where each stage starts on a computing layer "
        f"(dsa_indexer_topk_freq={topk_freq}, "
        f"dsa_indexer_skip_topk_offset={skip_topk_offset})."
    )


def get_dsa_module_spec_for_backend(
    backend: BackendSpecProvider,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    num_experts: Optional[int] = None,
    mlp: Optional[ModuleSpec] = None,
    enable_hyper_connection: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for Sparse Attention."""
    assert multi_latent_attention, "Currently only MLA supports sparse attention."
    assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."

    linear_q_up_proj = (
        backend.column_parallel_layer_norm_linear()
        if qk_layernorm
        else backend.column_parallel_linear()
    )
    linear_kv_up_proj = (
        backend.column_parallel_layer_norm_linear()
        if qk_layernorm
        else backend.column_parallel_linear()
    )

    hc_module = HyperConnectionModule if enable_hyper_connection else IdentityOp

    # Because TransformerEngine does not support sparse attention yet, we use local
    # implementation whether the backend is TransformerEngine or not.
    core_attention = ModuleSpec(
        module=DSAttention,
        submodules=DSAttentionSubmodules(
            indexer=ModuleSpec(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=backend.linear(),
                    linear_wk=backend.linear(),
                    k_norm=backend.layer_norm(rms_norm=False, for_qk=True),
                    linear_weights_proj=backend.linear(),
                ),
            )
        ),
    )

    attention = ModuleSpec(
        module=MLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=linear_q_up_proj,
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=linear_kv_up_proj,
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
        ),
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=backend.layer_norm(),
            self_attention=attention,
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=hc_module,
            pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            mlp_hyper_connection=hc_module,
        ),
    )


def get_experimental_attention_variant_module_spec_for_backend(
    backend: BackendSpecProvider,
    experimental_attention_variant: Optional[str] = None,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    num_experts: Optional[int] = None,
    mlp: Optional[ModuleSpec] = None,
    enable_hyper_connection: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for Attention"""
    if experimental_attention_variant == "dsa":
        return get_dsa_module_spec_for_backend(
            backend=backend,
            qk_layernorm=qk_layernorm,
            qk_l2_norm=qk_l2_norm,
            multi_latent_attention=multi_latent_attention,
            num_experts=num_experts,
            mlp=mlp,
            enable_hyper_connection=enable_hyper_connection,
        )
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {experimental_attention_variant}"
        )
