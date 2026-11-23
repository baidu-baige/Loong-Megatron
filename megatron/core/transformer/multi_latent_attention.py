# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, Optional, Union

import torch
import torch.nn.functional as F

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fusions.fused_row_copy import contiguous_rows
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    _yarn_get_mscale,
    apply_rotary_pos_emb,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention, LinearProjBuilder
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mla_qk_norm_config import QKNormConfigResolver
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import (
    deprecate_inference_params,
    get_pg_size,
    is_te_min_version,
    make_tp_sharded_tensor_for_checkpoint,
)

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        fused_apply_mla_rope_for_kv,
        fused_apply_mla_rope_for_q,
    )
except:
    fused_apply_mla_rope_for_kv = None
    fused_apply_mla_rope_for_q = None


if HAVE_TE:
    import transformer_engine_torch as tex

    try:
        from transformer_engine.pytorch.attention import (
            FusedMLAQUpProjFunction,
            FusedMLAQUpProjRopeQuant,
        )
        from transformer_engine.pytorch.attention.dot_product_attention.utils import (
            mxfp8_quantize_only,
            mxfp8_transpose_swizzle,
        )
    except ImportError:
        # Older TE lacks the fused MLA Q up-proj kernel and its MXFP8 quantize/swizzle
        # helpers. Stub them all together so this module still imports; the fused path is
        # gated on FusedMLAQUpProjRopeQuant being non-None.
        FusedMLAQUpProjFunction = None
        FusedMLAQUpProjRopeQuant = None
        mxfp8_quantize_only = None
        mxfp8_transpose_swizzle = None
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        set_save_original_input,
        split_te_layernorm_column_parallel_linear,
    )
    from megatron.core.post_training.modelopt.layers import Linear
else:
    (
        tex,
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        Linear,
        set_save_original_input,
        split_te_layernorm_column_parallel_linear,
        FusedMLAQUpProjFunction,
        FusedMLAQUpProjRopeQuant,
        mxfp8_quantize_only,
        mxfp8_transpose_swizzle,
        QuantizedTensor,
        MXFP8Quantizer,
    ) = (None, None, None, None, None, None, None, None, None, None, None, None, None)

if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext
    from megatron.core.packed_seq_params import PackedSeqParams


class _QuantizeKVForFusedAttn(torch.autograd.Function):
    """Bridge the autograd graph across mxfp8_quantize_only's tensor reconstruction."""

    @staticmethod
    def forward(ctx, key, value):
        """Quantize the k anv v tensors"""
        kq = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        vq = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        k_mx, v_mx = mxfp8_quantize_only([(key.contiguous(), kq), (value.contiguous(), vq)], "sbhd")
        return k_mx, v_mx

    @staticmethod
    def backward(ctx, dk, dv):
        """Attention returns bf16 tensors, so no need to quantize/unquantize in bwd"""
        return dk, dv


def _prepare_mla_core_attention_value(parallel_attention, query, value, packed_seq_params):
    """Pad the MLA value head when the selected attention path requires QK-width values."""
    orig_v_dim = value.shape[-1] if value is not None else None
    padded_v_dim = orig_v_dim
    thd_padding_workaround = (
        packed_seq_params is not None
        and packed_seq_params.qkv_format == "thd"
    )
    need_v_pad = (
        parallel_attention.config.experimental_attention_variant is None
        and value is not None
        and query.shape[-1] != orig_v_dim
        and (parallel_attention.config.padding_v_to_qk_dim or thd_padding_workaround)
    )
    if need_v_pad:
        value = F.pad(value, [0, query.shape[-1] - orig_v_dim])
        padded_v_dim = value.shape[-1]
    return value, need_v_pad, orig_v_dim, padded_v_dim


def _align_chunkpipe_cache_grad(grad, cached_grad):
    """Align an SFT cache gradient with the current packed activation gradient."""
    if grad.shape == cached_grad.shape:
        return cached_grad
    if grad.ndim == 4 and cached_grad.ndim == 3 and grad.size(1) == 1:
        return cached_grad.unsqueeze(1)
    if grad.ndim == 3 and cached_grad.ndim == 2 and grad.size(1) == 1:
        return cached_grad.unsqueeze(1)
    if grad.ndim == 3 and cached_grad.ndim == 4 and cached_grad.size(1) == 1:
        return cached_grad.squeeze(1)
    if grad.ndim == 2 and cached_grad.ndim == 3 and cached_grad.size(1) == 1:
        return cached_grad.squeeze(1)
    return cached_grad


def _trim_mla_core_attention_output(core_attn_out, need_v_pad, orig_v_dim, padded_v_dim):
    """Trim MLA core attention output back to the original per-head V dimension."""
    if need_v_pad:
        output_prefix = core_attn_out.shape[:-1]
        core_attn_out = core_attn_out.reshape(*output_prefix, -1, padded_v_dim)
        core_attn_out = core_attn_out[..., :orig_v_dim].reshape(*output_prefix, -1)
    return core_attn_out


def _up_proj_input(projection: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the up-projection input, copied to contiguous memory if ``projection`` needs it.

    ``x`` is a column slice of the down-projection output. Transformer Engine projections make
    such a row-strided input contiguous with ``Tensor.contiguous``, so for them the slice is
    copied first with the vectorized ``contiguous_rows``; other projections read it as it is.
    """
    if HAVE_TE and isinstance(projection, (TELinear, TELayerNormColumnParallelLinear)):
        return contiguous_rows(x)
    return x


@dataclass
class MLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_proj: LinearProjBuilder

    # TODO(nschank): Move layernorms back to the bottom once all other layers have defaults removed.
    q_layernorm: LayerNormBuilder
    kv_layernorm: LayerNormBuilder

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    linear_qkv_down_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None


class MultiLatentAttention(Attention):
    """Multi-Latent Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        # TODO(nschank): Restructure so that the Attention initializer knows which specific
        # submodules it will construct, so that MLASelfAttentionSubmodules honors that interface.
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )
        self.config: MLATransformerConfig

        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads

        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim
        self.v_channels = self.config.v_head_dim
        if (
            self.config.padding_v_to_qk_dim
            and self.q_head_dim > self.config.v_head_dim
            and self.config.experimental_attention_variant is None
        ):
            self.v_channels = self.q_head_dim

        # Overwrite the base class kv shape to support MLA inference
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim
        self.v_channels = self.config.v_head_dim
        self.padding_v_head_dim = False
        if self.config.enable_chunkpipe:
            self.v_channels = self.q_head_dim
            self.padding_v_head_dim = True

        self.recompute_up_proj = (
            self.config.recompute_granularity == 'selective'
            and "mla_up_proj" in self.config.recompute_modules
        )
        self.qkv_up_checkpoint = None

        # Fused MLA Q up-proj+rope+quant via cuDNN gemm_proj_rope_mxfp8.
        # The q_layernorm is absorbed into the fusion so that MXFP8 is produced in one step
        # from the norm's FP32 accumulator, as TELayerNormColumnParallelLinear does; that
        # requires the norm to exist and to be an RMSNorm (because currently it is only
        # tested with DSv3 which uses RMSNorm)
        self._use_fused_q_uproj = (
            getattr(self.config, "use_fused_mla_q_uproj", False)
            and FusedMLAQUpProjRopeQuant is not None
            and FusedMLAQUpProjRopeQuant.is_supported()
            and get_pg_size(self.tp_group) == 1
            and self.config.qk_layernorm
            and self.config.normalization == "RMSNorm"
        )

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale_all_dim)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)
        self.cache_mla_latents = self.config.cache_mla_latents

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":

            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
                "'rope' and 'yarn'"
            )

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.v_channels,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
        )

        # Output.
        self.linear_proj = submodules.linear_proj(
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=not_none(self.config.output_layer_init_method),
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
            tp_group=self.pg_collection.tp,
            name=(name + ".linear_proj") if name is not None else None,
        )

        if (
            HAVE_TE
            and isinstance(self.linear_proj, TELinear)
            and (
                (
                    self.config.fp8
                    and self.config.fp8_recipe != 'delayed'
                    and is_te_min_version("2.6.0dev0")
                )
                or (self.config.fp4 and is_te_min_version("2.7.0.dev0"))
            )
        ):
            # For fp8/fp4 training, the output of the fused core_attn is saved by itself, and
            # linear_proj also saves the quantized tensor of this output. Here we set the
            # linear_proj to save the original input tensors to avoid the extra memory usage of
            # the quantized tensor.
            set_save_original_input(self.linear_proj)

        if self.config.enable_chunkpipe:
            self.setup_chunkpipe_kv_cache_config()

    def init_chunk_key_value_cache_mla(self):
        """Initialize the chunk-based key-value cache for chunkpipe functionality in MLA."""

        # Initialize cache tensors with proper device and dtype
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        slot_size = self.config.max_chunk_size or self.config.chunksize
        total_cache_tokens = self.kv_cache_chunk_size * slot_size
        if self.config.sft_chunkpipe_mode:
            # SFT packed sequence: no batch dimension
            kv_cache_shape = (total_cache_tokens // get_pg_size(self.pg_collection.tp),
                              self.config.kv_lora_rank)
            k_pos_emb_cache_shape = (total_cache_tokens, 1, self.config.qk_pos_emb_head_dim)
        else:
            # Pretrain: with batch dimension
            kv_cache_shape = (total_cache_tokens // get_pg_size(self.pg_collection.tp),
                              self.config.micro_batch_size, self.config.kv_lora_rank)
            k_pos_emb_cache_shape = (total_cache_tokens, self.config.micro_batch_size,
                                     1, self.config.qk_pos_emb_head_dim)

        self.kv_compressed_cache = torch.zeros(kv_cache_shape, device=device, dtype=dtype)
        self.key_pos_emb_cache = torch.zeros(k_pos_emb_cache_shape, device=device, dtype=dtype)

        self.kv_compressed_cache_grad = {}
        self.key_pos_emb_cache_grad = {}

    def append_chunk_key_value_cache_mla(self, normed_kv_compressed, k_pos_emb):
        """Cache a chunk of compressed key-value pairs and positional embeddings for MLA.

        Args:
            normed_kv_compressed (Tensor): Normalized compressed key-value tensor
                with shape [chunksize, batch_size, kv_lora_rank]
            k_pos_emb (Tensor): Key positional embeddings with shape
                [chunksize, batch_size, 1, qk_pos_emb_head_dim]

        Raises:
            RuntimeError: If chunkpipe is not enabled in config
            RuntimeError: If no available cache chunks
        """
        if not self.config.enable_chunkpipe:
            raise RuntimeError("Chunk key-value cache operations require chunkpipe to be enabled.")

        # Skip caching during backward pass or for last chunk in sequence
        if not self.config.chunkpipe_forward:
            return
        if self.config.sft_chunkpipe_mode:
            # SFT: use scheduler-provided chunk index to detect last chunk
            if self.config.chunkpipe_chunk_idx_in_group >= self.config.chunkpipe_current_group_size - 1:
                return
        else:
            # Pretrain: derive from global counter (all groups same size)
            if (self.config.chunkpipe_forward_microbatch + 1) % self.num_chunks_per_seq == 0:
                return

        if not self.empty_chunk_indices:
            raise RuntimeError("No available cache chunks. Consider increasing cache size.")

        chunk_id = self.empty_chunk_indices.pop(0)
        self.micro_batch_to_cache_chunk_map[self.config.chunkpipe_forward_microbatch] = chunk_id

        # Cache slots use max_chunk_size as a fixed stride; each write uses the
        # current chunk's actual length.
        tp_size = get_pg_size(self.pg_collection.tp)
        slot_size = self.config.max_chunk_size or self.config.chunksize
        if self.config.sft_chunkpipe_mode:
            # SFT ChunkPipe uses packed 3-D/4-D tensors with a dummy batch
            # dimension of one, while the cache is stored without that batch
            # dimension.
            if normed_kv_compressed.ndim == 3 and normed_kv_compressed.size(1) == 1:
                normed_kv_compressed = normed_kv_compressed.squeeze(1)
            if k_pos_emb.ndim == 4 and k_pos_emb.size(1) == 1:
                k_pos_emb = k_pos_emb.squeeze(1)
        kv_len = normed_kv_compressed.shape[0]
        pos_len = k_pos_emb.shape[0]
        kv_cache_indices = torch.arange(kv_len) + (chunk_id * slot_size // tp_size)
        pos_cache_indices = torch.arange(pos_len) + (chunk_id * slot_size)

        # When the 3rd dim of k_pos_emb is smaller than the cache,
        # only write to the first corresponding indices
        if self.config.sft_chunkpipe_mode:
            # SFT packed: no batch dim — kv_compressed is 2D, k_pos_emb is 3D
            k_pos_emb_head_num = k_pos_emb.size(1)
            if self.is_enable_grad_chunkpipe():
                self.kv_compressed_cache[kv_cache_indices] = normed_kv_compressed.clone().detach()
                self.key_pos_emb_cache[pos_cache_indices, :k_pos_emb_head_num, :] = k_pos_emb.clone().detach()
            else:
                self.kv_compressed_cache[kv_cache_indices] = normed_kv_compressed
                self.key_pos_emb_cache[pos_cache_indices, :k_pos_emb_head_num, :] = k_pos_emb
        else:
            # Pretrain: with batch dim — kv_compressed is 3D, k_pos_emb is 4D
            k_pos_emb_head_num = k_pos_emb.size(2)
            if self.is_enable_grad_chunkpipe():
                self.kv_compressed_cache[kv_cache_indices, :, :] = normed_kv_compressed.clone().detach()
                self.key_pos_emb_cache[pos_cache_indices, :, :k_pos_emb_head_num, :] = k_pos_emb.clone().detach()
            else:
                self.kv_compressed_cache[kv_cache_indices, :, :] = normed_kv_compressed
                self.key_pos_emb_cache[pos_cache_indices, :, :k_pos_emb_head_num, :] = k_pos_emb

    def check_kv_cache_grad_consumed(self):
        """Validate that MLA cache gradients were consumed by the completed chunk group."""
        if getattr(self, "kv_compressed_cache_grad", None):
            orphan_keys = list(self.kv_compressed_cache_grad.keys())
            self.kv_compressed_cache_grad.clear()
            assert False, (
                f"[ChunkPipe][MLA] Orphan kv_compressed_cache_grad entries {orphan_keys} "
                f"remain after group backward. layer={self.layer_number}"
            )
        if getattr(self, "key_pos_emb_cache_grad", None):
            orphan_keys = list(self.key_pos_emb_cache_grad.keys())
            self.key_pos_emb_cache_grad.clear()
            assert False, (
                f"[ChunkPipe][MLA] Orphan key_pos_emb_cache_grad entries {orphan_keys} "
                f"remain after group backward. layer={self.layer_number}"
            )

    def recover_key_value_up_proj_tensors(self, normed_kv_compressed, k_pos_emb):
        """Recover full key and value tensors from cached compressed MLA representations."""
        kv, _ = self.linear_kv_up_proj(normed_kv_compressed)
        kv = kv.view(
            *kv.size()[:-1],
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )
        k_no_pe, value = torch.split(
            kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
        )
        if k_pos_emb.shape[-2] != k_no_pe.shape[-2]:
            k_pos_emb = k_pos_emb.expand(*k_no_pe.shape[:-1], k_pos_emb.shape[-1])
        key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

        if self.padding_v_head_dim:
            value = F.pad(value, [0, self.q_head_dim - self.config.v_head_dim])

        return key.contiguous(), value.contiguous()

    def concat_cached_chunk_key_value_mla(self, attention_mask, curr_key, curr_value):
        """Concatenate all cached key-value chunks for the current sequence in MLA.

        This function is used during chunkpipe processing to combine cached key-value pairs
        from previous chunks with the current chunk's keys and values. This enables the
        attention mechanism to see the full sequence context across chunk boundaries.

        Args:
            attention_mask (Tensor): Attention mask tensor for the current chunk
            curr_key (Tensor): Current chunk's key tensor with shape
                [chunksize, batch_size, num_heads, head_dim] (unpacked) or
                [chunksize, num_heads, head_dim] (packed sequence, SFT mode)
            curr_value (Tensor): Current chunk's value tensor with same shape as curr_key

        Returns:
            tuple: (concatenated_key, concatenated_value, adjusted_mask)
                concatenated_key: Concatenated key tensor with shape matching curr_key
                    dims, where first dim = (current_chunk_index + 1) * chunksize
                concatenated_value: Concatenated value tensor with same shape as key
                adjusted_mask: Adjusted attention mask with upper triangular masking
                    to prevent attending to future tokens in autoregressive scenarios

        Raises:
            RuntimeError: If chunkpipe is not enabled in config
        """
        # Check if chunkpipe functionality is enabled
        if not self.config.enable_chunkpipe:
            raise RuntimeError("Chunk concatenation requires chunkpipe to be enabled.")

        # Determine current processing stage (forward or backward pass) and position
        is_forward = self.config.chunkpipe_forward
        microbatch_idx = (self.config.chunkpipe_forward_microbatch if is_forward
                         else self.config.chunkpipe_backward_microbatch)
        if self.config.sft_chunkpipe_mode:
            # SFT: use scheduler-provided chunk index within group
            current_chunk_idx = self.config.chunkpipe_chunk_idx_in_group
        else:
            # Pretrain: derive from global counter (all groups same size)
            current_chunk_idx = microbatch_idx % self.num_chunks_per_seq
        start_microbatch_idx = microbatch_idx - current_chunk_idx

        # Initialize lists to store retrieved cache data
        cached_kv_compressed = []
        cached_key_pos_emb = []

        cu_chunks = self.config.cu_chunks
        chunk_sizes = self.config.chunk_sizes
        slot_size = self.config.max_chunk_size or self.config.chunksize
        total_concatenated_tokens = cu_chunks[current_chunk_idx + 1]
        if self.config.sft_chunkpipe_mode:
            if curr_key.ndim == 4 and curr_key.size(1) == 1:
                curr_key = curr_key.squeeze(1)
            if curr_value.ndim == 4 and curr_value.size(1) == 1:
                curr_value = curr_value.squeeze(1)

        # Detect packed sequence mode (SFT chunkpipe uses packed sequences,
        # where batch dim is squeezed: key/value are 3D [t, n, d] instead of 4D [s, b, n, d])
        if self.config.sft_chunkpipe_mode:
            # Packed sequence: [total_tokens, num_heads, head_dim]
            concatenated_key_shape = (total_concatenated_tokens,
                          self.num_attention_heads_per_partition, self.key_hidden_size)
            concatenated_value_shape = (total_concatenated_tokens,
                          self.num_attention_heads_per_partition, self.config.v_head_dim)
            if self.padding_v_head_dim:
                concatenated_value_shape = (total_concatenated_tokens,
                          self.num_attention_heads_per_partition, self.key_hidden_size)
        else:
            # Unpacked: [total_tokens, batch_size, num_heads, head_dim]
            concatenated_key_shape = (total_concatenated_tokens, self.config.micro_batch_size,
                          self.num_attention_heads_per_partition, self.key_hidden_size)
            concatenated_value_shape = (total_concatenated_tokens, self.config.micro_batch_size,
                          self.num_attention_heads_per_partition, self.config.v_head_dim)
            if self.padding_v_head_dim:
                concatenated_value_shape = (total_concatenated_tokens, self.config.micro_batch_size,
                          self.num_attention_heads_per_partition, self.key_hidden_size)

        # Initialize zero tensors for concatenated key and value of previous and current chunk
        concatenated_key = torch.zeros(concatenated_key_shape,
            device=self.kv_compressed_cache.device, dtype=self.kv_compressed_cache.dtype)
        concatenated_value = torch.zeros(concatenated_value_shape,
            device=self.kv_compressed_cache.device, dtype=self.kv_compressed_cache.dtype)

        def kv_compressed_hook_fn(chunk_index):
            """
            Hook function to accumulate gradient of loss of current chunk
            with respect to cached compressed KV of previous chunk during backward pass.
            """
            def hook_fn(grad):
                if chunk_index not in self.kv_compressed_cache_grad:
                    self.kv_compressed_cache_grad[chunk_index] = grad
                else:
                    self.kv_compressed_cache_grad[chunk_index] += grad
                return grad
            return hook_fn

        def key_pos_emb_hook_fn(chunk_index):
            """
            Hook function to accumulate gradient of loss of current chunk
            with respect to cached key position embeddings of previous chunks during backward pass.
            """
            def hook_fn(grad):
                if chunk_index not in self.key_pos_emb_cache_grad:
                    self.key_pos_emb_cache_grad[chunk_index] = grad
                else:
                    self.key_pos_emb_cache_grad[chunk_index] += grad
                return grad
            return hook_fn

        # Retrieve all previous chunks from cache and concatenate with current chunk
        current_pos = 0
        for prev_chunk_idx in range(current_chunk_idx):
            # Get the cache index for this previous chunk
            cache_chunk_idx = self.micro_batch_to_cache_chunk_map[start_microbatch_idx + prev_chunk_idx]

            # Cache slots are slot_size-strided, but only the previous chunk's
            # real token count is read.
            previous_chunk_size = chunk_sizes[prev_chunk_idx]
            tp_size = get_pg_size(self.pg_collection.tp)
            kv_indices = torch.arange(previous_chunk_size // tp_size) + (
                cache_chunk_idx * slot_size // tp_size
            )
            pos_indices = torch.arange(previous_chunk_size) + (
                cache_chunk_idx * slot_size
            )

            # Retrieve compressed KV and positional embeddings from cache
            if self.config.sft_chunkpipe_mode:
                # SFT packed: no batch dim
                cached_kv_compressed.append(self.kv_compressed_cache[kv_indices])
                cached_key_pos_emb.append(self.key_pos_emb_cache[pos_indices])
            else:
                # Pretrain: with batch dim
                cached_kv_compressed.append(self.kv_compressed_cache[kv_indices, :, :])
                cached_key_pos_emb.append(self.key_pos_emb_cache[pos_indices, :, :, :])

            # Set up gradient hooks for backward pass
            if self.is_enable_grad_chunkpipe():
                cached_kv_compressed[-1].requires_grad = True
                cached_key_pos_emb[-1].requires_grad = True
                cached_kv_compressed[-1].register_hook(kv_compressed_hook_fn(prev_chunk_idx))
                cached_key_pos_emb[-1].register_hook(key_pos_emb_hook_fn(prev_chunk_idx))

            # Recover full key and value tensors from compressed representations
            cached_key, cached_value = self.recover_key_value_up_proj_tensors(
                cached_kv_compressed[-1],
                cached_key_pos_emb[-1]
            )

            concatenated_key[current_pos : current_pos + previous_chunk_size] = cached_key
            concatenated_value[current_pos : current_pos + previous_chunk_size] = cached_value
            current_pos += previous_chunk_size

        # Add the current chunk's keys and values to the concatenated result
        current_chunk_size = curr_key.shape[0]
        concatenated_key[current_pos : current_pos + current_chunk_size] = curr_key
        concatenated_value[current_pos : current_pos + current_chunk_size] = curr_value

        # Adjust attention mask for autoregressive (causal) attention
        adjusted_mask = None
        if attention_mask is not None:
            mask_org = torch.ones(
                current_chunk_size,
                total_concatenated_tokens,
                dtype=torch.bool,
                device=self.kv_compressed_cache.device,
            )
            mask_index = cu_chunks[current_chunk_idx] + 1
            # Apply upper triangular masking (applying in-place operation triu_ to avoid doubling memory)
            mask_org.triu_(diagonal=mask_index)
            # Reshape mask to match attention mechanism requirements
            # Using .view creates a new tensor without allocating additional memory
            adjusted_mask = mask_org.view(
                1, 1, current_chunk_size, total_concatenated_tokens
            )

        return concatenated_key, concatenated_value, adjusted_mask

    def _run_core_attention(
        self,
        query,
        key,
        value,
        attention_mask,
        packed_seq_params=None,
        attn_mask_type=None,
        **extra_kwargs,
    ):
        """Run MLA core attention with the THD value pad/trim workaround."""
        packed_thd = (
            packed_seq_params is not None
            and getattr(packed_seq_params, "qkv_format", None) == "thd"
        )
        inputs_were_packed = any(
            tensor is not None and tensor.ndim == 3
            for tensor in (query, key, value)
        )
        if not packed_thd and inputs_were_packed:
            # TE requires [s, b, h, d] for the sbhd path. SFT ChunkPipe can
            # reach this branch for a one-chunk group, where the upstream
            # packed tensors are already [t, h, d]. Preserve THD when packed
            # metadata is present; otherwise add the dummy batch dimension
            # before MLA's value padding and TE dispatch.
            query = query.unsqueeze(1) if query is not None and query.ndim == 3 else query
            key = key.unsqueeze(1) if key is not None and key.ndim == 3 else key
            value = value.unsqueeze(1) if value is not None and value.ndim == 3 else value
            packed_seq_params = None
        value, need_v_pad, orig_v_dim, padded_v_dim = _prepare_mla_core_attention_value(
            self, query, value, packed_seq_params
        )
        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type

        orig_v_head_dim = None
        if (
            need_v_pad
            and value is not None
            and hasattr(self.core_attention, 'hidden_size_per_attention_head_v')
        ):
            orig_v_head_dim = self.core_attention.hidden_size_per_attention_head_v
            if value.shape[-1] == orig_v_head_dim:
                orig_v_head_dim = None
            else:
                self.core_attention.hidden_size_per_attention_head_v = value.shape[-1]

        try:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
                **extra_kwargs,
            )
        finally:
            if orig_v_head_dim is not None:
                self.core_attention.hidden_size_per_attention_head_v = orig_v_head_dim

        core_attn_out = _trim_mla_core_attention_output(
            core_attn_out, need_v_pad, orig_v_dim, padded_v_dim
        )
        if (
            not packed_thd
            and inputs_were_packed
            and query is not None
            and query.ndim == 4
            and query.size(1) == 1
            and core_attn_out.ndim == 4
            and core_attn_out.size(1) == 1
        ):
            core_attn_out = core_attn_out.squeeze(1)
        return core_attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        position_ids: torch.Tensor | None = None,
        sequence_len_offset: int | None = None,
        index_share_carrier: object = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        packed_sequence_cp_metadata=None,
    ):
        """Forward pass for multi-latent attention"""
        assert (
            packed_sequence_cp_metadata is None
        ), "MLA does not support packed-sequence chunkwise CP metadata."
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and self.cache_mla_latents
        ), "cache_mla_latents conflicts with training."

        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        if self.config.cache_mla_latents:
            self.prepare_for_absorption()

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        qkv_linear_manager = off_interface(self.offload_qkv_linear, hidden_states, "qkv_linear")
        with qkv_linear_manager as hidden_states:
            query, key, value, q_compressed, kv_compressed = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
            )
        query = qkv_linear_manager.group_offload(query, forced_released_tensors=[hidden_states])

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        if self.config.enable_chunkpipe:
            key, value, attention_mask = self.concat_cached_chunk_key_value_mla(
                attention_mask, key, value
            )
            if (
                not self.config.sft_chunkpipe_mode
                or (
                    self.config.chunkpipe_current_group_size > 1
                    and self.config.chunkpipe_chunk_idx_in_group > 0
                )
            ) and self.config.experimental_attention_variant != "dsa":
                attn_mask_type = AttnMaskType.causal_bottom_right
                packed_seq_params = None

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()

        # Value is none during decode for absorption
        if value is not None:
            value = value.contiguous()

        thd_packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        core_attention_extra_kwargs = {}
        if getattr(self.core_attention, "requires_dsa_inputs", False):
            core_attention_extra_kwargs = {
                "x": hidden_states,
                "qr": q_compressed,
                "index_share_carrier": index_share_carrier,
            }
        if self._use_fused_q_uproj and self.training:
            # Fused path fed pre-quantized MXFP8 q/k/v (is_input_fp8). Return bf16
            # grads (skip attention's backward re-quantize) so the Q/KV backward
            # paths receive plain bf16 gradients -- no swizzled-fp8 dequant needed.
            core_attention_extra_kwargs["bf16_backward"] = True

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        core_attn_manager = off_interface(
            self.offload_core_attention and self.training, query, "core_attn"
        )
        needs_output_trim = False
        orig_v_dim = None
        padded_v_dim = None
        if self.checkpoint_core_attention and self.training:
            assert not self._use_fused_q_uproj, (
                "use_fused_mla_q_uproj is incompatible with checkpoint_core_attention: "
                "bf16_backward is not threaded through the checkpointed attention path. "
                "Disable one of them."
            )
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
                core_attention_extra_kwargs=core_attention_extra_kwargs,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                with core_attn_manager as query:
                    if (
                        self.config.enable_chunkpipe
                        and self.config.sft_chunkpipe_mode
                        and self.config.chunkpipe_current_group_size > 1
                    ):
                        def _ensure_sbhd(tensor):
                            return (
                                tensor.unsqueeze(1)
                                if tensor is not None and tensor.ndim == 3
                                else tensor
                            )

                        core_attn_out = self._run_core_attention(
                            _ensure_sbhd(query),
                            _ensure_sbhd(key),
                            _ensure_sbhd(value),
                            attention_mask,
                            packed_seq_params=None,
                            attn_mask_type=AttnMaskType.causal_bottom_right,
                            **core_attention_extra_kwargs,
                        )
                    else:
                        core_attn_out = self._run_core_attention(
                            query,
                            key,
                            value,
                            attention_mask,
                            packed_seq_params=packed_seq_params,
                            attn_mask_type=attn_mask_type,
                            **core_attention_extra_kwargs,
                        )
            elif self.cache_mla_latents:
                value, need_v_pad, orig_v_dim, padded_v_dim = _prepare_mla_core_attention_value(
                    self, query, value, packed_seq_params
                )
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                    inference_context.is_decode_only(),
                )
                # Only rearrange if not in absorption mode (Flash MLA handles format correctly)
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')
                needs_output_trim = need_v_pad
            forced = [
                t
                for t in [query, key, value]
                if t is not None and not isinstance(t, QuantizedTensor)
            ]
            core_attn_out = core_attn_manager.group_offload(
                core_attn_out, forced_released_tensors=forced
            )

        # We are doing absorption with cache mla latents and decode mode.
        if self.cache_mla_latents and inference_context.is_decode_only():
            # core_attn_out = self.self.up_v_layer(core_attn_out)
            core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
            core_attn_out = core_attn_out.contiguous()

            # Flatten back: [seq, batch, num_heads * v_head_dim]
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        if self.padding_v_head_dim:
            prefix = core_attn_out.shape[:-1]
            core_attn_out = core_attn_out.reshape(*prefix, -1, self.v_channels)
            core_attn_out = core_attn_out[..., : self.config.v_head_dim].reshape(*prefix, -1)

        if needs_output_trim:
            core_attn_out = _trim_mla_core_attention_output(
                core_attn_out, need_v_pad, orig_v_dim, padded_v_dim
            )

        if thd_packed_seq or (
            self.config.enable_chunkpipe
            and self.config.sft_chunkpipe_mode
            and self.config.chunkpipe_current_group_size > 1
        ):
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        attn_proj_manager = off_interface(self.offload_attn_proj, core_attn_out, "attn_proj")
        with attn_proj_manager as core_attn_out:
            output, bias = apply_module(self.linear_proj)(core_attn_out)
        output = attn_proj_manager.group_offload(output, forced_released_tensors=[core_attn_out])

        return output, bias


class MLASelfAttention(MultiLatentAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )

        # Resolve which classes to use for Q and KV linear up projections and norms, based on
        # QK-norm selection.
        layer_classes = self._resolve_qk_norm_config(submodules)

        if self.config.q_lora_rank is None:
            # Not projecting query
            self.linear_q_proj = build_module(
                layer_classes["linear_q_proj"],
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_proj',
                name=(name + ".linear_q_proj") if name is not None else None,
            )

        else:
            q_down_proj_kwargs = {}
            if submodules.linear_q_down_proj in [TELinear]:
                q_down_proj_kwargs['parallel_mode'] = 'duplicated'
            elif submodules.linear_q_down_proj in [
                Linear,
                TEColumnParallelLinear,
                ColumnParallelLinear,
            ]:
                q_down_proj_kwargs['gather_output'] = False
            else:
                raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

            self.linear_q_down_proj = build_module(
                submodules.linear_q_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_down_proj',
                skip_weight_param_allocation=False,
                tp_group=(
                    pg_collection.tp
                    if q_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                    else None
                ),
                name=(name + ".linear_q_down_proj") if name is not None else None,
                **q_down_proj_kwargs,
            )

            self.linear_q_up_proj = build_module(
                layer_classes["linear_q_up_proj"],
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_up_proj',
                tp_group=pg_collection.tp,
                name=(name + ".linear_q_up_proj") if name is not None else None,
            )

        kv_down_proj_kwargs = {}
        if submodules.linear_kv_down_proj in [TELinear]:
            kv_down_proj_kwargs['parallel_mode'] = 'duplicated'
        elif submodules.linear_kv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            kv_down_proj_kwargs['gather_output'] = False
        else:
            raise ValueError(f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}")

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_down_proj',
            skip_weight_param_allocation=False,
            tp_group=(
                pg_collection.tp
                if kv_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                else None
            ),
            name=(name + ".linear_kv_down_proj") if name is not None else None,
            **kv_down_proj_kwargs,
        )

        self.linear_kv_up_proj = build_module(
            layer_classes["linear_kv_up_proj"],
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
            tp_group=pg_collection.tp,
            name=(name + ".linear_kv_up_proj") if name is not None else None,
        )

        if self.config.q_lora_rank is not None:
            self.q_layernorm = layer_classes["q_layernorm"](
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_layernorm = layer_classes["kv_layernorm"](
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        if self.config.enable_chunkpipe:
            self.init_chunk_key_value_cache_mla()

    def _resolve_qk_norm_config(
        self, submodules
    ) -> dict[str, ModuleSpec | type | LayerNormBuilder]:
        """Resolve which Q/KV norm and up-projection implementations to build."""
        return QKNormConfigResolver(self.config, submodules).resolve()

    def _qkv_down_projection(self, hidden_states):
        """Unfused q/kv down projection path."""
        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        return q_compressed, kv_combined

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        if packed_seq_params is not None:
            assert (
                packed_seq_params.local_cp_size is None
            ), "hybrid_context_parallel is not supported with MLA yet and is planned for future. \
            Please disable hybrid_context_parallel."

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        pos_emb_offset = 0
        if self.config.enable_chunkpipe:
            if self.config.sft_chunkpipe_mode:
                chunk_index = self.config.chunkpipe_chunk_idx_in_group
            else:
                chunk_index = self.config.chunkpipe_forward_microbatch % (
                    self.config.chunk_num_per_seq
                )
                if not self.config.chunkpipe_forward:
                    chunk_index = self.config.chunkpipe_backward_microbatch % (
                        self.config.chunk_num_per_seq
                    )
            pos_emb_offset = self.config.cu_chunks[chunk_index]

        # rotary_pos_emb:[s, b, 1, 64]
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        thd_packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len, offset=pos_emb_offset, packed_seq=thd_packed_seq
            )
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rotary_seq_len,
                    offset=pos_emb_offset,
                    dtype=hidden_states.dtype,
                    packed_seq=thd_packed_seq,
                )
                rotary_pos_emb = None
                assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
                assert (
                    fused_apply_mla_rope_for_q is not None
                    and fused_apply_mla_rope_for_kv is not None
                ), "Fused MLA RoPE apply is not imported successfully"
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(
                    rotary_seq_len, offset=pos_emb_offset, packed_seq=thd_packed_seq
                )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            rope_max_seqlen_q = packed_seq_params.max_seqlen_q
            rope_max_seqlen_kv = packed_seq_params.max_seqlen_kv
            rope_freqs_max_seqlen = (
                max(rope_max_seqlen_q, rope_max_seqlen_kv)
                if rope_max_seqlen_q is not None and rope_max_seqlen_kv is not None
                else None
            )
        else:
            cu_seqlens_q = cu_seqlens_kv = None
            rope_freqs_max_seqlen = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        q_compressed, kv_combined = self._qkv_down_projection(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if get_pg_size(self.tp_group) > 1 and self.config.sequence_parallel:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=self.tp_group)

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = apply_module(self.q_layernorm)(q_compressed)

        kv_compressed = apply_module(self.kv_layernorm)(kv_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply_for_cached_latent_kv(
            q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
        ):
            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            q_no_pe, q_pos_emb = torch.split(
                q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
            )

            # Dynamic batching: use inference context methods
            q_pos_emb = inference_context.apply_rotary_emb_query(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens_q=cu_seqlens_q,
                cp_group=self.pg_collection.cp,
                mscale=mscale,
            )
            # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = inference_context.apply_rotary_emb_key(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cp_group=self.pg_collection.cp,
                mscale=mscale,
            )

            # Create KV cache entry. It will the be the key vector in cache mla latents path
            k_pos_emb_squeezed = k_pos_emb.squeeze(1)
            kv_cached = torch.cat([kv_compressed, k_pos_emb_squeezed], dim=-1)

            # Flag for whether to use absorption. We only use absorption
            # when caching the latents and in decode-only mode
            use_absorption = (
                self.config.cache_mla_latents
                and inference_context
                and inference_context.is_decode_only()
            )
            # Compute query components. Multiply by up k if absorbing
            q_content = (
                torch.einsum("sbhd,hdk->sbhk", q_no_pe, self.up_k_weight)
                if use_absorption
                else q_no_pe
            )
            # Query: content + original positional (latent_dim + pos_dim)
            query = torch.cat([q_content, q_pos_emb], dim=-1)

            key = kv_cached
            value = None

            query = query.contiguous()
            key = key.contiguous()

            return query, key, value

        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            use_fused_q_uproj = (
                self._use_fused_q_uproj
                and self.config.apply_rope_fusion
                and self.config.q_lora_rank is not None
                and q_compressed.ndim == 3
                and self.pg_collection.cp.size() == 1
            )

            if use_fused_q_uproj:
                b = q_compressed.shape[1]

                # With sequence parallelism the input is sequence-split; gather to full
                # sequence before the fused kernel, which owns the norm.
                if self.config.sequence_parallel and get_pg_size(self.tp_group) > 1:
                    q_compressed = gather_from_sequence_parallel_region(
                        q_compressed, group=self.tp_group
                    )
                s = q_compressed.shape[0]

                query = FusedMLAQUpProjFunction.apply(
                    q_compressed,
                    self.linear_q_up_proj.layer_norm_weight,
                    self.linear_q_up_proj.weight,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.linear_q_up_proj.wgrad_store,
                    getattr(self.linear_q_up_proj, "fuse_wgrad_accumulation", False),
                    self.num_attention_heads_per_partition,
                    self.q_head_dim,
                    self.config.qk_head_dim,
                    self.config.qk_pos_emb_head_dim,
                    s,
                    b,
                    self.tp_group,
                    self.config.sequence_parallel,
                    self.config.layernorm_epsilon,
                    self.config.normalization,
                    self.config.layernorm_zero_centered_gamma,
                )
            else:
                if self.config.q_lora_rank is not None:
                    # q_compressed: [num_tokens, q_lora_rank]
                    # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                    q, _ = self.linear_q_up_proj(
                        _up_proj_input(self.linear_q_up_proj, q_compressed)
                    )
                else:
                    # q_compressed: [num_tokens, hidden_size]
                    # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                    q, _ = self.linear_q_proj(q_compressed)

                # q: [num_tokens, n, q_head_dim]
                q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            # kv: [num_tokens, n * (qk_head_dim + v_head_dim)]
            kv, _ = self.linear_kv_up_proj(_up_proj_input(self.linear_kv_up_proj, kv_compressed))

            # kv: [num_tokens, n, (qk_head_dim + v_head_dim)]
            kv = kv.view(
                *kv.size()[:-1],
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            # todo add assert about fusions and caching
            if self.config.apply_rope_fusion:
                cp_rank = self.pg_collection.cp.rank()
                cp_size = self.pg_collection.cp.size()
                if not use_fused_q_uproj:
                    query = fused_apply_mla_rope_for_q(
                        q,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        cp_rank,
                        cp_size,
                    )
                key, value = fused_apply_mla_rope_for_kv(
                    kv,
                    k_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_pos_emb_head_dim,
                    self.config.qk_head_dim,
                    self.config.v_head_dim,
                    cu_seqlens_kv,
                    cp_rank,
                    cp_size,
                )
                if use_fused_q_uproj:
                    # Quantize K/V to MXFP8, then transpose+swizzle scales for all three
                    key, value = _QuantizeKVForFusedAttn.apply(key, value)
                    mxfp8_transpose_swizzle([query, key, value], "sbhd")
            else:
                q_len = q.size()[0]
                if inference_context is not None:
                    # add offset to the sequence start for inference
                    sequence_start = inference_context.sequence_len_offset
                    sequence_end = sequence_start + q_len
                    rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
                elif packed_seq_params is None or self.config.context_parallel_size == 1:
                    # Shorten rotary_pos_emb to the sequence length when inference_params
                    # is not provided. This makes sure we can run forward directly with
                    # any sequence length. During training, the sequence length is always
                    # the full rotary_pos_emb length, except for sequence packing + CP.
                    # When sequence packing and context parallel are both enabled, the
                    # position embedding will not split rotary_pos_emb, so it may exceed
                    # the sequence length on this CP rank, but we need the full rotary_pos_emb
                    # to cover the full sequence, so we do not shorten it here.
                    rotary_pos_emb = rotary_pos_emb[0:q_len]

                # q_no_pe: [num_tokens, n, qk_head_dim]
                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_no_pe, q_pos_emb = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )

                # k_no_pe: [num_tokens, n, qk_head_dim]
                # value: [num_tokens, n, v_head_dim]
                k_no_pe, value = torch.split(
                    kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
                )

                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_pos_emb = apply_rotary_pos_emb(
                    q_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                    max_seqlen=rope_freqs_max_seqlen,
                )
                # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
                k_pos_emb = apply_rotary_pos_emb(
                    k_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                    max_seqlen=rope_freqs_max_seqlen,
                )

                # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
                query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

                # key: [num_tokens, n, (qk_head_dim + v_head_dim)]
                if k_pos_emb.ndim == 4:
                    k_pos_emb_per_head = k_pos_emb.expand(
                        -1, -1, self.num_attention_heads_per_partition, -1
                    )
                else:
                    assert k_pos_emb.ndim == 3
                    k_pos_emb_per_head = k_pos_emb.expand(
                        -1, self.num_attention_heads_per_partition, -1
                    )
                key = torch.cat([k_no_pe, k_pos_emb_per_head], dim=-1)

            if not use_fused_q_uproj:
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()

            if self.config.enable_chunkpipe:

                def _chunk_boundary():
                    if self.config.sft_chunkpipe_mode:
                        chunk_idx = self.config.chunkpipe_chunk_idx_in_group
                        is_last = chunk_idx >= self.config.chunkpipe_current_group_size - 1
                    else:
                        chunk_idx = (
                            self.config.chunkpipe_backward_microbatch
                            % self.num_chunks_per_seq
                        )
                        is_last = chunk_idx == self.num_chunks_per_seq - 1
                    return chunk_idx, is_last

                def kv_compressed_hook_fn(grad):
                    chunk_idx, is_last = _chunk_boundary()
                    if is_last:
                        return grad
                    if chunk_idx not in self.kv_compressed_cache_grad:
                        raise RuntimeError(
                            f"[ChunkPipe] kv_compressed_cache_grad[{chunk_idx}] is "
                            f"unavailable; layer={self.layer_number}, "
                            f"bwd_mb={self.config.chunkpipe_backward_microbatch}"
                        )
                    cached_grad = self.kv_compressed_cache_grad.pop(chunk_idx)
                    return grad + _align_chunkpipe_cache_grad(grad, cached_grad)

                def key_pos_emb_hook_fn(grad):
                    chunk_idx, is_last = _chunk_boundary()
                    if is_last:
                        return grad
                    if chunk_idx not in self.key_pos_emb_cache_grad:
                        raise RuntimeError(
                            f"[ChunkPipe] key_pos_emb_cache_grad[{chunk_idx}] is "
                            f"unavailable; layer={self.layer_number}, "
                            f"bwd_mb={self.config.chunkpipe_backward_microbatch}"
                        )
                    cached_grad = self.key_pos_emb_cache_grad.pop(chunk_idx)
                    return grad + _align_chunkpipe_cache_grad(grad, cached_grad)

                if self.is_enable_grad_chunkpipe():
                    kv_compressed.register_hook(kv_compressed_hook_fn)
                    k_pos_emb.register_hook(key_pos_emb_hook_fn)
                self.append_chunk_key_value_cache_mla(kv_compressed, k_pos_emb)

            if self.padding_v_head_dim:
                value = F.pad(value, [0, self.q_head_dim - self.config.v_head_dim])

            return query, key, value

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            if self.cache_mla_latents:
                assert (
                    inference_context and not inference_context.is_static_batching()
                ), "Caching MLA latents only works with dynamic backend inference"
                query, key, value = qkv_up_proj_and_rope_apply_for_cached_latent_kv(
                    q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
                )
            else:
                query, key, value = qkv_up_proj_and_rope_apply(
                    q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
                )

        return query, key, value, q_compressed, kv_compressed

    def uncompress_kv_from_cache(self, kv_cached):
        """
        Take a compressed kv and uncompress them
        """
        kv_compressed, k_pos_emb = torch.split(
            kv_cached, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )

        # Seperated out the norm and linear
        kv, _ = self.linear_kv_up_proj_linear(kv_compressed)

        kv = kv.view(
            *kv.size()[:-1],
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )

        k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

        # Add head dimension
        k_pos_emb = k_pos_emb.unsqueeze(-2)
        k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)

        key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
        return key, value

    def prepare_for_absorption(self):
        """Prepare the model for absorption optimization in MLA (Multi-Latent Attention).

        This method sets up the necessary components for the absorption technique, which
        optimizes memory during inference by caching compressed KV latents instead
        of full KV states. The absorption technique allows efficient decode-only operations
        by pre-computing certain matrix multiplications.

        Note (Peter): Right now we are not doing true absorption. We will add this support
        at a later time.

        The method performs the following operations:
        1. Splits the fused layernorm + linear layer (linear_kv_up_proj) into separate
        components.
        2. Extracts and stores the up-projection weights for K and V separately, which
        are used during the absorption process
        3. Replaces the identity kv_layernorm with the actual layernorm from the split
        4. Stores the linear component separately for uncompressing KV cache during
        prefill/mixed stages

        This is a one-time setup that should only be called once at initialization when
        cache_mla_latents is enabled.
        """
        # We should only have to call to set once at start
        if not hasattr(self, "up_k_weight"):
            with torch.no_grad():
                linear_kv_up_proj_norm, linear_kv_up_proj_linear = (
                    split_te_layernorm_column_parallel_linear(
                        self.linear_kv_up_proj, self.config, None, self.linear_kv_up_proj.tp_group
                    )
                )

                # Note: When caching latents we overide the kv_layernorm
                # which was an identity before because in the is path
                # we unfused the linear_kv_up_proj
                self.kv_layernorm = linear_kv_up_proj_norm

                # This is used in absorption when we are
                # uncompressing the KV cache in prefill/mixed stages
                self.linear_kv_up_proj_linear = linear_kv_up_proj_linear

                kv_up_weight = (
                    self.linear_kv_up_proj.weight
                )  # [num_heads * (qk_head_dim + v_head_dim), kv_lora_rank]
                kv_up_weight = kv_up_weight.view(
                    self.num_attention_heads_per_partition,
                    self.config.qk_head_dim + self.config.v_head_dim,
                    self.config.kv_lora_rank,
                )
                # Split into K and V up-projection weights. These are used for absorption
                self.up_k_weight = kv_up_weight[
                    :, : self.config.qk_head_dim, :
                ]  # [num_heads, qk_head_dim, kv_lora_rank]
                self.up_v_weight = kv_up_weight[
                    :, self.config.qk_head_dim :, :
                ]  # [num_heads, v_head_dim, kv_lora_rank]

                # We delete the original linear_kv_up_proj as we do not
                # need it for the absorbed path.
                del self.linear_kv_up_proj

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation"""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_output_proj()

    def _backward_kv_proj(self):
        """Computes weight gradients of KV projection layers"""
        self.linear_kv_up_proj.backward_dw()
        self.linear_kv_down_proj.backward_dw()

    def _backward_q_proj(self):
        """Computes weight gradients of Q projection layers"""
        if self.config.q_lora_rank is None:
            self.linear_q_proj.backward_dw()
        else:
            self.linear_q_down_proj.backward_dw()
            self.linear_q_up_proj.backward_dw()

    def _backward_output_proj(self):
        """Computes weight gradients of output projection layer"""
        self.linear_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8/fp4."""
        if self.config.q_lora_rank is not None:
            set_save_original_input(self.linear_q_down_proj)
        set_save_original_input(self.linear_kv_down_proj)

    def clip_qk(self):
        """
        QK Clipping is a technique to clip the query and key attention logits to prevent the
        attention logits from exploding. Per MuonClip usage, we update the weight by calling this
        function after Muon optimizer step.
        """

        if not self.config.qk_clip:
            raise ValueError("qk_clip option needs to be enabled")

        if self.core_attention.current_max_attn_logits is None:
            raise ValueError("current_max_attn_logits is None")

        # Check if we're in absorption mode
        if self.cache_mla_latents and not hasattr(self, 'linear_kv_up_proj'):
            raise ValueError(
                "qk_clip is not supported when cache_mla_latents is enabled and absorption is "
                "active. The linear_kv_up_proj layer has been deleted during absorption "
                "preparation."
            )

        assert self.core_attention.current_max_attn_logits.shape == (
            self.num_attention_heads_per_partition,
        ), f"current_max_attn_logits shape is not ({self.num_attention_heads_per_partition}, ) \
                    but {self.core_attention.current_max_attn_logits.shape}"

        # only update the weight if any head has
        # current_max_attn_logits > qk_clip_threshold
        if torch.any(self.core_attention.current_max_attn_logits > self.config.qk_clip_threshold):
            # Use num_attention_heads_per_partition for tensor parallel scenarios

            # qk_clip_balancing_eta (n, 1, 1)
            assert self.core_attention.current_max_attn_logits.shape == (
                self.num_attention_heads_per_partition,
            ), f"current_max_attn_logits shape is not ({self.num_attention_heads_per_partition},) \
                but {self.core_attention.current_max_attn_logits.shape}"
            self.qk_clip_balancing_eta = torch.clamp(
                self.config.qk_clip_threshold / self.core_attention.current_max_attn_logits, max=1.0
            ).view(self.num_attention_heads_per_partition, 1, 1)
            assert torch.all(self.qk_clip_balancing_eta <= 1.0)

            # Update q side weight, keep qk_pos_emb_head_dim side weight unchanged
            if self.config.q_lora_rank is None:
                q_proj_weight = self.linear_q_proj.weight
            else:
                q_proj_weight = self.linear_q_up_proj.weight

            # Handle different weight access patterns (main_param vs direct access)
            if hasattr(q_proj_weight, 'main_param'):
                q_proj_weight.main_param.data.copy_(
                    self._clip_q_proj_weight(q_proj_weight.main_param.data)
                )
            q_proj_weight.data.copy_(self._clip_q_proj_weight(q_proj_weight.data))

            # Update k side weight, keep v side weight unchanged
            kv_proj_weight = self.linear_kv_up_proj.weight

            # Handle different weight access patterns
            if hasattr(kv_proj_weight, 'main_param'):
                kv_proj_weight.main_param.data.copy_(
                    self._clip_kv_proj_weight(kv_proj_weight.main_param.data)
                )
            kv_proj_weight.data.copy_(self._clip_kv_proj_weight(kv_proj_weight.data))

        # reset current_max_attn_logits
        self.core_attention.current_max_attn_logits = None

    def _clip_q_proj_weight(self, weight):
        """Clip q_proj_weight"""
        # Reshape to (n, a + b, -1)
        weight_reshaped = weight.view(
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.qk_pos_emb_head_dim,
            -1,
        )

        # Split into qk_head_dim and qk_pos_emb_head_dim parts: (n, a, -1) and (n, b, -1)
        weight_q_nope = weight_reshaped[:, : self.config.qk_head_dim, :]
        weight_q_pe = weight_reshaped[:, self.config.qk_head_dim :, :]

        # Clipping
        weight_q_nope.mul_(torch.pow(self.qk_clip_balancing_eta, self.config.qk_clip_alpha))
        weight_q_pe.mul_(self.qk_clip_balancing_eta)

        # Concatenate back and reshape to original shape
        weight_q_updated = torch.cat([weight_q_nope, weight_q_pe], dim=1)
        weight_q_updated = weight_q_updated.view(
            self.num_attention_heads_per_partition
            * (self.config.qk_head_dim + self.config.qk_pos_emb_head_dim),
            -1,
        )

        return weight_q_updated

    def _clip_kv_proj_weight(self, weight):
        """Clip kv_proj_weight"""
        # shape: (n, qk_head_dim + v_head_dim, kv_lora_rank)
        weight_reshaped = weight.view(
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
            -1,
        )

        # Split into qk_head_dim and v_head_dim parts: (n, a, -1) and (n, b, -1)
        weight_k = weight_reshaped[:, : self.config.qk_head_dim, :]
        weight_v = weight_reshaped[:, self.config.qk_head_dim :, :]

        # Clipping
        weight_k.mul_(torch.pow(self.qk_clip_balancing_eta, 1 - self.config.qk_clip_alpha))

        # Concatenate back and reshape to original shape
        weight_kv_updated = torch.cat([weight_k, weight_v], dim=1)
        weight_kv_updated = weight_kv_updated.view(
            self.num_attention_heads_per_partition
            * (self.config.qk_head_dim + self.config.v_head_dim),
            -1,
        )

        return weight_kv_updated


class FusedMLASelfAttention(MLASelfAttention):
    """MLA self-attention with fused q/kv down projection."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        pp_layer_offset: Optional[int] = None,
        name: str | None = None,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        MultiLatentAttention.__init__(
            self,
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

        assert self.config.q_lora_rank is not None, (
            "FusedMLASelfAttention requires q_lora_rank to be set; "
            "fallback to MLASelfAttention for q_lora_rank=None."
        )
        # Resolve which linear class to use for Q and KV up projections,
        # based on QK-norm selection.
        layer_classes = self._resolve_qk_norm_config(submodules)

        qkv_down_proj_kwargs = {}
        if submodules.linear_qkv_down_proj in [TELinear]:
            qkv_down_proj_kwargs['parallel_mode'] = 'duplicated'
        elif submodules.linear_qkv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
            TELayerNormColumnParallelLinear,
        ]:
            qkv_down_proj_kwargs['gather_output'] = False
        else:
            raise ValueError(f"Unsupported linear_qkv_down_proj: {submodules.linear_qkv_down_proj}")

        self.linear_qkv_down_proj = build_module(
            submodules.linear_qkv_down_proj,
            self.config.hidden_size,
            self.config.q_lora_rank + self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv_down_proj',
            skip_weight_param_allocation=False,
            tp_group=(
                pg_collection.tp
                if qkv_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                else None
            ),
            name=(name + ".linear_qkv_down_proj") if name is not None else None,
            **qkv_down_proj_kwargs,
        )

        self.linear_q_up_proj = build_module(
            layer_classes["linear_q_up_proj"],
            self.config.q_lora_rank,
            self.config.num_attention_heads * self.q_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='q_up_proj',
            tp_group=pg_collection.tp,
            name=(name + ".linear_q_up_proj") if name is not None else None,
        )

        self.linear_kv_up_proj = build_module(
            layer_classes["linear_kv_up_proj"],
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
            tp_group=pg_collection.tp,
            name=(name + ".linear_kv_up_proj") if name is not None else None,
        )

        self.q_layernorm = layer_classes["q_layernorm"](
            hidden_size=self.config.q_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.kv_layernorm = layer_classes["kv_layernorm"](
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def _qkv_down_projection(self, hidden_states):
        """Fused q/kv down projection path."""
        qkv, _ = self.linear_qkv_down_proj(hidden_states)

        q_split = self.config.q_lora_rank
        kv_split = self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
        tp_size = get_pg_size(self.tp_group)
        is_tensor_parallel = tp_size > 1

        if is_tensor_parallel:
            assert q_split % tp_size == 0, (
                "q_lora_rank must be divisible by tensor model parallel size when "
                "using MLA down projection fusion"
            )
            assert kv_split % tp_size == 0, (
                "kv_lora_rank + qk_pos_emb_head_dim must be divisible by tensor model "
                "parallel size when using MLA down projection fusion"
            )
            q_split //= tp_size
            kv_split //= tp_size

        q_compressed, kv_combined = torch.split(qkv, [q_split, kv_split], dim=-1)

        if is_tensor_parallel:
            q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
            if self.config.sequence_parallel:
                q_compressed = scatter_to_sequence_parallel_region(q_compressed)

        return q_compressed, kv_combined

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation."""
        self.linear_kv_up_proj.backward_dw()
        self.linear_qkv_down_proj.backward_dw()
        self.linear_q_up_proj.backward_dw()
        self._backward_output_proj()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8/fp4."""
        set_save_original_input(self.linear_qkv_down_proj)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None):
        """Return a sharded state dict compatible with pre-fusion checkpoints."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        def _clone_sharded_object_with_key(obj: ShardedObject, new_key: str) -> ShardedObject:
            return ShardedObject(
                key=new_key,
                data=obj.data,
                global_shape=obj.global_shape,
                global_offset=obj.global_offset,
                replica_id=obj.replica_id,
            )

        fused_prefix = f"{prefix}linear_qkv_down_proj."

        fused_extra_keys = [
            k
            for k in sharded_state_dict.keys()
            if k.startswith(fused_prefix) and "_extra_state" in k
        ]
        for fused_extra_key in fused_extra_keys:
            suffix = fused_extra_key[len(fused_prefix) :]
            q_extra_key = f"{prefix}linear_q_down_proj.{suffix}"
            kv_extra_key = f"{prefix}linear_kv_down_proj.{suffix}"
            fused_obj = sharded_state_dict.get(fused_extra_key)
            if isinstance(fused_obj, ShardedObject):
                sharded_state_dict[q_extra_key] = _clone_sharded_object_with_key(
                    fused_obj, q_extra_key
                )
                sharded_state_dict[kv_extra_key] = _clone_sharded_object_with_key(
                    fused_obj, kv_extra_key
                )
            elif fused_obj is not None:
                sharded_state_dict[q_extra_key] = fused_obj
                sharded_state_dict[kv_extra_key] = fused_obj

        for key in list(sharded_state_dict.keys()):
            suffix = key[len(fused_prefix) :] if key.startswith(fused_prefix) else ""
            if key.startswith(fused_prefix) and not suffix.startswith("layer_norm_"):
                del sharded_state_dict[key]

        fused_weight = self.linear_qkv_down_proj.weight
        total_out = (
            self.config.q_lora_rank + self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
        )
        tp_size = get_pg_size(self.tp_group)

        if fused_weight.size(0) == total_out:
            q_split = self.config.q_lora_rank
            kv_split = self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
        else:
            assert (
                self.config.q_lora_rank % tp_size == 0
            ), "q_lora_rank must be divisible by tensor-parallel size"
            assert (
                self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
            ) % tp_size == 0, (
                "kv_lora_rank + qk_pos_emb_head_dim must be divisible by tensor-parallel size"
            )
            q_split = self.config.q_lora_rank // tp_size
            kv_split = (self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim) // tp_size

        if q_split + kv_split != fused_weight.size(0):
            raise ValueError(
                "Unexpected fused qkv-down weight shape: "
                f"got {tuple(fused_weight.size())}, expected dim0 {q_split + kv_split}"
            )

        q_weight, kv_weight = torch.split(fused_weight, [q_split, kv_split], dim=0)

        q_key = f"{prefix}linear_q_down_proj.weight"
        kv_key = f"{prefix}linear_kv_down_proj.weight"

        sharded_state_dict[q_key] = make_tp_sharded_tensor_for_checkpoint(
            tensor=q_weight, key=q_key, tp_axis=0, prepend_offsets=sharded_offsets
        )
        sharded_state_dict[kv_key] = make_tp_sharded_tensor_for_checkpoint(
            tensor=kv_weight, key=kv_key, tp_axis=0, prepend_offsets=sharded_offsets
        )

        return sharded_state_dict

    def _synthetic_state_dict_key_suffixes(self):
        """Return source checkpoint keys used to locate this module in a state dict."""
        return ("linear_q_down_proj.weight",)

    def _synthesize_fused_qkv_down_weight(self, state_dict, prefix):
        """Materialize fused qkv-down weight from old separate q/kv checkpoint keys."""
        q_key = f"{prefix}linear_q_down_proj.weight"
        kv_key = f"{prefix}linear_kv_down_proj.weight"
        fused_key = f"{prefix}linear_qkv_down_proj.weight"

        def _as_tensor(x):
            return x.data if hasattr(x, 'data') else x

        if fused_key not in state_dict and q_key in state_dict and kv_key in state_dict:
            q_weight = _as_tensor(state_dict[q_key])
            kv_weight = _as_tensor(state_dict[kv_key])
            state_dict[fused_key] = torch.cat([q_weight, kv_weight], dim=0)
            del state_dict[q_key]
            del state_dict[kv_key]
            state_dict.pop(f"{prefix}linear_q_down_proj.bias", None)
            state_dict.pop(f"{prefix}linear_kv_down_proj.bias", None)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Load state dict with automatic unfused->fused conversion."""
        self._synthesize_fused_qkv_down_weight(state_dict, prefix)

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
