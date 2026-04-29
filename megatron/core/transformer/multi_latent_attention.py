# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
from dataclasses import dataclass
from typing import NoReturn, Optional, Union

import torch
import torch.nn.functional as F

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    _yarn_get_mscale,
    apply_rotary_pos_emb,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.custom_layers.transformer_engine import (
    split_te_layernorm_column_parallel_linear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import deprecate_inference_params, is_te_min_version

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        fused_apply_mla_rope_for_kv,
        fused_apply_mla_rope_for_q,
    )
except:
    fused_apply_mla_rope_for_kv = None
    fused_apply_mla_rope_for_q = None


try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear,
        set_save_original_input,
    )
    from megatron.core.post_training.modelopt.layers import Linear

    HAVE_TE = True
except ImportError:
    TEColumnParallelLinear, TELinear, Linear, set_save_original_input = None, None, None, None
    HAVE_TE = False


@dataclass
class MLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None


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
        pg_collection: ProcessGroupCollection = None,       
    ) -> None:

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
        )

        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads

        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        # Overwrite the base class kv shape to support MLA inference
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        self.recompute_up_proj = (
            self.config.recompute_granularity == 'selective'
            and "mla_up_proj" in self.config.recompute_modules
        )
        self.qkv_up_checkpoint = None

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
        
        self.v_channels = self.config.v_head_dim
        self.padding_v_head_dim = False

        if self.config.padding_v_to_qk_dim and self.q_head_dim > self.config.v_head_dim:
            self.v_channels = self.q_head_dim
            self.padding_v_head_dim = True
        
        if self.config.enable_chunkpipe:
            self.v_channels = self.q_head_dim
            self.padding_v_head_dim = True

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
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

        if (
            HAVE_TE
            and self.config.fp8
            and self.config.fp8_recipe != 'delayed'
            and is_te_min_version("2.6.0dev0")
            and isinstance(self.linear_proj, TELinear)
        ):
            # For fp8 training, the output of the fused core_attn is saved by itself, and
            # linear_proj also saves the quantized tensor of this output. Here we set the
            # linear_proj to save the original input tensors to avoid the extra memory usage of
            # the quantized tensor.
            set_save_original_input(self.linear_proj)

        # Initialize chunkpipe KV cache configuration
        if self.config.enable_chunkpipe:
            self.setup_chunkpipe_kv_cache_config()

    def init_chunk_key_value_cache_mla(self):
        """Initialize the chunk-based key-value cache for chunkpipe functionality in MLA."""

        # Initialize cache tensors with proper device and dtype
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        total_cache_tokens = self.kv_cache_chunk_size * self.config.chunksize
        if self.config.sft_chunkpipe_mode:
            # SFT packed sequence: no batch dimension
            kv_cache_shape = (total_cache_tokens // parallel_state.get_tensor_model_parallel_world_size(),
                              self.config.kv_lora_rank)
            k_pos_emb_cache_shape = (total_cache_tokens,
                                     self.num_attention_heads_per_partition, self.config.qk_pos_emb_head_dim)
        else:
            # Pretrain: with batch dimension
            kv_cache_shape = (total_cache_tokens // parallel_state.get_tensor_model_parallel_world_size(),
                              self.config.micro_batch_size, self.config.kv_lora_rank)
            k_pos_emb_cache_shape = (total_cache_tokens, self.config.micro_batch_size,
                                     self.num_attention_heads_per_partition, self.config.qk_pos_emb_head_dim)
        
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
        
        # Store compressed KV and positional embeddings
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        kv_cache_indices = torch.arange(self.config.chunksize // tp_size) + \
            (chunk_id * self.config.chunksize // tp_size)
        pos_cache_indices = torch.arange(self.config.chunksize) + (chunk_id * self.config.chunksize)

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

    def recover_key_value_up_proj_tensors(self, normed_kv_compressed, k_pos_emb):
        """Recover full key and value tensors from compressed representations.
        
        This function performs the up-projection operation to recover full key and 
        value tensors from their compressed latent representations. This is needed
        for chunkpipe when reconstructing cached chunks.
        
        Args:
            normed_kv_compressed (Tensor): Normalized compressed KV tensor with shape 
                [seq_len / TP, batch_size, kv_lora_rank]
            k_pos_emb (Tensor): Key positional embeddings with shape 
                [seq_len, batch_size, 1, qk_pos_emb_head_dim]
        
        Returns:
            tuple:
                recovered_key: Full key tensor with shape 
                    [seq_len, batch_size, num_heads, q_head_dim]
                recovered_value: Full value tensor with shape 
                    [seq_len, batch_size, num_heads, v_head_dim]
        """
        # Project compressed KV through up-projection
        kv, _ = self.linear_kv_up_proj(normed_kv_compressed)

        # Reshape and split into key and value components
        kv = kv.view(
            *kv.size()[:-1],
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )

        k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

        # Combine with positional embeddings (k_pos_emb is already [seq, batch, heads, dim])
        key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

        # Pad value dimension if needed for flash attention compatibility
        if self.padding_v_head_dim:
            value = F.pad(value, [0, self.q_head_dim - self.config.v_head_dim])

        key = key.contiguous()
        value = value.contiguous()

        return key, value

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

        # Calculate total sequence length after concatenation
        total_concatenated_tokens = (current_chunk_idx + 1) * self.config.chunksize
        
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
            
            # Calculate tensor parallel indices for retrieving cached data
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            kv_indices = torch.arange(self.config.chunksize // tp_size) + \
                (cache_chunk_idx * self.config.chunksize // tp_size)
            pos_indices = torch.arange(self.config.chunksize) + (cache_chunk_idx * self.config.chunksize)

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

            # Concatenate along sequence dimension
            concatenated_key[current_pos : current_pos + self.config.chunksize] = \
                cached_key
            concatenated_value[current_pos : current_pos + self.config.chunksize] = \
                cached_value
            current_pos += self.config.chunksize

        # Add the current chunk's keys and values to the concatenated result
        concatenated_key[current_pos : current_pos + self.config.chunksize] = \
            curr_key
        concatenated_value[current_pos : current_pos + self.config.chunksize] = \
            curr_value  
        
        # Adjust attention mask for autoregressive (causal) attention
        adjusted_mask = None
        if attention_mask is not None:
            chunksize = self.config.chunksize
            # Create a mask that prevents attending to future tokens
            mask_org = torch.ones(chunksize, total_concatenated_tokens, dtype=torch.bool,
                device=self.kv_compressed_cache.device)
            mask_index = current_chunk_idx * chunksize + 1
            # Apply upper triangular masking (applying in-place operation triu_ to avoid doubling memory)
            mask_org.triu_(diagonal=mask_index)
            # Reshape mask to match attention mechanism requirements
            # Using .view creates a new tensor without allocating additional memory
            adjusted_mask = mask_org.view(1, 1, chunksize, total_concatenated_tokens)

        return concatenated_key, concatenated_value, adjusted_mask

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for multi-latent attention"""
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
        query, key, value, q_compressed, kv_compressed = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        # cache key, value for chunkpipe
        if self.config.enable_chunkpipe:
            # need to concat all chunk keys & values belong to the same sequence
            key, value, attention_mask = \
                self.concat_cached_chunk_key_value_mla(attention_mask, key, value)

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()

        # Value is none during decode for absorption
        if value is not None:
            value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            if self.offload_core_attention and self.training:
                query = fine_grained_offloading_group_start(query, name="core_attn")

            if inference_context is None or inference_context.is_static_batching():
                extra_kwargs = {}
                if self.config.experimental_attention_variant == "dsa":
                    # For dsa we need to pass in the original hidden states and the compressed
                    # query representation.
                    extra_kwargs["x"] = hidden_states
                    extra_kwargs["qr"] = q_compressed
                with get_fine_grained_offloading_context(self.offload_core_attention):
                    # core_attn_out: [..., h, kv_low_rank]
                    if self.config.sft_chunkpipe_mode and self.config.chunkpipe_current_group_size > 1:
                        # Multi-chunk long sequence: Q(chunksize) < K/V(total_concatenated_tokens).
                        # TE requires "padding" in attn_mask_type for thd format, but Flash
                        # Attention rejects padding_causal for asymmetric Q/K lengths.
                        # Switch to sbhd by unsqueezing batch dim, use causal_bottom_right
                        # which Flash Attention natively supports for this decoder-style case.
                        core_attn_out = self.core_attention(
                            query.unsqueeze(1),
                            key.unsqueeze(1),
                            value.unsqueeze(1),
                            attention_mask,
                            packed_seq_params=None,
                            attn_mask_type=AttnMaskType.causal_bottom_right,
                            **extra_kwargs,
                        )
                    else:
                        core_attn_out = self.core_attention(
                            query,
                            key,
                            value,
                            attention_mask,
                            packed_seq_params=packed_seq_params,
                            attn_mask_type=attn_mask_type,
                            **extra_kwargs,
                        )
            elif self.cache_mla_latents:
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
                )
                # Only rearrange if not in absorption mode (Flash MLA handles format correctly)
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')
            if self.offload_core_attention and self.training:
                (core_attn_out,) = fine_grained_offloading_group_commit(
                    core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                )
        # We are doing absorption with cache mla latents and decode mode.
        if self.cache_mla_latents and inference_context.is_decode_only():
            # core_attn_out = self.self.up_v_layer(core_attn_out)
            core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
            core_attn_out = core_attn_out.contiguous()

            # Flatten back: [seq, batch, num_heads * v_head_dim]
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        # add for fa padding
        if self.padding_v_head_dim:
            _prefix = core_attn_out.shape[:-1]
            core_attn_out = core_attn_out.reshape(*_prefix, -1, self.v_channels)
            core_attn_out = core_attn_out[..., : self.config.v_head_dim].reshape(*_prefix, -1)

        if (packed_seq_params is not None and packed_seq_params.qkv_format == 'thd') or \
                (self.config.sft_chunkpipe_mode and self.config.chunkpipe_current_group_size > 1):
            # reshape to same output shape as unpacked case:
            #   thd path:  (t, np, hn)      -> (t, b=1, h=np*hn)
            #   sbhd path: (t, b=1, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum(sq_i); batch is a dummy dimension in both cases.
            #
            # Note: when sft_chunkpipe_mode and chunk_idx_in_group > 0, core_attention was
            # called with packed_seq_params=None (sbhd path) to work around TE's thd-format
            # constraint vs Flash Attention's causal_bottom_right requirement. The original
            # packed_seq_params is still thd (not modified), so the first branch of this
            # condition already covers the multi-chunk case; the second branch is kept as
            # explicit documentation of the sbhd output path.
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        if self.offload_attn_proj:
            core_attn_out = fine_grained_offloading_group_start(core_attn_out, name="attn_proj")
        with get_fine_grained_offloading_context(self.offload_attn_proj):
            output, bias = self.linear_proj(core_attn_out)
        if self.offload_attn_proj:
            output, bias = fine_grained_offloading_group_commit(
                output, bias, name="attn_proj", forced_released_tensors=[core_attn_out]
            )

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
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        if self.config.q_lora_rank is None:
            # Not projecting query
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_proj',
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
                **q_down_proj_kwargs,
            )

            self.linear_q_up_proj = build_module(
                submodules.linear_q_up_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_up_proj',
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
            **kv_down_proj_kwargs,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
        )

        if self.config.q_lora_rank is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        # Initialize chunkpipe MLA KV cache
        if self.config.enable_chunkpipe:
            self.init_chunk_key_value_cache_mla()

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

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        # Calculate position embedding offset for chunkpipe
        pos_emb_offset = 0
        if self.config.enable_chunkpipe:
            if self.config.sft_chunkpipe_mode:
                # SFT: scheduler sets chunk_idx_in_group for both forward and backward
                ck_fwd_mic = self.config.chunkpipe_chunk_idx_in_group
            else:
                # Pretrain: derive from global counter (all groups same size)
                ck_fwd_mic = self.config.chunkpipe_forward_microbatch % self.config.chunk_num_per_seq
                if not self.config.chunkpipe_forward:
                    ck_fwd_mic = self.config.chunkpipe_backward_microbatch % self.config.chunk_num_per_seq
            pos_emb_offset = ck_fwd_mic * self.config.chunksize

        # rotary_pos_emb:[s, b, 1, 64]
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, offset=pos_emb_offset, packed_seq=packed_seq)
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rotary_seq_len, dtype=hidden_states.dtype, packed_seq=packed_seq
                )
                rotary_pos_emb = None
                assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
                assert (
                    fused_apply_mla_rope_for_q is not None
                    and fused_apply_mla_rope_for_kv is not None
                ), "Fused MLA RoPE apply is not imported successfully"
            else:
                rotary_pos_emb, mscale = \
                    self.rotary_pos_emb(rotary_seq_len, offset=pos_emb_offset, packed_seq=packed_seq)

        if packed_seq_params is not None:
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection
        # =========================================
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
            if (
                parallel_state.get_tensor_model_parallel_world_size() > 1
                and self.config.sequence_parallel
            ):
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

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
            q_compressed = self.q_layernorm(q_compressed)

        kv_compressed = self.kv_layernorm(kv_compressed)

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

            # kv: [num_tokens, n * (qk_head_dim + v_head_dim)]
            kv, _ = self.linear_kv_up_proj(kv_compressed)

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
                )
                # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
                k_pos_emb = apply_rotary_pos_emb(
                    k_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                )

                # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
                query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

                # key: [num_tokens, n, (qk_head_dim + v_head_dim)]
                if k_pos_emb.ndim == 4:
                    k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
                else:
                    assert k_pos_emb.ndim == 3
                    k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            if self.config.enable_chunkpipe:
                def kv_compressed_hook_fn(grad):
                    """
                    Hook function to combine compressed KV gradients of loss of subsequent chunk
                    with respect to that of current chunk.
                    """
                    if self.config.sft_chunkpipe_mode:
                        # SFT: use scheduler-provided chunk index
                        chunks_in_current_sequence = self.config.chunkpipe_chunk_idx_in_group
                        is_last = (chunks_in_current_sequence >= self.config.chunkpipe_current_group_size - 1)
                    else:
                        # Pretrain: derive from global counter
                        chunks_in_current_sequence = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
                        is_last = (chunks_in_current_sequence == self.num_chunks_per_seq - 1)
                    if is_last:
                        return grad
                    else:
                        grad_from_prev_chunk = self.kv_compressed_cache_grad.pop(chunks_in_current_sequence)
                        return grad + grad_from_prev_chunk
                
                def key_pos_emb_hook_fn(grad):
                    """
                    Hook function to combine gradient of loss of current chunk
                    with respect to key position embeddings of previous chunks.
                    """
                    if self.config.sft_chunkpipe_mode:
                        # SFT: use scheduler-provided chunk index
                        chunks_in_current_sequence = self.config.chunkpipe_chunk_idx_in_group
                        is_last = (chunks_in_current_sequence >= self.config.chunkpipe_current_group_size - 1)
                    else:
                        # Pretrain: derive from global counter
                        chunks_in_current_sequence = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
                        is_last = (chunks_in_current_sequence == self.num_chunks_per_seq - 1)
                    if is_last:
                        return grad
                    else:
                        grad_from_prev_chunk = self.key_pos_emb_cache_grad.pop(chunks_in_current_sequence)
                        return grad + grad_from_prev_chunk
             
                if self.is_enable_grad_chunkpipe():
                    kv_compressed.register_hook(kv_compressed_hook_fn)
                    k_pos_emb.register_hook(key_pos_emb_hook_fn)
                self.append_chunk_key_value_cache_mla(kv_compressed, k_pos_emb)

            if self.padding_v_head_dim:
                value = F.pad(value, [0, self.q_head_dim - self.config.v_head_dim])

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            return query, key, value

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=self.config.fp8)
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
        """Set the attention layer for recompute input_layernorm. Only needed for fp8."""
        from megatron.core.extensions.transformer_engine import set_save_original_input

        if self.config.q_lora_rank is not None:
            set_save_original_input(self.linear_q_down_proj)
        set_save_original_input(self.linear_kv_down_proj)
