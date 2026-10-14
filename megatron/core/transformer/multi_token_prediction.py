# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import warnings
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_set_last_layer,
)
from megatron.core.pipeline_parallel.utils import is_vp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.hyper_connection import learned_output_contract
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import (
    is_torch_min_version,
    make_tp_sharded_tensor_for_checkpoint,
    make_viewless_tensor,
)

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def tie_word_embeddings_state_dict(
    sharded_state_dict: ShardedStateDict, word_emb_weight: Tensor, word_emb_weight_key: str
) -> None:
    """tie the embedding of the mtp processing stage in a given sharded state dict.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with the weight to tie.
        word_emb_weight (Tensor): weight of the word embedding.
        word_emb_weight_key (str): key of the word embedding in the sharded state dict.

    Returns: None, acts in-place
    """
    mtp_word_emb_replica_id = (
        1,  # copy of embedding in pre processing stage
        0,
        parallel_state.get_data_parallel_rank(with_context_parallel=True),
    )
    assert word_emb_weight_key in sharded_state_dict
    del sharded_state_dict[word_emb_weight_key]
    sharded_state_dict[word_emb_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=word_emb_weight,
        key=word_emb_weight_key,
        replica_id=mtp_word_emb_replica_id,
        allow_shape_mismatch=True,
    )


def tie_output_layer_state_dict(
    sharded_state_dict: ShardedStateDict, output_layer_weight: Tensor, output_layer_weight_key: str
) -> None:
    """tie the output layer of the mtp processing stage in a given sharded state dict.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with the weight to tie.
        output_layer_weight (Tensor): weight of the output layer.
        output_layer_weight_key (str): key of the output layer in the sharded state dict.

    Returns: None, acts in-place
    """
    mtp_output_layer_replica_id = (
        1,  # copy of output layer in post processing stage
        0,
        parallel_state.get_data_parallel_rank(with_context_parallel=True),
    )
    assert output_layer_weight_key in sharded_state_dict
    del sharded_state_dict[output_layer_weight_key]
    sharded_state_dict[output_layer_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=output_layer_weight,
        key=output_layer_weight_key,
        replica_id=mtp_output_layer_replica_id,
        allow_shape_mismatch=True,
    )


def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
    """Roll the tensor input along the sequence dimension with Context Parallelism (CP) support.

    This function extends the original roll_tensor to support Context Parallelism, which allows
    MTP to work with CP > 1. When CP is enabled, the sequence dimension is split across CP ranks,
    and tensor rolling requires communication between adjacent CP ranks to properly handle the
    boundary conditions.

    For CP=1 (default behavior): Uses standard torch.roll with zero padding
    For CP>1: Splits tensor into chunks, performs rolling within each chunk, then exchanges
    boundary elements between adjacent CP ranks to maintain sequence continuity.

    For packed sequences: Respects sequence boundaries when rolling to avoid mixing tokens
    from different sequences.

    Args:
        tensor (Tensor): The input tensor to roll.
        shifts (int): The shift of the tensor (typically -1 for MTP).
        dims (int): The dimension to roll (typically -1 for sequence dimension).
        cp_group (ProcessGroup): The context parallelism process group. If None or size=1,
                               falls back to standard rolling behavior.
        packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
                                            If provided, respects sequence boundaries.
    Returns:
        tuple: (rolled_tensor, sum_of_rolled_tensor)
    """
    # Handle packed sequences cases
    if packed_seq_params is not None:
        return _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group)

    # Standard rolling behavior when CP is not enabled (cp_group is None or size=1)
    if cp_group is None or cp_group.size() == 1:
        rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
        rolled_tensor.select(dims, shifts).fill_(0)
        return rolled_tensor, rolled_tensor.sum()

    # CP-enabled rolling: Split tensor into chunks and handle boundary communication
    # This matches the batch splitting logic in get_batch_on_this_cp_rank() function
    tensor_list = tensor.chunk(2, dim=dims)
    rolled_tensor_list = []
    for i in range(len(tensor_list)):
        rolled_tensor_list.append(torch.roll(tensor_list[i], shifts=shifts, dims=dims))

    # Prepare tensors for communication between CP ranks
    # Each CP rank needs to send boundary elements to adjacent ranks
    tensor_send_list = []
    tensor_recv_list = []
    for i in range(len(rolled_tensor_list)):
        tensor_send_list.append(rolled_tensor_list[i].select(dims, shifts).contiguous())
        empty_tensor = torch.empty(
            tensor_send_list[i].shape,
            dtype=tensor_send_list[i].dtype,
            device=torch.cuda.current_device(),
        )
        tensor_recv_list.append(empty_tensor)

    # Get the global rank of next and prev process in the cp group
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    local_rank = torch.distributed.get_rank(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % len(global_ranks)]
    prev_rank = global_ranks[(local_rank - 1) % len(global_ranks)]

    # Start send and recv ops
    ops = []
    if local_rank != 0:
        req_send_first_part = torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank)
        ops.append(req_send_first_part)
        req_recv_second_part = torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank)
        ops.append(req_recv_second_part)
    else:
        # Inserted elements are set to be 0.0.
        tensor_recv_list[1] = 0
    if local_rank != len(global_ranks) - 1:
        req_recv_first_part = torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank)
        ops.append(req_recv_first_part)
        req_send_second_part = torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank)
        ops.append(req_send_second_part)
    else:
        # For the last CP rank, the removed elements of second part go into the first part
        tensor_recv_list[0] = tensor_send_list[1]

    # Wait for all communication operations to complete
    for op in ops:
        op.wait()

    # Splicing: Replace boundary elements with received elements from adjacent ranks
    # This ensures proper sequence continuity across CP boundaries
    index = [slice(None)] * rolled_tensor_list[0].dim()
    index[dims] = shifts
    for i in range(len(rolled_tensor_list)):
        rolled_tensor_list[i][tuple(index)] = tensor_recv_list[i]

    # Concatenate the processed chunks back into a single tensor
    rolled_tensor = torch.cat(rolled_tensor_list, dim=dims)

    return rolled_tensor, rolled_tensor.sum()


def _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group=None):
    """Roll tensor with packed sequence support.
    This function handles rolling for packed sequences by respecting sequence boundaries
    """

    # Notice: This is a naive implementation to test the correctness,
    # a better solution will only sync the boundary tokens once.
    assert (
        dims == -1 or dims == tensor.dim() - 1
    ), "Packed sequence roll only supports the last dimension."
    assert shifts == -1, "Packed sequence roll only supports a single-token left shift."
    cu_seqlens = packed_seq_params.cu_seqlens_q
    assert cu_seqlens is not None, "Packed sequence parameters must provide cu_seqlens_q."

    rolled_tensor = tensor.clone()

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        # CP disabled: roll each packed sequence independently within its boundaries
        for i in range(len(cu_seqlens) - 1):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_slice = tensor[..., start_idx:end_idx]
            rolled_seq = torch.roll(seq_slice, shifts=shifts, dims=dims)
            # Zero out the last position(s) that would cross sequence boundaries
            rolled_seq[..., shifts:] = 0
            rolled_tensor[..., start_idx:end_idx] = rolled_seq
        return rolled_tensor, rolled_tensor.sum()

    cp_partition_mode = getattr(packed_seq_params, 'cp_partition_mode', 'zigzag')
    if cp_partition_mode == 'contiguous':
        # The contiguous-CP local THD layout is derived from the padded
        # per-sequence lengths, so index with cu_seqlens_q_padded when present;
        # the unpadded cu_seqlens would produce wrong local boundaries.
        if getattr(packed_seq_params, 'cu_seqlens_q_padded', None) is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        rolled_tensor = _roll_tensor_packed_seq_contiguous_cp(tensor, dims, cu_seqlens, cp_group)
        return rolled_tensor, rolled_tensor.sum()

    # CP enabled (zigzag): each rank owns two chunks per sequence (front and mirrored tail).
    local_rank = torch.distributed.get_rank(group=cp_group)
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % cp_size]
    prev_rank = global_ranks[(local_rank - 1) % cp_size]

    # Iterate over each sequence individually
    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]

        # the idx has been multiplied by cp_size, need to divide it by cp_size to get the local idx
        local_start_idx = start_idx // cp_size
        local_end_idx = end_idx // cp_size
        tensor_slice = rolled_tensor[..., local_start_idx:local_end_idx].clone()

        # The following code is very similar as the code in roll_tensor function
        local_chunks = tensor_slice.chunk(2, dim=dims)
        rolled_chunks = [torch.roll(chunk, shifts=shifts, dims=dims) for chunk in local_chunks]

        tensor_send_list = []
        tensor_recv_list = []
        for chunk in rolled_chunks:
            boundary = chunk.select(dims, shifts).contiguous().clone()
            tensor_send_list.append(boundary)
            tensor_recv_list.append(torch.empty_like(boundary))

        ops = []
        if local_rank != 0:
            ops.append(torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank))
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank))
        else:
            tensor_recv_list[1].zero_()

        if local_rank != cp_size - 1:
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank))
            ops.append(torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank))
        else:
            tensor_recv_list[0].copy_(tensor_send_list[1])

        for op in ops:
            op.wait()

        index = [slice(None)] * rolled_chunks[0].dim()
        index[dims] = shifts
        for chunk, recv in zip(rolled_chunks, tensor_recv_list):
            chunk[tuple(index)] = recv

        seq_result = torch.cat(rolled_chunks, dim=dims)

        # update the rolled tensor
        rolled_tensor[..., local_start_idx:local_end_idx] = seq_result

    return rolled_tensor, rolled_tensor.sum()


def _roll_tensor_packed_seq_contiguous_cp(tensor, dims, cu_seqlens, cp_group):
    """Roll a contiguous-CP THD shard without crossing packed sequence boundaries.

    Ported from Megatron-LM. Rank r owns global rows
    [r * local_seq_len, (r + 1) * local_seq_len); a left shift pulls the first
    row of the next rank into this rank's last position, except at packed
    sequence boundaries where the rolled-in value is zeroed.
    """
    local_seq_len = tensor.size(dims)
    rolled_tensor = torch.roll(tensor, shifts=-1, dims=dims)
    if local_seq_len == 0:
        return rolled_tensor

    cp_size = cp_group.size()
    local_rank = torch.distributed.get_rank(group=cp_group)
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)

    cu = cu_seqlens.to(device=tensor.device, dtype=torch.long)
    if cu.numel() > 1:
        nonduplicate_boundaries = torch.ones(cu.numel(), device=cu.device, dtype=torch.bool)
        nonduplicate_boundaries[1:] = cu[1:] != cu[:-1]
        cu = cu[nonduplicate_boundaries]
    if cu.numel() <= 1:
        rolled_tensor.zero_()
        return rolled_tensor

    global_start = local_rank * local_seq_len
    global_positions = global_start + torch.arange(local_seq_len, device=tensor.device)
    seq_idx = torch.bucketize(global_positions, cu[1:], right=True).clamp(max=cu.numel() - 2)
    seq_ends = cu[1:][seq_idx]
    valid_next = (global_positions < cu[-1]) & (global_positions + 1 < seq_ends)

    invalid_next = ~valid_next
    rolled_tensor[..., invalid_next] = 0

    recv_next_first = torch.empty_like(tensor.select(dims, 0))
    ops = []
    if local_rank < cp_size - 1:
        next_rank = global_ranks[local_rank + 1]
        ops.append(torch.distributed.irecv(tensor=recv_next_first, src=next_rank))
    if local_rank > 0:
        prev_rank = global_ranks[local_rank - 1]
        send_first = tensor.select(dims, 0).contiguous()
        ops.append(torch.distributed.isend(tensor=send_first, dst=prev_rank))
    for op in ops:
        op.wait()

    if local_rank < cp_size - 1:
        last = rolled_tensor.select(dims, -1)
        last.copy_(torch.where(valid_next[-1], recv_next_first, last))

    return rolled_tensor


class MTPLossLoggingHelper:
    """Helper class for logging MTP losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the mtp loss for logging.
        Args:
            loss (torch.Tensor): The loss tensor.
            layer_number (int): Layer index of the loss.
            num_layers (int): The number of total layers.
            reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
            mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
        """
        # Skip mtp loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["values"][layer_number] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    def clean_loss_in_tracker():
        """Clear the mtp losses."""
        tracker = MTPLossLoggingHelper.tracker
        tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    def reduce_loss_in_tracker():
        """Collect and reduce the mtp losses across ranks."""
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]
        # Reduce mtp losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )

    def track_mtp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """Track the Multi-Token Prediction (MTP) metrics for logging."""
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        mtp_losses = tracker["values"] * loss_scale
        mtp_num_layers = mtp_losses.shape[0]
        for i in range(mtp_num_layers):
            name = f"mtp_{i+1} loss"
            loss = mtp_losses[i]
            if total_loss_dict is not None:
                if name in total_loss_dict:
                    total_loss_dict[name] += loss
                else:
                    total_loss_dict[name] = loss
            if writer is not None:
                writer.add_scalar(name, loss, iteration)
            if wandb_writer is not None:
                wandb_writer.log({f"{name}": loss}, iteration)

        MTPLossLoggingHelper.clean_loss_in_tracker()


@dataclass
class MultiTokenPredictionLayerSubmodules:
    """
    Dataclass for specifying the submodules of a MultiTokenPrediction module.

    Args:
        hnorm (Union[ModuleSpec, type]): Specification or instance of the
             hidden states normalization to be applied.
        enorm (Union[ModuleSpec, type]): Specification or instance of the
            embedding normalization to be applied.
        eh_proj (Union[ModuleSpec, type]): Specification or instance of the
            linear projection to be applied.
        transformer_layer (Union[ModuleSpec, type]): Specification
            or instance of the transformer block to be applied.
    """

    enorm: Union[ModuleSpec, type] = None
    hnorm: Union[ModuleSpec, type] = None
    eh_proj: Union[ModuleSpec, type] = None
    e_proj: Union[ModuleSpec, type] = None
    h_proj: Union[ModuleSpec, type] = None
    transformer_layer: Union[ModuleSpec, type] = None
    layer_norm: Union[ModuleSpec, type] = None


def get_mtp_layer_spec(
    transformer_layer_spec: ModuleSpec,
    use_transformer_engine: bool,
    enable_hyper_connections: bool = False,
) -> ModuleSpec:
    """Get the MTP layer spec.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    return get_mtp_layer_spec_for_backend(
        transformer_layer_spec,
        backend=TESpecProvider() if use_transformer_engine else LocalSpecProvider(),
        enable_hyper_connections=enable_hyper_connections,
    )


def get_mtp_layer_spec_for_backend(
    transformer_layer_spec: ModuleSpec,
    backend: BackendSpecProvider,
    enable_hyper_connections: bool = False,
) -> ModuleSpec:
    """Get the MTP layer spec.

    Args:
        transformer_layer_spec: Spec for the transformer layer used inside MTP.
        backend: Backend providing the linear/layernorm impls.
        enable_hyper_connections: Whether the MTP layer should preserve mHC submodules
            (upstream PR #4518). Defaults to False to keep prior behavior.

    Returns:
        ModuleSpec: Module specification with modules from the backend.
    """
    column_parallel_linear_impl: type = backend.column_parallel_linear()
    layer_norm_impl: type = backend.layer_norm()
    submodules_kwargs = dict(
        enorm=layer_norm_impl,
        hnorm=layer_norm_impl,
        transformer_layer=transformer_layer_spec,
        layer_norm=layer_norm_impl,
    )
    if enable_hyper_connections:
        submodules_kwargs["e_proj"] = column_parallel_linear_impl
        submodules_kwargs["h_proj"] = column_parallel_linear_impl
    else:
        submodules_kwargs["eh_proj"] = column_parallel_linear_impl
    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(**submodules_kwargs),
    )
    return mtp_layer_spec


def mtp_on_this_rank(
    config: TransformerConfig, ignore_virtual: Optional[bool] = True, vp_stage: Optional[int] = None
) -> bool:
    """
    Check if there is MTP on the current rank.

    Behavior:
        - If a custom pipeline model parallel layout is provided in the config:
            - If virtual pipeline parallelism is enabled (and `ignore_virtual` is False), checks
              whether any MTP layers are present on this (pp_rank, vp_stage) pair.
            - Otherwise, checks all virtual pipeline ranks of the current pipeline rank. Returns
              True if any virtual sub-rank includes at least one MTP layer.
        - If no custom layout is provided, assumes all MTP layers (if any) are placed on the last
          pipeline stage. The function returns True only on the last pipeline stage.
    """
    mtp_on_this_rank = False
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if config.pipeline_model_parallel_layout is not None:
        # with custom PP layout, we support put MTP layers on any pipeline stage
        layout = config.pipeline_model_parallel_layout.layout
        if (
            not ignore_virtual
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
            num_layers_to_build = layout[pp_rank][vp_stage].count(LayerType.mtp)
            mtp_on_this_rank = num_layers_to_build > 0
        else:
            for vpp_rank in range(len(layout[pp_rank])):
                num_layers_to_build = layout[pp_rank][vpp_rank].count(LayerType.mtp)
                if num_layers_to_build > 0:
                    mtp_on_this_rank = True
                    break
    else:
        # without custom PP layout, we only support put all of MTP layers on the last pipeline stage
        if config.mtp_num_layers is not None:
            mtp_on_this_rank = parallel_state.is_pipeline_last_stage(
                ignore_virtual=ignore_virtual, vp_stage=vp_stage
            )
        else:
            mtp_on_this_rank = False
    return mtp_on_this_rank


def get_mtp_ranks(pp_ranks: List[int], config: TransformerConfig) -> List[int]:
    """Get the ranks of the MTP layers."""
    mtp_ranks = set()
    if config.mtp_num_layers is None:
        return []
    if config.pipeline_model_parallel_layout is None:
        return [pp_ranks[-1]]
    layout = config.pipeline_model_parallel_layout.layout
    for pp_rank in range(len(layout)):
        for vpp_rank in range(len(layout[pp_rank])):
            num_layers_to_build = layout[pp_rank][vpp_rank].count(LayerType.mtp)
            if num_layers_to_build:
                mtp_ranks.add(pp_ranks[pp_rank])
    return list(mtp_ranks)


def get_mtp_layer_offset(config: TransformerConfig, vp_stage: Optional[int] = None) -> int:
    """Get the offset of the MTP layer."""
    # TODO(shifangx): Currently, we only support put all of MTP layers
    # on the last pipeline stage, so the offset is always 0.
    # We will support more flexible MTP placement in the future.
    if config.pipeline_model_parallel_size > 1:
        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.mtp, vp_stage=vp_stage
            )
        else:
            offset = 0
    else:
        offset = 0
    return offset


def get_mtp_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """Get the number of MTP layers to build."""
    if config.pipeline_model_parallel_layout is not None:
        # If we have a custom PP layout, get the number of mtp layers in the layout array.
        num_layers_to_build = config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.mtp, vp_stage=vp_stage
        )
        assert num_layers_to_build == config.mtp_num_layers or num_layers_to_build == 0, (
            f"Currently, we only support put all of MTP layers on the last pipeline stage, "
            f"so the number of MTP layers to build ({num_layers_to_build}) must match "
            f"mtp_num_layers ({config.mtp_num_layers}) or be 0."
        )
    else:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
            num_layers_to_build = config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            num_layers_to_build = 0
    return num_layers_to_build


class MTPLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for mtp loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, mtp_loss: torch.Tensor):
        """Preserve the mtp by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            mtp_loss (torch.Tensor): The mtp loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for mtp loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled mtp loss
                                               gradient.
        """
        (mtp_loss,) = ctx.saved_tensors
        mtp_loss_backward_scale = MTPLossAutoScaler.main_loss_backward_scale
        scaled_mtp_loss_grad = torch.ones_like(mtp_loss) * mtp_loss_backward_scale
        return grad_output, scaled_mtp_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the mtp loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        MTPLossAutoScaler.main_loss_backward_scale = scale


class MultiTokenPredictionLayer(MegatronModule):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MultiTokenPredictionLayerSubmodules,
        layer_number: int = 1,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.sequence_parallel = config.sequence_parallel
        self.submodules = submodules
        self.layer_number = layer_number + get_mtp_layer_offset(self.config, vp_stage)
        self.vp_stage = vp_stage
        self.cp_group = pg_collection.cp

        self_attention_spec = self.submodules.transformer_layer.submodules.self_attention
        attn_mask_type = self_attention_spec.params.get('attn_mask_type', '')
        assert attn_mask_type in SUPPORTED_ATTN_MASK, (
            f"Multi-Token Prediction (MTP) is not jet supported with "
            + f"{attn_mask_type} attention mask type."
            + f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
        )

        self.enorm = build_module(
            self.submodules.enorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.hnorm = build_module(
            self.submodules.hnorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.mhc_enabled = self.config.enable_hyper_connections
        if self.mhc_enabled:
            # mHC mode: separate e_proj and h_proj, operating per-stream.
            # e_proj: [h] -> [h] applied on the embedding then broadcast across streams.
            # h_proj: [h] -> [h] applied per-stream on the multi-stream hidden states.
            self.e_proj = build_module(
                self.submodules.e_proj,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
            self.h_proj = build_module(
                self.submodules.h_proj,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
            self.eh_proj = None
        else:
            # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
            # of the i-th token's hidden states and the (i + K)-th token's decoder input,
            # so the input's shape is [s, b, 2*h].
            # The output will be send to the following transformer layer,
            # so the output's shape should be [s, b, h].
            self.eh_proj = build_module(
                self.submodules.eh_proj,
                self.config.hidden_size * 2,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
            self.e_proj = None
            self.h_proj = None

        diff_transformer_layer_offset = self.config.num_layers - get_transformer_layer_offset(
            self.config, vp_stage
        )

        self.transformer_layer = build_module(
            self.submodules.transformer_layer,
            config=self.config,
            vp_stage=vp_stage,
            layer_number=self.layer_number + diff_transformer_layer_offset,
            is_mtp_layer=True,
        )

        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        if self.mhc_enabled:
            # Learned output-contract parameters per MTP layer (upstream PR #4518).
            hc_mult = self.config.num_residual_streams
            hc_dim = self.config.hidden_size * hc_mult
            self.hc_head_fn = nn.Parameter(torch.randn(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.zeros(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.ones(1))
            nn.init.xavier_uniform_(self.hc_head_fn)
            if self.config.sequence_parallel:
                setattr(self.hc_head_fn, 'sequence_parallel', True)
                setattr(self.hc_head_base, 'sequence_parallel', True)
                setattr(self.hc_head_scale, 'sequence_parallel', True)

        self.offload_context = nullcontext()

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Preprocesses input data for the Multi-Token Prediction (MTP) layers.

        This function computes the decoder input and sends updated input_ids and position_ids to
        the next layer.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            position_ids (torch.Tensor): The position IDs corresponding to the input tokens.
            embedding (Callable): The embedding module
                from gpt model to compute the decoder input.
            hidden_states (torch.Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
        """
        # For chunkpipe, rolling and truncation are handled by the caller
        # (MultiTokenPredictionBlock.forward) to preserve the full sequence
        # with next-batch tokens across MTP layers.
        if not getattr(self.config, 'enable_chunkpipe', False):
            # Standard MTP: roll input_ids and position_ids left by 1
            input_ids, _ = roll_tensor(
                input_ids,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )
            position_ids, _ = roll_tensor(
                position_ids,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )

        # embedding (for chunkpipe, input_ids are already rolled and truncated)
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return input_ids, position_ids, decoder_input, hidden_states

    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
        """
        Concatenate the tokens before sending to transformer layer.
        """
        decoder_input = self.enorm(decoder_input)
        decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)

        if self.mhc_enabled:
            n = self.config.num_residual_streams
            h = self.config.hidden_size
            # hidden_states is [s, b, n*h] (multi-stream).
            # hnorm operates per-stream on the h dimension.
            s, b, _ = hidden_states.shape
            hs_streams = hidden_states.view(s, b, n, h)
            hs_streams = self.hnorm(hs_streams)
            hs_streams = make_viewless_tensor(
                inp=hs_streams, requires_grad=True, keep_graph=True
            )
            # e_proj: [s, b, h] -> [s, b, h], then broadcast to [s, b, n, h]
            e_out, _ = self.e_proj(decoder_input)
            e_out = gather_from_tensor_model_parallel_region(e_out)
            # h_proj: applied per-stream on the h dimension
            h_out, _ = self.h_proj(hs_streams)
            h_out = gather_from_tensor_model_parallel_region(h_out)
            s, b, n, h = h_out.shape
            e_out = e_out.unsqueeze(2).expand(s, b, n, h)
            # Combine and flatten back to [s, b, n*h]
            hidden_states = (e_out + h_out).reshape(s, b, n * h)
            if self.sequence_parallel:
                hidden_states = scatter_to_sequence_parallel_region(hidden_states)
            return hidden_states

        hidden_states = self.hnorm(hidden_states)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        # At the (k - 1)-th MTP module, concatenates the i-th token's hidden_states
        # and the (i + K)-th token's embedding, and combine them with linear projection.
        hidden_states = torch.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)
        # For tensor parallel we need to gather the tensor across the model-parallel
        # ranks after the linear projection. This used to call
        # `all_gather_last_dim_from_tensor_parallel_region`, but that utility reduces
        # the gradient in backward pass and was therefore incorrect in this context.
        # It has been replaced with the correct `gather_from_tensor_model_parallel_region`.
        hidden_states = gather_from_tensor_model_parallel_region(hidden_states)
        # For sequence parallel, scatter after linear_fc and before transformer layer.
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        return hidden_states

    def _proj_and_transformer_layer(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Concatenates embeddings with hidden states and then applies transformer layer forward.
        """
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Unlike transformer_block.py which needs to support mixed-precision in
        # different layers,currently MTP only use global fp8 context.
        if self.config.fp8:
            fp8_context = get_fp8_context(self.config)
            transformer_layer_fp8_context = get_fp8_context(self.config)
        else:
            fp8_context = nullcontext()
            transformer_layer_fp8_context = nullcontext()

        # TODO: currently ignoring FP4 in MTP layers because we need more numerical validation

        with rng_context:
            with fp8_context:
                hidden_states = self._concat_embeddings(hidden_states, decoder_input)

            # Use a separate fp8 context for the transformer layer. This is to ensure that when the
            # transformer layer is cudagraphed, the FP8GlobalStateManager.is_first_fp8_module() is
            # True so that the fp8 weight caching can be triggered correctly.
            with transformer_layer_fp8_context:
                hidden_states, _ = self.transformer_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )

        hidden_states = self._postprocess(hidden_states) if not self.mhc_enabled else hidden_states

        return hidden_states

    def _postprocess(self, hidden_states: torch.Tensor):
        """
        Postprocesses the output of the transformer layers.
        """
        if self.mhc_enabled:
            # Contract n-stream -> 1-stream using the layer's learned head.
            hidden_states = learned_output_contract(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
                self.config.num_residual_streams,
                eps=self.config.layernorm_epsilon,
            )

        # Layer norm before shared head layer.
        hidden_states = self.final_layernorm(hidden_states)
        # TENorm produces a "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return hidden_states

    def _checkpointed_forward(self, forward_func, *args, **kwargs):
        def checkpoint_handler():
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8:
                from megatron.core.extensions.transformer_engine import te_checkpoint

                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    *args,
                    **kwargs,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func, self.config.distribute_saved_activations, *args, *kwargs.values()
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert (
                self.config.recompute_num_layers == 1
            ), "recompute_num_layers must be 1 for MTP recompute"
            outputs = checkpoint_handler()
        elif self.config.recompute_method == 'block':
            # TODO: implement block-based recompute for MTP
            warnings.warn(
                "recompute_method == 'block' is not supported for MTP yet." " Skipping recompute."
            )
            outputs = forward_func(*args, **kwargs)
        elif self.config.recompute_method is None and getattr(self.config, 'enable_chunkpipe', False):
            # Chunkpipe may trigger recompute without normal recompute config being set.
            # In this case, directly do checkpoint.
            outputs = checkpoint_handler()
        else:
            raise ValueError("Invalid activation recompute method.")

        return outputs

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
        embedding=None,
    ):
        """
        Execute the forward pass through the Multi-Token Prediction (MTP) layer.

        Args:
            input_ids (Tensor): Input token IDs .
            position_ids (Tensor): Positional IDs of the input tokens.
            hidden_states (Tensor): Hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention, if applicable.
            context_mask (Tensor, optional): Mask for cross-attention context, if applicable.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Cosine component of rotary positional embeddings.
            rotary_pos_sin (Tensor, optional): Sine component of rotary positional embeddings.
            sequence_len_offset (Tensor, optional): Offset for sequence length, if applicable.
            embedding (Callable): The embedding module from gpt model to compute the decoder input.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        assert context is None, f"multi token prediction + cross attention is not yet supported."

        input_ids, position_ids, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
        )

        recompute_for_chunkpipe = False
        if self.config.enable_chunkpipe:
            # SFT chunkpipe uses scheduler-provided dynamic (chunk_idx, group_size);
            # pretrain uses the global forward counter over a fixed chunk_num_per_seq.
            if getattr(self.config, 'sft_chunkpipe_mode', False):
                chunk_num = self.config.chunkpipe_chunk_idx_in_group
                effective_group = self.config.chunkpipe_current_group_size
            else:
                chunk_num = self.config.chunkpipe_forward_microbatch % self.config.chunk_num_per_seq
                effective_group = self.config.chunk_num_per_seq
            if chunk_num + self.config.keep_activations_chunks < effective_group:
                recompute_for_chunkpipe = True

        if (self.config.recompute_granularity == 'full' or recompute_for_chunkpipe) and self.training:
            hidden_states = self._checkpointed_forward(
                partial(
                    self._proj_and_transformer_layer,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                ),
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
            )
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        return hidden_states, input_ids, position_ids

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the multi token prediction layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the multi
            token prediction layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        return sharded_state_dict


@dataclass
class MultiTokenPredictionBlockSubmodules:
    """
    Dataclass for specifying the submodules of a multi token prediction block.

    This class defines the structure for configuring the layers, allowing for
    flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the multi token prediction block. Each specification typically
            defines a complete multi token prediction layer (e.g., shared embedding,
            projection matrix, transformer block, shared output head).
    """

    layer_specs: List[ModuleSpec] = None


def _get_mtp_block_submodules(
    config: TransformerConfig, spec: Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]
) -> MultiTokenPredictionBlockSubmodules:
    """
    Retrieve or construct MultiTokenPredictionBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]): Specification for the
            multi token prediction block submodules.
            Can be either a MultiTokenPredictionBlockSubmodules instance or a ModuleSpec.

    Returns:
        MultiTokenPredictionBlockSubmodules: The submodules for the multi token prediction block.
    """

    # Transformer block submodules.
    if isinstance(spec, MultiTokenPredictionBlockSubmodules):
        return spec
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, MultiTokenPredictionBlock):
            return spec.submodules
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class MultiTokenPredictionBlock(MegatronModule):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        vp_stage: Optional[int] = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)
        self.submodules = _get_mtp_block_submodules(config, spec)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self.vp_stage = vp_stage

        # Initialize Context Parallelism (CP) support for MTP
        # This enables MTP to work with CP > 1 by providing the CP process group
        # to the roll_tensor function for proper boundary communication
        if pg_collection is None:
            # Use default MPU process groups if not provided
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['cp'])
        else:
            # Ensure the provided process groups include CP
            assert hasattr(
                pg_collection, 'cp'
            ), "MultiTokenPredictionBlock pg_collection must have cp process group"

        self._build_layers(pg_collection)
        assert len(self.layers) > 0, "MultiTokenPredictionBlock must have at least one layer."
        self.cp_group = pg_collection.cp

    def _build_layers(self, pg_collection):
        def build_layer(layer_spec, layer_number):
            fp8_init_context = get_fp8_context(self.config, is_init=True)
            with fp8_init_context:
                module = build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                )
            return module

        if self.config.mtp_shared_layers:
            # Build a single MTP layer and reuse it for all prediction depths.
            shared_layer = build_layer(self.submodules.layer_specs[0], 1)
            self.layers = torch.nn.ModuleList([shared_layer])
            self.mtp_num_forward_passes = len(self.submodules.layer_specs)
        else:
            self.layers = torch.nn.ModuleList(
                [
                    build_layer(layer_spec, i + 1 )
                    for i, layer_spec in enumerate(self.submodules.layer_specs)
                ]
            )
            self.mtp_num_forward_passes = len(self.layers)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
        extra_block_kwargs: dict = None,
        embedding=None,
        mhc_multistream: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Perform the forward pass through all of the MTP modules.

        Args:
            hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Contracted decoder hidden states [s, b, h] when mHC is enabled.
            mhc_multistream (Tensor, optional): When mHC is enabled, the pre-contraction
                multi-stream decoder output [s, b, n*h] used as input to MTP depths.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.

        Returns:
            (Tensor): The mtp loss tensor of shape [b, s].
        """
        # get hidden states from previous mtp stages
        offset = get_mtp_layer_offset(self.config, self.vp_stage)
        hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
        hidden_states_main = hidden_states_list[offset]
        if mhc_multistream is not None:
            # mHC mode: use multi-stream for MTP depth input, contracted for loss list.
            mhc_chunks = list(torch.chunk(mhc_multistream, 1 + offset, dim=0))
            hidden_states = mhc_chunks[offset]
        else:
            hidden_states = hidden_states_main

        # For chunkpipe, manage rolling at the block level to preserve the full
        # input_ids (with appended next-batch tokens) across MTP layers.
        # Each iteration rolls the full copy left by 1, then truncates to chunk_size
        # for the layer's embedding, while keeping the full rolled version for the
        # next iteration so that subsequent rolls chain correctly.

        if getattr(self.config, 'enable_chunkpipe', False):
            full_input_ids = input_ids
            full_position_ids = position_ids
            # For SFT chunkpipe MTP, the full tensor may span
            # [chunksize + mtp_num_layers] after preprocessing appends bridge tokens.
            # The cu_seqlens_q in packed_seq_params only describes the current base
            # chunk (len=chunksize) and would leave the trailing k positions unrolled,
            # breaking the bridge token between chunk boundaries. Treat the window as
            # contiguous (packed_seq_params=None) for rolling in that case.
            # Detection: SFT mode + tensor length > chunksize (bridge tokens present).
            roll_packed_seq_params = packed_seq_params
            if (
                getattr(self.config, 'sft_chunkpipe_mode', False)
                and full_input_ids.size(-1) > self.config.chunksize
            ):
                roll_packed_seq_params = None


        for layer_number in range(self.mtp_num_forward_passes):
            if self.config.fine_grained_activation_offloading:
                fine_grained_offloading_set_last_layer(layer_number == self.mtp_num_forward_passes - 1)

            if getattr(self.config, 'enable_chunkpipe', False):
                # Roll the full sequence (including next-batch tokens) left by 1
                full_input_ids, _ = roll_tensor(
                    full_input_ids,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=roll_packed_seq_params,
                )
                full_position_ids, _ = roll_tensor(
                    full_position_ids,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=roll_packed_seq_params,
                )
                # Truncate to chunk_size for this layer's embedding.
                # Clone to avoid sharing memory with full_input_ids/full_position_ids,
                # preventing accidental in-place corruption in downstream layers.
                # Use the CURRENT chunk's real token width, not the scalar
                # self.config.chunksize (which is max(chunk_sizes) in variable-length
                # mode). The bridge-token preprocessing extends each chunk to
                # chunk_len + mtp_num_layers; truncating a short chunk (e.g. 512) by
                # the max width (1024) would leave it at 513, so the sequence-parallel
                # reduce-scatter in the embedding sees a length not divisible by TP and
                # crashes. Uniform mode is unchanged (chunk_sizes[idx] == chunksize).
                chunk_sizes = getattr(self.config, 'chunk_sizes', None)
                if chunk_sizes is not None:
                    chunk_size = chunk_sizes[self.config.chunkpipe_chunk_idx_in_group]
                else:
                    chunk_size = self.config.chunksize
                layer_input_ids = full_input_ids[:, :chunk_size].clone()
                layer_position_ids = full_position_ids[:, :chunk_size].clone()
            else:
                layer_input_ids = input_ids
                layer_position_ids = position_ids

            # When layers are shared, always use self.layers[0]; otherwise index normally.
            layer = self.layers[0] if self.config.mtp_shared_layers else self.layers[layer_number]
            (layer_hidden_states, input_ids, position_ids) = layer(
                input_ids=layer_input_ids,
                position_ids=layer_position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=embedding,
                **(extra_block_kwargs or {}),
            )

            # append the output hidden states of the current mtp layer
            # to the hidden_states_list
            if mhc_multistream is not None:
                # mHC: layer_hidden_states is multi-stream [s,b,n*h]; chain it as next
                # input, but append the contracted form to the list so the LM head gets
                # [s,b,h] for each depth.
                hidden_states_list.append(layer._postprocess(layer_hidden_states))
                if self.config.mtp_connection_type == 'parallel':
                    hidden_states = mhc_chunks[offset]
                else:
                    hidden_states = layer_hidden_states
            else:
                hidden_states_list.append(layer_hidden_states)
                # sequential: chain hidden states through MTP layers;
                # parallel: every MTP layer receives the main model's hidden states.
                if self.config.mtp_connection_type == 'parallel':
                    hidden_states = hidden_states_main
                else:
                    hidden_states = layer_hidden_states

        # concat the hidden states of all mtp layers
        hidden_states = torch.cat(hidden_states_list, dim=0)
        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the multi token prediction module.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the multi
            token prediction module.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        layer_prefix = f'{prefix}layers.'
        for layer in self.layers:
            offset = get_mtp_layer_offset(self.config, self.vp_stage)
            sharded_prefix = f'{layer_prefix}{layer.layer_number - 1 }.'

            state_dict_prefix = f'{layer_prefix}{layer.layer_number - 1 - offset}.'
            sharded_pp_offset = []
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)
        return sharded_state_dict
