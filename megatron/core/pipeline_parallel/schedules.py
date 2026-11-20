# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from collections import deque
from functools import partial
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_reset,
)
from megatron.core.pipeline_parallel.p2p_communication import (
    P2PCommunicator,
    p2p_comm_tensor_remove_padding,
)
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
)
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    nvtx_range_pop,
    nvtx_range_push,
)

from .combined_1f1b import (
    combined_1f1b_schedule_for_interleaved_pipelining,
    combined_1f1b_schedule_for_no_pipelining,
)

# Types
Shape = Union[List[int], torch.Size]


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation
        step.

    adjust_tensor_shapes_fn (Callable, optional): A function that adjusts the receive and send
        tensor shapes. Only applicable in forward_backward_pipelining_without_interleaving for now.
        Takes in a list of receive shapes and a list of send shapes and returns the adjusted
        respective list of shapes. Thus it is not used in the other forward-backward functions
        which have different shape handling.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def set_current_microbatch(model, microbatch_id):
    """Set the current microbatch."""
    decoder_exists = True
    model_with_decoder = None
    try:
        model_with_decoder = get_attr_wrapped_model(
            model, "decoder", allow_none=False, return_model_obj=True
        )
    except RuntimeError:
        decoder_exists = False
    if decoder_exists and model_with_decoder is not None:
        for layer in model_with_decoder.decoder.layers:
            layer.current_microbatch = microbatch_id
        if hasattr(model_with_decoder, 'mtp'):
            for layer in model_with_decoder.mtp.layers:
                layer.transformer_layer.current_microbatch = microbatch_id


def forward_step_calc_loss(
        model,
        output_tensor,
        loss_func,
        config,
        vp_stage,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size=None,
        is_last_stage=None,
):
    """Calculate the loss and number of tokens for forward_step()"""

    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    model_vp_stage = getattr(model, "vp_stage", None)
    if vp_stage is not None and model_vp_stage is not None:
        assert (
                vp_stage == model_vp_stage
        ), f"vp_stage ({vp_stage}) doesn't match model_vp_stage ({model_vp_stage})"

    if cp_group_size is None and is_last_stage is None:
        # fallback to parallel state
        cp_group_size = parallel_state.get_context_parallel_world_size()
        is_last_stage = parallel_state.is_pipeline_last_stage(
            ignore_virtual=False, vp_stage=vp_stage
        )
    else:
        assert (
                cp_group_size is not None and is_last_stage is not None
        ), "cp_group_size and is_last_stage must be provided"

    num_tokens = torch.tensor(0, dtype=torch.int)
    if is_last_stage:
        if not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    # Protect against division by zero when all tokens are masked
                    #   in a microbatch.
                    output_tensor /= torch.clamp(num_tokens, min=1)
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor *= cp_group_size
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # Set the loss scale for Multi-Token Prediction (MTP) loss.
    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            if hasattr(config, 'enable_chunkpipe') and config.enable_chunkpipe:
                if getattr(config, 'sft_chunkpipe_mode', False):
                    step_num_groups = getattr(config, 'chunkpipe_step_num_groups', None)
                    assert step_num_groups is not None and step_num_groups > 0, (
                        "SFT chunkpipe MTP backward scaling requires config.chunkpipe_step_num_groups > 0."
                    )
                    dp_size = parallel_state.get_data_parallel_world_size()
                    MTPLossAutoScaler.set_loss_scale(loss_scale * dp_size / step_num_groups)
                else:
                    MTPLossAutoScaler.set_loss_scale(loss_scale / (num_microbatches / config.chunk_num_per_seq))
            else:
                MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # Set the loss scale for DSA indexer loss.
    if (
        hasattr(config, 'dsa_indexer_loss_coeff')
        and config.dsa_indexer_loss_coeff is not None
        and config.dsa_indexer_loss_coeff > 0
    ):
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        if config.calculate_per_token_loss:
            DSAIndexerLossAutoScaler.set_loss_scale(loss_scale)
        else:
            if hasattr(config, 'enable_chunkpipe') and config.enable_chunkpipe:
                DSAIndexerLossAutoScaler.set_loss_scale(loss_scale / (num_microbatches / config.chunk_num_per_seq))
            else:
                DSAIndexerLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    return output_tensor, num_tokens


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    cp_group_size,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    vp_stage=None,
    is_last_stage=True,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.
        vp_stage (int, optional):
            The virtual pipeline stage. Defaults to None.
        is_last_stage (bool, optional):
            Whether it is the last stage. Defaults to True.
            Also considering virtual stages.
            In case of PP/VPP, is_last_stage/is_vp_last_stage.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )
    output_tensor, num_tokens = forward_step_calc_loss(
        model,
        output_tensor,
        loss_func,
        config,
        vp_stage,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size,
        is_last_stage,
    )

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    # In multi-modal models like VLM, some batches may not have images.
    # When no image is present, the vision encoder (as a separate pipeline stage)
    # will not participate in the computation.
    # This results in a tensor that does not require gradients.
    # In such cases, we intentionally skip the backward pass while preserving zero gradients.
    if (output_tensor[0].requires_grad) and (output_tensor[0].grad_fn is not None):
        if config.deallocate_pipeline_outputs:
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def check_first_val_step(first_val_step, forward_only, cond):
    """Check if it is the first validation step."""
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond


def get_min_key(chunks_micro: dict):
    """get min key for a dict"""
    min_key = -1
    for tmp_seq in chunks_micro:
        if min_key == -1 or min_key > tmp_seq:
            min_key = tmp_seq
    assert min_key > -1, "chunks_micro is None, min_key could not be -1"
    return min_key


def remove_key_value_cache(model, micro_batch_index, mtp_num_layers):
    """
    Remove the key-value cache for a specific micro-batch index.

    In chunked pipeline parallel training, each sequence is split into multiple micro-batches.
    This function removes the attention key-value cache for a specific micro-batch after its backward pass.

    Args:
        model: The target model containing decoder layers and possibly MTP layers
        micro_batch_index: Index of the micro-batch whose cache should be removed

    Note:
        This function iterates through all decoder layers and calls delete_chunk_key_value_cache
        on each layer's self-attention module.
    """
    decoder = get_attr_wrapped_model(model, "decoder")
    num_layers = len(decoder.layers)
    for layer_idx in range(num_layers):
        decoder.layers[layer_idx].self_attention.delete_chunk_key_value_cache(
            micro_batch_index
        )
        indexer = getattr(decoder.layers[layer_idx].self_attention.core_attention, 'indexer', None)
        if indexer is not None:
            indexer.delete_chunk_indexer_key_cache(micro_batch_index)
    
    if mtp_num_layers is None or mtp_num_layers == 0:
        return
    if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        return
    mtp_layers = get_attr_wrapped_model(model, "mtp")
    for mtp_layer in mtp_layers.layers[:mtp_num_layers]:
        mtp_layer.transformer_layer.self_attention.delete_chunk_key_value_cache(
            micro_batch_index
        )
        indexer = getattr(mtp_layer.transformer_layer.self_attention.core_attention, 'indexer', None)
        if indexer is not None:
            indexer.delete_chunk_indexer_key_cache(micro_batch_index)


def clear_key_value_cache(model, mtp_num_layers):
    """
    Clear all key-value caches in the model's attention layers.

    Used in forward-only scenarios (like inference/validation) to reset all attention caches,
    typically after processing a complete sequence to prepare for the next sequence.

    Args:
        model: The target model containing decoder layers and possibly MTP layers

    Note:
        This function iterates through all decoder layers and calls clear_chunk_key_value_cache
        on each layer's self-attention module.
    """
    decoder = get_attr_wrapped_model(model, "decoder")
    num_layers = len(decoder.layers)
    for layer_idx in range(num_layers):
        decoder.layers[layer_idx].self_attention.clear_chunk_key_value_cache()
        indexer = getattr(decoder.layers[layer_idx].self_attention.core_attention, 'indexer', None)
        if indexer is not None:
            indexer.clear_chunk_indexer_key_cache()
    
    if mtp_num_layers is None or mtp_num_layers == 0:
        return
    if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        return
    mtp_layers = get_attr_wrapped_model(model, "mtp")
    for mtp_layer in mtp_layers.layers[:mtp_num_layers]:
        mtp_layer.transformer_layer.self_attention.clear_chunk_key_value_cache()
        indexer = getattr(mtp_layer.transformer_layer.self_attention.core_attention, 'indexer', None)
        if indexer is not None:
            indexer.clear_chunk_indexer_key_cache()


def forward_backward_no_pipelining_with_chunkpipe(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    Returns dictionary with losses.

    Supports two modes:
      - Pretrain: fixed chunk_num_per_seq, original logic preserved.
      - SFT: dynamic chunk_group_size per group, discovered after first forward_step.
    """
    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    # Determine whether this is SFT chunkpipe (dynamic group_size) or pretrain (fixed).
    is_sft_chunkpipe = config.sft_chunkpipe_mode

    if not is_sft_chunkpipe:
        # ========== Original pretrain logic (unchanged) ==========
        assert num_microbatches % config.chunk_num_per_seq == 0, "num microbatches should be divided by num chunks"
        num_sequences = num_microbatches // config.chunk_num_per_seq

        chunkpipe_forward_microbatch = 0
        with no_sync_func():
            for seq_index in range(num_sequences - 1):
                chunk_losses = []
                config.chunkpipe_forward = True
                for chunk_index in range(config.chunk_num_per_seq):
                    config.chunkpipe_forward_microbatch = chunkpipe_forward_microbatch
                    config.chunkpipe_chunk_idx_in_group = chunk_index
                    output_tensor, num_tokens = forward_step(
                        forward_step_func,
                        data_iterator,
                        model,
                        num_microbatches,
                        input_tensor,
                        forward_data_store,
                        config,
                        parallel_state.get_context_parallel_group(),
                        collect_non_loss_data,
                        is_first_microbatch=check_first_val_step(
                            first_val_step, forward_only, chunkpipe_forward_microbatch == 0
                        ),
                        current_microbatch=chunkpipe_forward_microbatch,
                    )
                    total_num_tokens += num_tokens.item()
                    output_and_microbatch = (output_tensor, chunkpipe_forward_microbatch)
                    chunk_losses.append(output_and_microbatch)
                    chunkpipe_forward_microbatch += 1

                # backward according to reverse direction of forward
                if not forward_only:
                    config.chunkpipe_forward = False
                    for chunk_index in range(config.chunk_num_per_seq):
                        output_and_microbatch = chunk_losses.pop()
                        config.chunkpipe_backward_microbatch = output_and_microbatch[1]
                        config.chunkpipe_chunk_idx_in_group = config.chunk_num_per_seq - 1 - chunk_index
                        backward_step(input_tensor, output_and_microbatch[0],
                                        output_tensor_grad, model_type, config)

                        # remove caches for key & values
                        remove_key_value_cache(model, output_and_microbatch[1], config.mtp_num_layers)
                else:
                    # clear keys & values cache
                    clear_key_value_cache(model, config.mtp_num_layers)


        # Run computation for last microbatch out of context handler (want to
        # synchronize gradients).
        chunk_losses = []
        config.chunkpipe_forward = True
        for chunk_index in range(config.chunk_num_per_seq):
            config.chunkpipe_forward_microbatch = chunkpipe_forward_microbatch
            config.chunkpipe_chunk_idx_in_group = chunk_index
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                parallel_state.get_context_parallel_group(),
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(
                    first_val_step, forward_only, chunkpipe_forward_microbatch == 0
                ),
                current_microbatch=chunkpipe_forward_microbatch,
            )
            total_num_tokens += num_tokens.item()
            output_and_microbatch = (output_tensor, chunkpipe_forward_microbatch)
            chunk_losses.append(output_and_microbatch)
            chunkpipe_forward_microbatch += 1

        if not forward_only:
            config.chunkpipe_forward = False
            for chunk_index in range(config.chunk_num_per_seq):
                output_and_microbatch = chunk_losses.pop()
                config.chunkpipe_backward_microbatch = output_and_microbatch[1]
                config.chunkpipe_chunk_idx_in_group = config.chunk_num_per_seq - 1 - chunk_index
                backward_step(input_tensor, output_and_microbatch[0],
                                output_tensor_grad, model_type, config)

                # remove caches for key & values
                remove_key_value_cache(model, output_and_microbatch[1], config.mtp_num_layers)
        else:
            # clear keys & values cache
            clear_key_value_cache(model, config.mtp_num_layers)

    else:
        # ========== SFT dynamic group_size scheduling ==========
        # Composite group support:
        #   A "composite group" is the unit the sampler hands one rank in one
        #   pipeline step. It is described by ``config.chunkpipe_component_sizes``
        #   = [c1, c2, ...] where each c_i is the size of one real source group.
        #   The total chunk count of a composite is sum(c_i).
        #
        #   For the equal-DP / legacy path the sampler emits trivial composites
        #   ``[group_size]`` so behavior is identical to the prior single-real-
        #   group loop. For unequal-DP synthesis, a composite may contain
        #   multiple real groups whose ``chunk_idx_in_group`` resets per real
        #   group; backward replays them in LIFO via the stored idx so each
        #   real group's KV-chain reset is correctly triggered.
        chunkpipe_forward_microbatch = 0
        microbatch_idx = 0  # total micro-batches consumed so far

        last_group_losses = None
        last_group_total_chunks = 0

        def _forward_one_chunk(chunk_idx_in_group, is_first_global):
            """Forward a single chunk and return (output_tensor, microbatch_id, chunk_idx_in_group, real_group_size)."""
            nonlocal chunkpipe_forward_microbatch, total_num_tokens
            config.chunkpipe_forward_microbatch = chunkpipe_forward_microbatch
            config.chunkpipe_chunk_idx_in_group = chunk_idx_in_group
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                parallel_state.get_context_parallel_world_size(),
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(
                    first_val_step, forward_only, is_first_global
                ),
                current_microbatch=chunkpipe_forward_microbatch,
            )
            total_num_tokens += num_tokens.item()
            # Snapshot real_group_size set by get_batch so backward can restore it.
            # For heterogeneous composites (e.g. components=[3,2]) the config value
            # changes between real groups; backward must replay the correct per-chunk
            # size so that "last chunk in group" detection in MLA/Attention/Router
            # uses the original forward value, not a stale one from a later real group.
            real_group_size = config.chunkpipe_current_group_size
            result = (output_tensor, chunkpipe_forward_microbatch, chunk_idx_in_group, real_group_size)
            chunkpipe_forward_microbatch += 1
            return result

        def _backward_one_chunk(chunk_losses):
            """Backward a single chunk (reverse / LIFO order) and free its KV cache.

            The stored ``chunk_idx_in_group`` and ``real_group_size`` from the forward
            pass are restored so that downstream layers (MLA, Attention, Router) see
            the original per-real-group state when computing "last chunk in group"
            boundaries.  Without restoring ``chunkpipe_current_group_size``, a
            heterogeneous composite such as ``[3, 2]`` would leave the config at 2
            after forward, causing the backward pass over the size-3 real group to
            misidentify chunk indices (e.g. idx 1 appearing as the last chunk).

            Stack-pop ordering naturally yields per-real-group LIFO: for a
            composite ``[c1, c2, ..., cN]`` the chunks are popped in real-group
            order N-1, ..., 0 and within each real group from idx c_i-1 down to 0.
            ``chunk_idx_in_group == 0`` (the first-forwarded chunk of each real
            group) is the last to be popped per real group and is the canonical
            grad-ready / KV-reset boundary.
            """
            popped = chunk_losses.pop()  # (output_tensor, mb_id, chunk_idx_in_group, real_group_size)
            config.chunkpipe_backward_microbatch = popped[1]
            config.chunkpipe_chunk_idx_in_group = popped[2]
            config.chunkpipe_current_group_size = popped[3]
            backward_step(input_tensor, popped[0],
                          output_tensor_grad, model_type, config)
            remove_key_value_cache(model, popped[1], config.mtp_num_layers)

        def _backward_group(chunk_losses, total_chunks, sync_last_only=False):
            """Backward all chunks in a composite (reverse order) and manage KV cache.

            ``total_chunks`` is the composite length (sum of component sizes).
            When ``sync_last_only`` is True, the first ``total_chunks - 1`` chunks
            run inside ``no_sync_func()`` so that only the very last chunk
            (the first-forwarded chunk of the first real group, with reverse
            ``chunk_idx_in_group == 0``) triggers DDP grad-ready notifications.
            This avoids ``register_grad_ready`` being called multiple times on
            the same parameter within a single composite when
            ``overlap_grad_reduce=True``.
            """
            config.chunkpipe_forward = False
            if sync_last_only and total_chunks > 1:
                with no_sync_func():
                    for _ in range(total_chunks - 1):
                        _backward_one_chunk(chunk_losses)
                _backward_one_chunk(chunk_losses)
            else:
                for _ in range(total_chunks):
                    _backward_one_chunk(chunk_losses)

        # Phase 1: process all composites inside no_sync (except last composite's backward)
        with no_sync_func():
            while microbatch_idx < num_microbatches:
                # Forward first chunk of the composite (idx 0 of real-group 0).
                # This invocation lets get_batch populate config with the
                # composite descriptor (chunkpipe_component_sizes) and the
                # current real-group size.
                config.chunkpipe_forward = True
                is_first_global = (chunkpipe_forward_microbatch == 0)
                chunk_losses = [_forward_one_chunk(0, is_first_global)]

                # Read composite descriptor written by get_batch -> config.
                # Fall back to a single real group if the composite path is not
                # active (legacy callers / ad-hoc tests that don't set it).
                components = list(getattr(config, 'chunkpipe_component_sizes', None) or [])
                if not components:
                    components = [config.chunkpipe_current_group_size or 1]
                first_size = components[0]

                # Forward remaining chunks of the first real group
                for ci in range(1, first_size):
                    chunk_losses.append(_forward_one_chunk(ci, False))

                # Forward subsequent real groups: chunk_idx_in_group resets
                # to 0 at each real-group boundary so get_batch / model code
                # see the correct per-real-group state and KV cache resets
                # are triggered properly.
                for real_size in components[1:]:
                    for ci in range(real_size):
                        chunk_losses.append(_forward_one_chunk(ci, False))

                total_chunks = sum(components)
                microbatch_idx += total_chunks

                # Check if this is the last composite
                if microbatch_idx >= num_microbatches:
                    last_group_losses = chunk_losses
                    last_group_total_chunks = total_chunks
                    break

                # Non-last composite: backward inside no_sync
                if not forward_only:
                    _backward_group(chunk_losses, total_chunks)
                else:
                    if total_chunks > 1:
                        clear_key_value_cache(model, config.mtp_num_layers)

        # Phase 2: last composite's backward. Only the final chunk (reverse
        # chunk_idx_in_group == 0 of real-group 0) runs outside no_sync to
        # trigger grad sync; earlier chunks stay inside no_sync to avoid
        # duplicate register_grad_ready calls on shared params when
        # overlap_grad_reduce is enabled.
        if last_group_losses is not None:
            if not forward_only:
                _backward_group(last_group_losses, last_group_total_chunks,
                                sync_last_only=True)
            else:
                if last_group_total_chunks > 1:
                    clear_key_value_cache(model, config.mtp_num_layers)

    # Common finalization
    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store

def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: Optional[int] = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Run forward and backward passes with no pipeline parallelism"""

    if pg_collection is None:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    elif pg_collection is not None:
        assert hasattr(pg_collection, 'tp')
        assert hasattr(pg_collection, 'cp')
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it is used default "
            "`parallel_state.default_embedding_ranks` to create the process group. If you are "
            "using the default process group, please use `parallel_state.get_embedding_group()` "
            "to get the process group. If you don't need explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have a pos_embd. In previous version, it is used default "
            "`parallel_state.default_position_embedding_ranks` to create the process group. "
            "If you are using the default process group, "
            "please use `parallel_state.get_position_embedding_group()` "
            "to get the process group. If you don't need explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pp')
        assert hasattr(pg_collection, 'dp_cp')

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for non-pipeline-parallel schedule"

    config = get_model_config(model)
    if config.enable_chunkpipe:
        return forward_backward_no_pipelining_with_chunkpipe(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                decoder_seq_length=decoder_seq_length,
                forward_only=forward_only)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if not forward_only and config.fine_grained_activation_offloading:
        fine_grained_offloading_reset()

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if config.overlap_moe_expert_parallel_comm and not forward_only:
        forward_data_store, total_num_tokens = combined_1f1b_schedule_for_no_pipelining(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            output_tensor_grad,
            forward_data_store,
            config,
            collect_non_loss_data,
            first_val_step,
            forward_only,
            no_sync_func,
            total_num_tokens,
            partial(check_first_val_step, first_val_step, forward_only),
        )
    else:
        with no_sync_func():
            for i in range(num_microbatches - 1):
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    pg_collection.cp.size(),
                    collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                    current_microbatch=i,
                )
                total_num_tokens += num_tokens
                if not forward_only:
                    backward_step(
                        input_tensor, output_tensor, output_tensor_grad, model_type, config
                    )
        # Run computation for last microbatch out of context handler (want to
        # synchronize gradients).
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            pg_collection.cp.size(),
            collect_non_loss_data,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, num_microbatches == 1
            ),
            current_microbatch=num_microbatches - 1,
        )

        total_num_tokens += num_tokens

        if not forward_only:
            backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and config.cuda_graph_scope != "full_iteration"
    ):
        create_cudagraphs()

    return forward_data_store


def clear_embedding_activation_buffer(config, model, is_last_stage):
    """Clear embedding activation buffer."""

    if is_last_stage and config.defer_embedding_wgrad_compute:
        if isinstance(model, list):
            embedding_module = get_attr_wrapped_model(
                model[-1], 'post_process', return_model_obj=True
            )
        else:
            embedding_module = get_attr_wrapped_model(model, 'post_process', return_model_obj=True)

        # Need to ensure no stray activations exists in this buffer
        embedding_module.embedding_activation_buffer.clear()

        return embedding_module
    else:
        return None


def finish_embedding_wgrad_compute(config, embedding_module, is_last_stage, tp_group):
    """Finish embedding wgrad compute."""
    if is_last_stage and config.defer_embedding_wgrad_compute:
        embedding_activation_buffer = embedding_module.embedding_activation_buffer
        grad_output_buffer = embedding_module.grad_output_buffer
        weight = (
            embedding_module.output_layer.weight
            if embedding_module.share_embeddings_and_output_weights
            else embedding_module.shared_embedding_or_output_weight()
        )

        drain_embedding_wgrad_compute(
            config, embedding_activation_buffer, grad_output_buffer, weight, tp_group
        )


def get_pp_rank_microbatches(
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage,
    forward_only=False,
    overlap_moe_expert_parallel_comm=False,
    p2p_communicator: Optional[P2PCommunicator] = None,
    enable_chunkpipe=False,
    num_chunks=0
):
    """Get the number of total, warmup, and remaining microbatches in PP scheduling."""
    if p2p_communicator is not None:
        pipeline_parallel_size = p2p_communicator.pp_group.size()
        pipeline_parallel_rank = p2p_communicator.pp_group.rank()
        virtual_pipeline_parallel_size = p2p_communicator.virtual_pipeline_model_parallel_size
    else:
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        virtual_pipeline_parallel_size = (
            parallel_state.get_virtual_pipeline_model_parallel_world_size()
        )

    total_num_microbatches = num_microbatches * num_model_chunks
    are_all_microbatches_in_warmup = False

    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    elif pipeline_parallel_size > 1:
        if virtual_pipeline_parallel_size is None:
            # forward_backward_pipelining_without_interleaving
            num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
        else:
            # forward_backward_pipelining_with_interleaving
            # Run (num_model_chunks-1)*microbatch_group_size_per_vp_stage on
            # all workers, followed by more microbatches after depending on
            # stage ID (more forward passes for earlier stages, later stages can
            # immediately start with 1F1B).
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            if not enable_chunkpipe:
                num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
            else:
                num_warmup_microbatches += num_chunks * virtual_pipeline_parallel_size - 1
            # When enabling overlap_moe_expert_parallel_comm, we schedule one extra micro-batch
            # forward step before the 1f1b stages. This is needed to ensure the forward
            # and backward computations are independent in all 1f1b steps.
            if overlap_moe_expert_parallel_comm:
                num_warmup_microbatches = num_warmup_microbatches + 1
    else:
        # forward_backward_no_pipelining
        # This path is only used for cuda graph capturing compatibility for the PP=1 case.
        num_warmup_microbatches = 0

    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches
        are_all_microbatches_in_warmup = True
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    return (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    )


def get_extended_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage, num_chunks,
                                override_mgspvs=None):
    """Extended schedule table to support sequence chunking"""
    schedule_table = []
    if override_mgspvs is not None:
        microbatch_group_size_per_vp_stage = override_mgspvs
    else:
        microbatch_group_size_per_vp_stage = min(microbatch_group_size_per_vp_stage,
                        int(parallel_state.get_pipeline_model_parallel_world_size() / num_chunks))
        microbatch_group_size_per_vp_stage = max(microbatch_group_size_per_vp_stage, 1)
    num_microbatches = int(num_microbatches / num_chunks)
    for min_microbatch_id_in_group in range(0, num_microbatches, microbatch_group_size_per_vp_stage):
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:
            # Last group
            for model_chunk_id in range(num_model_chunks):
                for microbatch_id in range(min_microbatch_id_in_group, num_microbatches):
                    for chunk_id in range(num_chunks):
                        schedule_table.append((microbatch_id, model_chunk_id, chunk_id))
        else:
            # Other groups
            for model_chunk_id in range(num_model_chunks):
                for microbatch_id in range(min_microbatch_id_in_group,
                                         min_microbatch_id_in_group + microbatch_group_size_per_vp_stage):
                    for chunk_id in range(num_chunks):
                        schedule_table.append((microbatch_id, model_chunk_id, chunk_id))
    return schedule_table


def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """Get the schedule table for PP scheduling."""
    schedule_table = []
    for min_microbatch_id_in_group in range(
        0, num_microbatches, microbatch_group_size_per_vp_stage
    ):
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:
            # Construct schedule for the last microbatch group
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:
            # Construct schedule for other microbatch groups
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + microbatch_group_size_per_vp_stage,
                    )
                ]
            )
    return schedule_table


def convert_schedule_table_to_order(num_warmup_microbatches, num_model_chunks, schedule_table):
    """Convert a tunable schedule lookup table to the te.make_graphed_callables() accepted
    order format. For example, the tunable schedule table for PP2 N3M5 with VP2 is as below:
    virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    model_chunk_id        | 0 0 0 1 1 1 0 0 1 1

    Then the forward backward separated order is:
    forward               | 1 1 1 2 2 2 1 1 2 2
    backward              | -2 -2 -2 -1 -1 -1 -2 -2 -1 -1

    If num_warmup_microbatches is 5, the output order is:
    1 1 1 2 2 2 -2 1 -2 1 -2 2 -1 2 -1 -1 -2 -2 -1 -1
    """
    _, model_chunk_id_table = zip(*schedule_table)
    forward_order = [chunk_id + 1 for chunk_id in model_chunk_id_table]
    backward_order = [chunk_id - num_model_chunks for chunk_id in model_chunk_id_table]
    order = forward_order[:num_warmup_microbatches]
    for i in range(num_warmup_microbatches, len(forward_order)):
        order.append(forward_order[i])
        order.append(backward_order[i - num_warmup_microbatches])
    if num_warmup_microbatches > 0:
        order.extend(backward_order[-num_warmup_microbatches:])
    return order


def forward_backward_pipelining_with_interleaving_with_chunkpipe(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Run interleaved 1F1B with chunk schedule (model split into model chunks), with
    communication between pipeline stages as needed.
        
    Convention used in this function:
    num_microbatches for number of microbatches per pipeline stage;
    num_model_chunks for virtual pipeline size;
    then total_num_microbatches = num_microbatches * num_model_chunks.
    Their corresponding index variables are
    microbatch_id in [0, num_microbatches)
    model_chunk_id in [0, num_model_chunks)
    virtual_microbatch_id in [0, total_num_microbatches)

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    config = get_model_config(model[0])
    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model[0])
        assert model_type != ModelType.encoder_and_decoder, (
            "encoder PP stages not yet supported when passing custom process groups. "
            "support coming soon!"
        )
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have a tp_group"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have a cp_group"
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it is used default "
            "`parallel_state.default_embedding_ranks` to create the process group. If you are "
            "using the default process group, please use `parallel_state.get_embedding_group()` "
            "to get the process group. If you don't need explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have a pos_embd. In previous version, it is used default "
            "`parallel_state.default_position_embedding_ranks` to create the process group."
            " If you are using the default process group, please use "
            "`parallel_state.get_position_embedding_group()` "
            "If you don't need pos_embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pp'), "pg_collection must have a pp_group"
        assert hasattr(pg_collection, 'dp_cp'), "pg_collection must have a dp_cp_group"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection"
            " provide none or provide all the process groups"
        )

    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for interleaved pipeline parallelism"

    if not forward_only and config.fine_grained_activation_offloading:
        fine_grained_offloading_reset()

    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        # vp is ignored for clear_embedding_activation_buffer
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
    # They will be re-enabled at the end of this function.
    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # SFT dynamic group_size support for interleaved chunkpipe
    is_sft_chunkpipe = getattr(config, 'sft_chunkpipe_mode', False)

    if is_sft_chunkpipe:
        # Clear KV cache at the start of each training iteration for all model chunks.
        # In VPP train mode (non-forward_only), clear_key_value_cache is NOT called
        # at group boundaries (unlike forward_only mode), so caches accumulate across
        # iterations. We reset them here to ensure each iteration starts clean.
        for _mc_idx in range(len(model)):
            clear_key_value_cache(model[_mc_idx], config.mtp_num_layers)

    if is_sft_chunkpipe:
        # group_size_cache: progressively records discovered group info
        # IMPORTANT: Per-VP-stage (per-model-chunk) to ensure consistency.
        # Each VP stage independently tracks group info for the data it processes.
        group_size_cache = [{} for _ in range(len(model))]
        # forward_chunk_info: maps (model_chunk_id, chunkpipe_forward_microbatch) to
        # (chunk_idx_in_group, group_size) for backward lookup.
        forward_chunk_info = {}
        # Per-model-chunk sequential counter for tracking forward progress
        chunkpipe_seq_counter = [0] * len(model)
        # forward_group_info: per-VP-stage list of (start_fwd_mb, group_size)
        # Records group structure for backward scheduling. Groups are discovered
        # in forward order, and backward will process them in reverse order.
        forward_group_info = [[] for _ in range(len(model))]
        # forward_chunk_queue: per-model-chunk per-microbatch queue of
        # (chunkpipe_forward_microbatch, chunk_idx_in_group, group_size).
        # For SFT VPP, we use a different approach: backward is scheduled by
        # group structure, not by LIFO order. This queue stores forward info
        # indexed by fwd_mb for backward lookup.
        forward_chunk_queue = [{} for _ in range(len(model))]
        # vp_bwd_group_queue: per-VP-stage dict of group_id -> list of fwd_mb.
        # Populated during forward. Used for group-aware backward scheduling.
        vp_bwd_group_queue = [{} for _ in range(len(model))]
        # vp_bwd_pending_mb: per-VP-stage list tracking next backward mb to execute.
        # Filled dynamically when backward_preprocess is called.
        vp_bwd_pending_mb = [[] for _ in range(len(model))]
        # backward_group_state: per-VP-stage dict tracking backward progress.
        # Key: group_id, Value: set of completed chunk indices within group.
        # Used to determine when entire group's backward is done for cache cleanup.
        backward_group_state = [{} for _ in range(len(model))]

        def _get_group_id_from_cache(mb_id, _group_size_cache, chunk_num_per_seq):
            """Compute group_id from known group_size_cache."""
            if not _group_size_cache:
                return mb_id // chunk_num_per_seq
            acc = 0
            for gid, gs in sorted(_group_size_cache.items()):
                if mb_id < acc + gs:
                    return gid
                acc += gs
            return len(_group_size_cache) + (mb_id - acc) // chunk_num_per_seq

        def _get_chunk_pos_from_cache(mb_id, group_id, _group_size_cache, chunk_num_per_seq):
            """Get position of current chunk within its group (0-based)."""
            if not _group_size_cache:
                return mb_id % chunk_num_per_seq
            acc = 0
            for gid, gs in sorted(_group_size_cache.items()):
                if gid == group_id:
                    return mb_id - acc
                acc += gs
            return (mb_id - acc) % chunk_num_per_seq

        def _is_last_chunk_of_group_dynamic(mb_id, _group_size_cache, chunk_num_per_seq):
            """Check if current chunk is the last in its group."""
            if not _group_size_cache:
                return (mb_id + 1) % chunk_num_per_seq == 0
            acc = 0
            for gid, gs in sorted(_group_size_cache.items()):
                if mb_id < acc + gs:
                    return mb_id == acc + gs - 1
                acc += gs
            return (mb_id + 1) % chunk_num_per_seq == 0

    num_chunks = config.chunk_num_per_seq
    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    # SFT VPP Chunkpipe: Use dict for direct fwd_mb access instead of LIFO list.
    # The backward execution order MUST follow the VPP schedule to keep P2P
    # communication aligned across pipeline stages. Group-based ordering would
    # break P2P gradient alignment. Instead, group-awareness is only applied
    # to KV cache deletion (delete all caches for a group when the group's
    # entire backward is complete).
    if is_sft_chunkpipe:
        input_tensors = [{} for _ in range(len(model))]
        output_tensors = [{} for _ in range(len(model))]
    else:
        input_tensors = [[[] for _ in range(num_microbatches // num_chunks)] for _ in range(len(model))]
        output_tensors = [[[] for _ in range(num_microbatches // num_chunks)] for _ in range(len(model))]
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    forward_data_store = []
    output_tensor_grads = None
    if not forward_only:
        if is_sft_chunkpipe:
            output_tensor_grads = [{} for _ in range(len(model))]
            # p2p_grad_queue: FIFO queue for cross-rank P2P gradients.
            # The sender (last VP stage) determines group-aware backward order,
            # and receivers (non-last VP stages) follow FIFO order.
            # This replaces the dict-based storage that required key prediction.
            p2p_grad_queue = [[] for _ in range(len(model))]
        else:
            output_tensor_grads = [[[] for _ in range(num_microbatches // num_chunks)] for _ in range(len(model))]
    else:
        output_tensor_grads = None

    pipeline_parallel_size = p2p_communicator.pp_group.size()
    pipeline_parallel_rank = p2p_communicator.pp_group.rank()

    if not is_sft_chunkpipe:
        # SFT chunkpipe uses override_mgspvs (all sequences in one VP group),
        # so these standard VPP constraints are inapplicable for SFT.
        # Pretrain chunkpipe still enforces them.
        if (
                config.microbatch_group_size_per_vp_stage > num_microbatches
                or config.microbatch_group_size_per_vp_stage < pipeline_parallel_size
        ):
            msg = (
                'The number of contiguous micro-batches in a virtual pipeline stage'
                f'should range in [PP={pipeline_parallel_size} , M={num_microbatches}]'
            )
            raise ValueError(msg)

        final_microbatch_group_size = num_microbatches % config.microbatch_group_size_per_vp_stage
        if 0 < final_microbatch_group_size < pipeline_parallel_size:
            msg = 'The remainder of M (the total micro-batches) divided by N (number of '
            msg += 'contiguous micro-batches in a virtual pipeline stage) should be 0, '
            msg += 'or larger than or equal to the pipeline-parallel size, but it is '
            msg += f'{final_microbatch_group_size}. '
            msg += 'Otherwise, it introduces dependency bubbles in the pipeline '
            msg += 'and reduces throughput.'
            raise RuntimeError(msg)

    model_type = get_model_type(model[0])

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // cp_group.size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // tp_group.size()

    # Compute number of warmup and remaining microbatches.
    # seems only used for vpp
    num_model_chunks = len(model)

    # SFT chunkpipe: put all sequences in one VP group (override_mgspvs = num_sequences)
    # so VP block boundaries never split a group. Warmup effective_chunks ensures VP1
    # forwards stay ahead of backward. Formula: (num_microbatches + num_chunks + 1) // 2
    _sft_effective_chunks = num_chunks
    _sft_table_mgspvs = None
    if config.sft_chunkpipe_mode and num_chunks > 1:
        _num_sequences = num_microbatches // num_chunks
        _sft_table_mgspvs = _num_sequences
        _sft_effective_chunks = (num_microbatches + num_chunks + 1) // 2
        _sft_effective_chunks = max(_sft_effective_chunks, num_chunks)

    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches,
        num_model_chunks,
        config.microbatch_group_size_per_vp_stage,
        forward_only=forward_only,
        overlap_moe_expert_parallel_comm=config.overlap_moe_expert_parallel_comm,
        p2p_communicator=p2p_communicator,
        enable_chunkpipe=config.enable_chunkpipe,
        num_chunks=_sft_effective_chunks,
    )

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    # Create a tunable schedule lookup table.
    # The schedule lookup table uses the virtual_microbatch_id to find the corresponding
    # microbatch_id and model_chunk_id. For example, the tunable schedule table for
    # PP2 N3M5 with VP2 is constructed as below:
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 0 0 0 1 1 1 1 2 2
    # model_chunk_id        | 0 0 1 1 0 0 1 1 0 0
    # chunk_id              | 0 1 0 1 0 1 0 1 0 1
    schedule_table = get_extended_schedule_table(
        num_microbatches, len(model), config.microbatch_group_size_per_vp_stage, num_chunks,
        override_mgspvs=_sft_table_mgspvs
    )
    microbatch_id_table, model_chunk_id_table, sequence_id_table = zip(*schedule_table)

    def get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        if virtual_microbatch_id >= total_num_microbatches:
            return 0 if forward else num_model_chunks - 1
        if forward:
            model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        else:
            if is_sft_chunkpipe:
                model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
            else:
                model_chunk_id = int(virtual_microbatch_id / num_chunks % num_model_chunks)
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        if iteration_id >= total_num_microbatches:
            return -1
        if not forward and not is_sft_chunkpipe:
            microbatch_id_in_model_chunk = int(iteration_id / num_chunks / num_model_chunks)
        else:
            microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def get_sequence_id_in_microbatch(iteration_id, forward):
        """Helper method to get the sequence_id in microbatch  given the iteration number."""
        if not forward:
            sequence_id_in_microbatch = (-iteration_id - 1) % num_chunks
        else:
            sequence_id_in_microbatch = sequence_id_table[iteration_id]
        return sequence_id_in_microbatch

    def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
        """Helper method to count number of released (i.e. popped from input_tensors)
        microbatches for a model chunk."""
        if forward_only:  # Micro-batch is released after forward prop.
            return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
        else:  # Micro-batch is released after backward prop.
            # Zero backward prop in warmup.
            if virtual_microbatch_id < num_warmup_microbatches:
                return 0
            else:
                backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
                model_chunk_id = num_model_chunks - model_chunk_id - 1
                return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)

    def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            # 必须是第一个 microbatch 的第一个 chunk 才算真正的 "first"
            return (microbatch_id_table[virtual_microbatch_id] == 0
                    and sequence_id_table[virtual_microbatch_id] == 0)
        else:
            return False

    def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            # 必须是最后一个 microbatch 的最后一个 chunk 才算真正的 "last"
            adjusted_num_microbatches = num_microbatches // num_chunks
            return (microbatch_id_table[virtual_microbatch_id] == adjusted_num_microbatches - 1
                    and sequence_id_table[virtual_microbatch_id] == num_chunks - 1)
        else:
            return False

    def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
        """Determine if peers are sending, and where in data structure
        to put received tensors.
        Return a boolean if the pipeline stage expects to recv from peers, and the
        corresponding model_chunk_id for the received tensor.
        """
        recv = True
        # The leading pipeline stage is the first rank in fwd and the last rank in bwd.
        is_leading_pipeline_stage = (
            is_pp_first_stage(p2p_communicator.pp_group)
            if forward
            else is_pp_last_stage(p2p_communicator.pp_group)
        )

        last_model_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading_pipeline_stage:
            # The leading pipeline stage is ahead of the ending pipeline stage
            # (i.e. last rank in fwd and first rank in bwd) by (pipeline_parallel_size - 1).
            # Let's consider bwd as an example with PP 4:
            #       0 1 2 3 ...
            #     0 1 2 3 ...
            #   0 1 2 3 ...
            # 0 1 2 3 ...
            if virtual_microbatch_id < (pipeline_parallel_size - 1):
                # The ending stage has not produced any tensors, so no recv will be initiated.
                recv = False
                next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
                next_microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id + 1, forward)
            else:
                # Find the model chunk of the aligned microbatches in the ending stage.
                # For example, microbatch 0 in the ending stage is aligned with microbatch 3
                # in the leading stage.
                next_model_chunk_id = get_model_chunk_id(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )
                next_microbatch_id = get_microbatch_id_in_model_chunk(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )
            # Last model chunk in the final stage does not produce tensors.
            if next_model_chunk_id == last_model_chunk:
                recv = False
            if forward:
                # Model chunk id increases in forward.
                next_model_chunk_id += 1
            else:
                # Model chunk id decreases in backward.
                next_model_chunk_id -= 1
        else:
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
            next_microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id + 1, forward)

        return recv, next_model_chunk_id, next_microbatch_id

    def forward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id, microbatch_id, chunk_id):
        """Preprocess for forward_step_helper"""
        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
            if (
                param_sync_virtual_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
            ):
                param_sync_chunk_id = (
                    get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
                )
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        chunkpipe_forward_microbatch = microbatch_id * num_chunks + chunk_id
        if _is_vp_first_stage(vp_stage=model_chunk_id) and is_pp_first_stage(pp_group):
            if is_sft_chunkpipe:
                # SFT: use fwd_mb as key
                if chunkpipe_forward_microbatch not in output_tensors[model_chunk_id]:
                    input_tensors[model_chunk_id][chunkpipe_forward_microbatch] = None
            else:
                if len(input_tensors[model_chunk_id][microbatch_id]) == len(
                        output_tensors[model_chunk_id][microbatch_id]):
                    input_tensors[model_chunk_id][microbatch_id].append(None)

        # For non-depth-first pipeline schedules, the first rank would buffer multiple received
        # activation tensors for a model chunk until accessed during warmup.
        # This input buffering is needed to overlap the computation with the receipt of
        # the next inputs. To index the proper buffered inputs for forword_step, we use
        # microbatch_id offset with number of released microbatches that have completed backprop.
        if is_sft_chunkpipe:
            # SFT: use fwd_mb as key
            input_tensor = input_tensors[model_chunk_id].get(chunkpipe_forward_microbatch)
        else:
            offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
            input_tensor = input_tensors[model_chunk_id][microbatch_id][chunk_id]

        # add for config.reduce_variable_seq_shape_p2p_comm, remove padding.
        if config.reduce_variable_seq_shape_p2p_comm:
            input_tensor = p2p_comm_tensor_remove_padding(input_tensor,
                                                          config.p2p_comm_fixed_seq_lengths_per_rank)
            if is_sft_chunkpipe:
                input_tensors[model_chunk_id][chunkpipe_forward_microbatch] = input_tensor
            else:
                input_tensors[model_chunk_id][microbatch_id][-1] = input_tensor

        return input_tensor

    def forward_step_helper_postprocess(model_chunk_id, microbatch_id, chunk_id, output_tensor, num_tokens):
        """Postprocess for forward_step_helper"""
        chunkpipe_forward_microbatch = microbatch_id * num_chunks + chunk_id
        if is_sft_chunkpipe:
            # SFT: use fwd_mb as key
            output_tensors[model_chunk_id][chunkpipe_forward_microbatch] = output_tensor
        else:
            output_tensors[model_chunk_id][microbatch_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens

        # If forward-only, no need to save tensors for a backward pass.
        if forward_only:
            # Release the tensor that have completed forward step.
            if is_sft_chunkpipe:
                del input_tensors[model_chunk_id][chunkpipe_forward_microbatch]
                del output_tensors[model_chunk_id][chunkpipe_forward_microbatch]
            else:
                input_tensors[model_chunk_id][microbatch_id].pop(0)
                output_tensors[model_chunk_id][microbatch_id].pop()
            if is_sft_chunkpipe:
                _info = forward_chunk_info.get((model_chunk_id, microbatch_id * num_chunks + chunk_id))
                if _info is not None:
                    _is_last = _info[0] + 1 == _info[1]  # chunk_idx + 1 == group_size
                else:
                    _is_last = chunk_id == num_chunks - 1  # fallback
            else:
                _is_last = chunk_id == num_chunks - 1
            if _is_last:
                clear_key_value_cache(model[model_chunk_id], config.mtp_num_layers)

        return

    def forward_step_helper(virtual_microbatch_id, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks"""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
        microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id, forward=True)
        chunk_id = get_sequence_id_in_microbatch(virtual_microbatch_id, forward=True)
        chunkpipe_forward_microbatch = microbatch_id * num_chunks + chunk_id

        input_tensor = forward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id, microbatch_id, chunk_id
        )

        if config.enable_chunkpipe:
            decoder = get_attr_wrapped_model(model[model_chunk_id], 'decoder')
            decoder.update_config(True, chunkpipe_forward_microbatch)
            if is_sft_chunkpipe:
                _seq_idx = chunkpipe_seq_counter[model_chunk_id]
                _gid = _get_group_id_from_cache(_seq_idx, group_size_cache[model_chunk_id], config.chunk_num_per_seq)
                if _gid in group_size_cache[model_chunk_id]:
                    # Group is known: compute exact chunk position
                    config.chunkpipe_chunk_idx_in_group = _get_chunk_pos_from_cache(
                        _seq_idx, _gid, group_size_cache[model_chunk_id], config.chunk_num_per_seq
                    )
                    if config.chunkpipe_chunk_idx_in_group > 0:
                        config.chunkpipe_current_group_size = group_size_cache[model_chunk_id][_gid]
                else:
                    # Group is unknown (not yet in cache): conservatively treat as first
                    # chunk of a new group. This ensures append_chunk_key_value_cache will
                    # cache this chunk (chunk_idx=0 < any group_size - 1).
                    # get_batch() will write the true group_size to chunkpipe_current_group_size
                    # (Fix in sft_llm.py), which the post-step code will use to populate
                    # group_size_cache correctly.
                    config.chunkpipe_chunk_idx_in_group = 0
            else:
                # Pretrain: use chunk_id directly
                config.chunkpipe_chunk_idx_in_group = chunk_id

        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
            vp_stage=model_chunk_id,
            is_last_stage=_is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group),
        )

        # SFT: discover group_size from get_batch() and update cache
        if is_sft_chunkpipe:
            _discovered_gs = getattr(config, 'chunkpipe_current_group_size', None)
            _seq_idx = chunkpipe_seq_counter[model_chunk_id]
            _gid = _get_group_id_from_cache(_seq_idx, group_size_cache[model_chunk_id], config.chunk_num_per_seq)
            if _discovered_gs is not None and _discovered_gs > 0 and _gid not in group_size_cache[model_chunk_id]:
                group_size_cache[model_chunk_id][_gid] = _discovered_gs
                # Record new group info for backward scheduling
                # Calculate start_fwd_mb for this group
                _start_fwd_mb = sum(gs for g, gs in sorted(group_size_cache[model_chunk_id].items()) if g < _gid)
                forward_group_info[model_chunk_id].append((_start_fwd_mb, _discovered_gs))
            # Re-compute chunk_idx_in_group using the now-accurate group_size_cache.
            _corrected_gid = _get_group_id_from_cache(_seq_idx, group_size_cache[model_chunk_id],
                                                      config.chunk_num_per_seq)
            _corrected_chunk_idx = _get_chunk_pos_from_cache(
                _seq_idx, _corrected_gid, group_size_cache[model_chunk_id], config.chunk_num_per_seq
            )
            config.chunkpipe_chunk_idx_in_group = _corrected_chunk_idx
            _final_group_size = group_size_cache[model_chunk_id].get(_corrected_gid, config.chunk_num_per_seq)
            # Store info for backward lookup indexed by fwd_mb
            forward_chunk_info[(model_chunk_id, chunkpipe_forward_microbatch)] = (
                _corrected_chunk_idx,
                _final_group_size,
            )
            # Store in forward_chunk_queue indexed by fwd_mb for backward lookup
            if not forward_only:
                forward_chunk_queue[model_chunk_id][chunkpipe_forward_microbatch] = (
                    _corrected_chunk_idx,
                    _final_group_size
                )
                # Populate backward queue for group-aware scheduling
                if _corrected_gid not in vp_bwd_group_queue[model_chunk_id]:
                    vp_bwd_group_queue[model_chunk_id][_corrected_gid] = []
                vp_bwd_group_queue[model_chunk_id][_corrected_gid].append(chunkpipe_forward_microbatch)
            chunkpipe_seq_counter[model_chunk_id] += 1

        forward_step_helper_postprocess(model_chunk_id, microbatch_id, chunk_id, output_tensor, num_tokens)

        return output_tensor

    def backward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id, microbatch_id):
        """Preprocess for backward_step_helper"""
        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
            virtual_microbatch_id
        ):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        # SFT: group-aware backward scheduling.
        # FIFO across groups (get_min_key), LIFO within group (list.pop).
        # Uses vp_bwd_pending_mb list, filled from vp_bwd_group_queue when empty.
        if is_sft_chunkpipe:
            # For LAST VP stage: use vp_bwd_group_queue for group-aware backward order.
            # For NON-LAST VP stages: receive gradient via same-rank VP P2P from
            # the previous VP stage (stored in output_tensor_grads by backward_step_helper).
            # Cross-rank P2P gradients are stored in p2p_grad_queue (FIFO).

            is_last_vp = _is_vp_last_stage(vp_stage=model_chunk_id)
            is_pp_last = is_pp_last_stage(pp_group)

            if is_last_vp:
                # Last VP stage: determines backward order using vp_bwd_group_queue
                if not vp_bwd_pending_mb[model_chunk_id]:
                    _min_group = get_min_key(vp_bwd_group_queue[model_chunk_id])
                    _chunks_list = vp_bwd_group_queue[model_chunk_id][_min_group]
                    _expected_gs = group_size_cache[model_chunk_id].get(_min_group)
                    assert _expected_gs is None or len(_chunks_list) == _expected_gs, (
                        f"[ChunkPipe] Attempting backward on incomplete group {_min_group} "
                        f"(have {len(_chunks_list)} chunks, need {_expected_gs}). "
                        f"VP{model_chunk_id}, rank{pipeline_parallel_rank}. "
                        f"This indicates warmup is too short for chunk_num_per_seq."
                    )
                    # LIFO within group: pop from end of list (last chunk first)
                    vp_bwd_pending_mb[model_chunk_id] = _chunks_list
                    del vp_bwd_group_queue[model_chunk_id][_min_group]
                chunkpipe_backward_microbatch = vp_bwd_pending_mb[model_chunk_id].pop()

                # For VP last stage on PP last rank, no incoming gradient — set to None
                # so backward_step generates the initial gradient from loss.
                # For VP last stage on non-PP-last rank, gradient comes from cross-rank P2P (FIFO queue).
                if is_pp_last:
                    if chunkpipe_backward_microbatch not in output_tensor_grads[model_chunk_id]:
                        output_tensor_grads[model_chunk_id][chunkpipe_backward_microbatch] = None
                    output_tensor_grad = output_tensor_grads[model_chunk_id].pop(chunkpipe_backward_microbatch)
                else:
                    # Cross-rank P2P: use FIFO queue
                    output_tensor_grad = p2p_grad_queue[model_chunk_id].pop(0)
            else:
                # Non-last VP stage: gradient comes from cross-rank P2P (p2p_grad_queue).
                # In VPP interleaved, ALL VP stage gradients are transferred via cross-rank P2P.
                # The backward order is determined by the sender (next rank), so we follow FIFO.
                # Use vp_bwd_group_queue to get the fwd_mb for this backward step.
                if not vp_bwd_pending_mb[model_chunk_id]:
                    _min_group = get_min_key(vp_bwd_group_queue[model_chunk_id])
                    _chunks_list = vp_bwd_group_queue[model_chunk_id][_min_group]
                    _expected_gs = group_size_cache[model_chunk_id].get(_min_group)
                    assert _expected_gs is None or len(_chunks_list) == _expected_gs, (
                        f"[ChunkPipe] Attempting backward on incomplete group {_min_group} "
                        f"(have {len(_chunks_list)} chunks, need {_expected_gs}). "
                        f"VP{model_chunk_id}, rank{pipeline_parallel_rank}. "
                        f"This indicates warmup is too short for chunk_num_per_seq."
                    )
                    # LIFO within group: pop from end of list (last chunk first)
                    vp_bwd_pending_mb[model_chunk_id] = _chunks_list
                    del vp_bwd_group_queue[model_chunk_id][_min_group]
                chunkpipe_backward_microbatch = vp_bwd_pending_mb[model_chunk_id].pop()
                output_tensor_grad = p2p_grad_queue[model_chunk_id].pop(0)

            input_tensor = input_tensors[model_chunk_id].pop(chunkpipe_backward_microbatch)
            output_tensor = output_tensors[model_chunk_id].pop(chunkpipe_backward_microbatch)

            # add for reduce_variable_seq_shape_p2p_comm, remove padding.
            if config.reduce_variable_seq_shape_p2p_comm:
                output_tensor_grad = p2p_comm_tensor_remove_padding(output_tensor_grad,
                                                                    config.p2p_comm_fixed_seq_lengths_per_rank)

            return input_tensor, output_tensor, output_tensor_grad, chunkpipe_backward_microbatch
        else:
            # pylint: disable=E0606
            if _is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group):
                if len(output_tensor_grads[model_chunk_id][microbatch_id]) == 0:
                    output_tensor_grads[model_chunk_id][microbatch_id].append(None)

            input_tensor = input_tensors[model_chunk_id][microbatch_id].pop()
            output_tensor = output_tensors[model_chunk_id][microbatch_id].pop()
            output_tensor_grad = output_tensor_grads[model_chunk_id][microbatch_id].pop(0)

        # add for reduce_variable_seq_shape_p2p_comm, remove padding.
        if config.reduce_variable_seq_shape_p2p_comm:
            output_tensor_grad = p2p_comm_tensor_remove_padding(output_tensor_grad,
                                                                config.p2p_comm_fixed_seq_lengths_per_rank)

        return input_tensor, output_tensor, output_tensor_grad, None

    def backward_step_helper_postprocess(virtual_microbatch_id):
        """Postprocess for backward_step_helper"""
        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
            if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_virtual_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(
                    grad_sync_virtual_microbatch_id, forward=False
                )
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

    def backward_step_helper(virtual_microbatch_id):
        """Helper method to run backward step with model split into chunks"""
        nonlocal output_tensor_grads
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)
        microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id, forward=False)
        chunk_id = get_sequence_id_in_microbatch(virtual_microbatch_id, forward=False)
        chunkpipe_backward_microbatch = microbatch_id * num_chunks + chunk_id

        # Get tensors and fwd_mb (for SFT) from preprocess
        input_tensor, output_tensor, output_tensor_grad, sft_fwd_mb = backward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id, microbatch_id
        )

        # SFT: use fwd_mb from backward preprocess to look up chunk info
        sft_chunk_idx = None
        sft_group_size = None
        if is_sft_chunkpipe:
            sft_bwd_mb = sft_fwd_mb  # Use the actual fwd_mb from backward preprocess
            # forward_chunk_queue is indexed by fwd_mb
            chunk_info = forward_chunk_queue[model_chunk_id].get(sft_bwd_mb)
            if chunk_info is not None:
                sft_chunk_idx, sft_group_size = chunk_info
        else:
            sft_bwd_mb = chunkpipe_backward_microbatch

        # 新增chunkpipe必须配置项，传入backward_step
        if config.enable_chunkpipe:
            if is_sft_chunkpipe:
                decoder = get_attr_wrapped_model(model[model_chunk_id], 'decoder')
                decoder.update_config(False, sft_bwd_mb)
                if sft_chunk_idx is not None:
                    config.chunkpipe_chunk_idx_in_group = sft_chunk_idx
                    config.chunkpipe_current_group_size = sft_group_size
            else:
                decoder = get_attr_wrapped_model(model[model_chunk_id], 'decoder')
                decoder.update_config(False, chunkpipe_backward_microbatch)
                config.chunkpipe_chunk_idx_in_group = chunk_id

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # SFT: manage KV cache deletion by group
        if is_sft_chunkpipe and sft_chunk_idx is not None and sft_group_size is not None:
            # NOTE: In PP2 VP2, all VP stage gradients are transferred via cross-rank P2P,
            # not same-rank VP P2P. The gradient (input_tensor_grad) is sent by
            # pp_post_backward() to the previous rank. Do NOT store to output_tensor_grads here.
            # Track backward progress for this group
            _gid = None
            for gid, (start_mb, gs) in enumerate(forward_group_info[model_chunk_id]):
                if start_mb <= sft_bwd_mb < start_mb + gs:
                    _gid = gid
                    break

            if _gid is not None:
                if _gid not in backward_group_state[model_chunk_id]:
                    backward_group_state[model_chunk_id][_gid] = set()
                backward_group_state[model_chunk_id][_gid].add(sft_chunk_idx)

                # Check if all chunks in this group have completed backward
                _completed_chunks = backward_group_state[model_chunk_id][_gid]
                if len(_completed_chunks) == sft_group_size:
                    # Verify no orphan cache grads (LIFO backward invariant check)
                    _decoder = get_attr_wrapped_model(model[model_chunk_id], "decoder")
                    for _layer in _decoder.layers:
                        _layer.self_attention.check_kv_cache_grad_consumed()
                    # All chunks in group have completed backward, delete all caches
                    _start_mb = forward_group_info[model_chunk_id][_gid][0]
                    for _del_mb in range(_start_mb, _start_mb + sft_group_size):
                        remove_key_value_cache(model[model_chunk_id], _del_mb, config.mtp_num_layers)
        else:
            # Pretrain or non-SFT: immediate deletion
            remove_key_value_cache(model[model_chunk_id],
                                   sft_bwd_mb if is_sft_chunkpipe else chunkpipe_backward_microbatch,
                                   config.mtp_num_layers)

        backward_step_helper_postprocess(virtual_microbatch_id)

        return input_tensor_grad

    def forward_backward_helper_wrapper(
        f_virtual_microbatch_id=None,
        b_virtual_microbatch_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        checkpoint_activations_microbatch=None,
    ):
        """
        wrap forward_helper, backward_helper, and combined_forward_backward_helper in a unified way
        """
        if config.overlap_moe_expert_parallel_comm and not forward_only:  # Combined 1F1B path
            return combined_1f1b_schedule_for_interleaved_pipelining(
                config,
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                forward_data_store,
                forward_step_helper_preprocess,
                forward_step_helper_postprocess,
                backward_step_helper_preprocess,
                backward_step_helper_postprocess,
                get_microbatch_id_in_model_chunk,
                get_model_chunk_id,
                partial(check_first_val_step, first_val_step, forward_only),
                is_first_microbatch_for_model_chunk,
                collect_non_loss_data,
                f_virtual_microbatch_id=f_virtual_microbatch_id,
                b_virtual_microbatch_id=b_virtual_microbatch_id,
                pre_forward=pre_forward,
                pre_backward=pre_backward,
                post_forward=post_forward,
                post_backward=post_backward,
            )
        else:  # Conventional interleaved 1F1B path
            forward_output_tensor = None
            backward_input_tensor_grad = None
            # forward pass
            if f_virtual_microbatch_id is not None:
                forward_model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
                if pre_forward is not None:
                    pre_forward()
                forward_output_tensor = forward_step_helper(
                    f_virtual_microbatch_id, checkpoint_activations_microbatch
                )
                if post_forward is not None:
                    forward_output_tensor = post_forward(forward_output_tensor)

            # Backward pass.
            if b_virtual_microbatch_id is not None:
                backward_model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
                if pre_backward is not None:
                    pre_backward()
                backward_input_tensor_grad = backward_step_helper(b_virtual_microbatch_id)
                if post_backward is not None:
                    backward_input_tensor_grad = post_backward(backward_input_tensor_grad)
            return forward_output_tensor, backward_input_tensor_grad

    # ==============================main logic=========================================
    _is_vp_first_stage = partial(
        is_vp_first_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    _is_vp_last_stage = partial(
        is_vp_last_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    pp_group = p2p_communicator.pp_group

    # Run warmup forward passes.
    nvtx_range_push(suffix="warmup")
    # For SFT: use dict operation; for non-SFT: use list operation
    if is_sft_chunkpipe:
        # Initial forward tensor for VP stage 0, fwd_mb = 0
        # This tensor will be consumed by forward_step_helper_preprocess
        input_tensors[0][0] = p2p_communicator.recv_forward(
            tensor_shape, _is_vp_first_stage(vp_stage=0) and is_pp_first_stage(pp_group)
        )
    else:
        input_tensors[0][0].append(
            p2p_communicator.recv_forward(
                tensor_shape, _is_vp_first_stage(vp_stage=0) and is_pp_first_stage(pp_group)
            )
        )

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if is_pp_first_stage(p2p_communicator.pp_group):
        fwd_recv_buffer_size = max(1,
            num_chunks - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if is_pp_last_stage(p2p_communicator.pp_group):
        bwd_recv_buffer_size = max(1,
            num_chunks - pipeline_parallel_size + 1
        )
    else:
        bwd_recv_buffer_size = 1
    fwd_recv_buffer = [None] * fwd_recv_buffer_size
    bwd_recv_buffer = [None] * bwd_recv_buffer_size
    recv_prev_wait_handles = []
    send_next_wait_handle = None
    send_prev_wait_handle = None
    recv_next_wait_handles = []

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)
        cur_microbatch_id_in_chunk = get_microbatch_id_in_model_chunk(k, forward=True)

        # Wait for pending async recv handles from previous iteration's post-forward.
        # When overlap_p2p_comm=True but overlap_p2p_comm_warmup_flush=False,
        # the warmup uses async recv (lines 2186-2202) but never waits on the
        # handles. This causes the next forward step to read incomplete tensor data.
        if not config.overlap_p2p_comm_warmup_flush and recv_prev_wait_handles:
            recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
            recv_prev_wait_handle.wait()

        if config.overlap_p2p_comm_warmup_flush:
            if (
                not (
                    _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group)
                )
                and k != 0
            ):
                assert recv_prev_wait_handles, (
                    f'pp rank {pipeline_parallel_rank}, iteration {k},'
                    'should have registered recv handle'
                )
                recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                recv_prev_wait_handle.wait()

        # Determine if tensor should be received from previous stage.
        recv_prev, next_forward_model_chunk_id, next_forward_microbatch_id = \
            recv_tensor_from_previous_stage(k, forward=True)

        # No receive in last iteration when recv iteration k+1.
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Prefetch recv for iteration k+1 for non-first ranks.
        if config.overlap_p2p_comm_warmup_flush and not is_pp_first_stage(
            p2p_communicator.pp_group
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communicator.send_forward_recv_forward(
                    output_tensor=None,  # No output_tensor to send.
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )
            )

            if fwd_wait_recv_handles:
                recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        output_tensor, _ = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=k,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # Don't send tensor downstream if on last stage.
        if _is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group):
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm_warmup_flush:
            if (
                k == (num_warmup_microbatches - 1)
                and not config.overlap_p2p_comm
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communicator.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                    )
                )
                # SFT: use FIFO queue for cross-rank P2P gradient storage.
                if is_sft_chunkpipe:
                    p2p_grad_queue[num_model_chunks - 1].append(output_tensor_grad)
                else:
                    output_tensor_grads[num_model_chunks - 1][0].append(output_tensor_grad)
            else:
                input_tensor = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape
                )
            if recv_prev:
                # SFT: The received P2P tensor is the previous PP rank's output from step k-stagger
                # (due to P2P stagger of pipeline_parallel_size-1). Its fwd_mb equals
                # the sender's chunkpipe_forward_microbatch at step k-stagger.
                if is_sft_chunkpipe:
                    _stagger = pipeline_parallel_size - 1
                    _sender_virt_mb = k - _stagger if is_pp_first_stage(pp_group) else k + 1
                    _prev_fwd_mb = (get_microbatch_id_in_model_chunk(_sender_virt_mb, forward=True) * num_chunks
                                    + get_sequence_id_in_microbatch(_sender_virt_mb, forward=True))
                    input_tensors[next_forward_model_chunk_id][_prev_fwd_mb] = input_tensor
                else:
                    input_tensors[next_forward_model_chunk_id][next_forward_microbatch_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not is_pp_first_stage(p2p_communicator.pp_group):
                # Send only since recv prefetched.
                _, fwd_wait_handles = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=False, tensor_shape=tensor_shape, overlap_p2p_comm=True
                )
            else:  # No prefetch for first rank, so both send and recv initiated.
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            if recv_prev:
                # SFT: The received P2P tensor is the previous PP rank's output from step k-stagger
                # (due to P2P stagger of pipeline_parallel_size-1). Its fwd_mb equals
                # the sender's chunkpipe_forward_microbatch at step k-stagger.
                if is_sft_chunkpipe:
                    _stagger = pipeline_parallel_size - 1
                    _sender_virt_mb = k - _stagger if is_pp_first_stage(pp_group) else k + 1
                    _prev_fwd_mb = (get_microbatch_id_in_model_chunk(_sender_virt_mb, forward=True) * num_chunks
                                    + get_sequence_id_in_microbatch(_sender_virt_mb, forward=True))
                    _buf_val = fwd_recv_buffer[k % fwd_recv_buffer_size]
                    input_tensors[next_forward_model_chunk_id][_prev_fwd_mb] = _buf_val
                else:
                    input_tensors[next_forward_model_chunk_id][next_forward_microbatch_id].append(
                        fwd_recv_buffer[k % fwd_recv_buffer_size]
                    )
                fwd_recv_buffer[(k + 1) % fwd_recv_buffer_size] = None

        if config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                if recv_next:
                    # SFT: use FIFO queue for cross-rank P2P gradient storage.
                    if is_sft_chunkpipe:
                        p2p_grad_queue[num_model_chunks - 1].append(bwd_recv_buffer[-1])
                    else:
                        output_tensor_grads[num_model_chunks - 1][0].append(bwd_recv_buffer[-1])

    nvtx_range_pop(suffix="warmup")

    # Run 1F1B in steady state.
    nvtx_range_push(suffix="steady")
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        cur_microbatch_id_in_chunk = get_microbatch_id_in_model_chunk(forward_k, forward=True)
        if config.overlap_p2p_comm:
            backward_k = k

            # Sync forward recv
            def pp_pre_forward(vp_stage=None):
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                if not (_is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_prev_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, fwd iteration {forward_k}, '
                            'should have registered recv handle'
                        )
                        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                        recv_prev_wait_handle.wait()
                    else:
                        if recv_prev_wait_handles is not None and recv_prev_wait_handles:
                            recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                            recv_prev_wait_handle.wait()

                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Async forward send / receive
            def pp_post_forward(output_tensor, vp_stage=None):
                nonlocal send_next_wait_handle
                nonlocal fwd_recv_buffer
                nonlocal fwd_wait_handles
                nonlocal recv_prev_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                # Last virtual stage no activation tensor to send.
                if _is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group):
                    output_tensor = None

                recv_prev, next_forward_model_chunk_id, next_forward_microbatch_id = recv_tensor_from_previous_stage(
                    forward_k, forward=True
                )

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_next_wait_handle is not None:
                    send_next_wait_handle.wait()
                if fwd_wait_handles is not None:
                    send_next_wait_handle = (
                        fwd_wait_handles.pop("send_next")
                        if "send_next" in fwd_wait_handles
                        else None
                    )
                    if "recv_prev" in fwd_wait_handles:
                        recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
                # assert fwd_wait_handles is not None

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.
                if recv_prev:
                    # SFT: The received P2P tensor is the previous PP rank's output from step forward_k-stagger
                    # (due to P2P stagger of pipeline_parallel_size-1). Its fwd_mb equals
                    # the sender's chunkpipe_forward_microbatch at step forward_k-stagger.
                    if is_sft_chunkpipe:
                        _stagger = pipeline_parallel_size - 1
                        _sender_virt_mb = forward_k - _stagger if is_pp_first_stage(pp_group) else forward_k + 1
                        _prev_fwd_mb = (get_microbatch_id_in_model_chunk(_sender_virt_mb, forward=True) * num_chunks
                                        + get_sequence_id_in_microbatch(_sender_virt_mb, forward=True))
                        _buf_val = fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                        input_tensors[next_forward_model_chunk_id][_prev_fwd_mb] = _buf_val
                    else:
                        input_tensors[next_forward_model_chunk_id][next_forward_microbatch_id].append(
                            fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                        )
                    fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None

                return output_tensor

            # Sync backward recv
            def pp_pre_backward(vp_stage=None):
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                if not (_is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_next_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, bwd iteration {backward_k}, '
                            'should have registered recv next handle'
                        )
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()
                    else:
                        if recv_next_wait_handles is not None and recv_next_wait_handles:
                            recv_next_wait_handle = recv_next_wait_handles.pop(0)
                            recv_next_wait_handle.wait()

            # Async backward send / receive
            def pp_post_backward(input_tensor_grad, vp_stage=None):
                nonlocal send_prev_wait_handle
                nonlocal bwd_wait_handles
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                # First virtual stage no activation gradient tensor to send.
                if _is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group):
                    input_tensor_grad = None

                recv_next, next_backward_model_chunk_id, next_backward_microbatch_id = recv_tensor_from_previous_stage(
                    backward_k, forward=False
                )
                

                (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.

                if recv_next:
                    # SFT: use FIFO queue for cross-rank P2P gradient storage.
                    # The sender (next PP rank) sends gradients in group-aware order,
                    # so we just follow FIFO order here.
                    if is_sft_chunkpipe:
                        p2p_grad_queue[next_backward_model_chunk_id].append(
                            bwd_recv_buffer[backward_k % bwd_recv_buffer_size])
                    else:
                        output_tensor_grads[next_backward_model_chunk_id][next_backward_microbatch_id].append(
                            bwd_recv_buffer[backward_k % bwd_recv_buffer_size]
                        )
                    bwd_recv_buffer[(backward_k + 1) % bwd_recv_buffer_size] = None
                return input_tensor_grad

            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                pre_forward=pp_pre_forward,
                pre_backward=pp_pre_backward,
                post_forward=pp_post_forward,
                post_backward=pp_post_backward,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )

        else:  # No p2p overlap.
            backward_k = k
            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )
            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            if _is_vp_last_stage(vp_stage=forward_model_chunk_id) and is_pp_last_stage(pp_group):
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            if _is_vp_first_stage(vp_stage=backward_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            recv_prev, next_forward_model_chunk_id, next_forward_microbatch_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            recv_next, next_backward_model_chunk_id, next_backward_microbatch_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (input_tensor, output_tensor_grad) = (
                p2p_communicator.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                # SFT: The received P2P tensor is the previous PP rank's output from step forward_k-stagger
                # (due to P2P stagger of pipeline_parallel_size-1). Its fwd_mb equals
                # the sender's chunkpipe_forward_microbatch at step forward_k-stagger.
                if is_sft_chunkpipe:
                    _stagger = pipeline_parallel_size - 1
                    _sender_virt_mb = forward_k - _stagger if is_pp_first_stage(pp_group) else forward_k + 1
                    _prev_fwd_mb = (get_microbatch_id_in_model_chunk(_sender_virt_mb, forward=True) * num_chunks
                                    + get_sequence_id_in_microbatch(_sender_virt_mb, forward=True))
                    input_tensors[next_forward_model_chunk_id][_prev_fwd_mb] = input_tensor
                else:
                    input_tensors[next_forward_model_chunk_id][next_forward_microbatch_id].append(input_tensor)
            if recv_next:
                # SFT: use FIFO queue for cross-rank P2P gradient storage.
                if is_sft_chunkpipe:
                    p2p_grad_queue[next_backward_model_chunk_id].append(output_tensor_grad)
                else:
                    output_tensor_grads[next_backward_model_chunk_id][next_backward_microbatch_id].append(
                        output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
    nvtx_range_pop(suffix="steady")

    # Run cooldown backward passes (flush out pipeline) for the last model chunk.
    nvtx_range_push(suffix="cooldown")
    curr_vp_stage = config.virtual_pipeline_model_parallel_size - 1
    if not forward_only:
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if are_all_microbatches_in_warmup:
            # SFT: use FIFO queue for cross-rank P2P gradient storage.
            if is_sft_chunkpipe:
                p2p_grad_queue[num_model_chunks - 1].append(
                    p2p_communicator.recv_backward(
                        tensor_shape,
                        is_last_stage=(
                                _is_vp_last_stage(vp_stage=curr_vp_stage) and is_pp_last_stage(pp_group)
                        ),
                    ))
            else:
                output_tensor_grads[num_model_chunks - 1][-1].append(
                    p2p_communicator.recv_backward(
                        tensor_shape,
                        is_last_stage=(
                            _is_vp_last_stage(vp_stage=curr_vp_stage) and is_pp_last_stage(pp_group)
                        ),
                    )
                )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            if (
                not (_is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group))
                and k != 0
            ):
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, backward iteration {k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            recv_next, next_backward_model_chunk_id, next_backward_microbatch_id = recv_tensor_from_previous_stage(
                k, forward=False
            )

            if k == (total_num_microbatches - 1):
                recv_next = False

            # Prefetch recv for backward iteration k+1 for non last ranks.
            if config.overlap_p2p_comm_warmup_flush and not is_pp_last_stage(
                p2p_communicator.pp_group
            ):
                bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad=None,  # No input_tensor_grad to send.
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            _, input_tensor_grad = forward_backward_helper_wrapper(b_virtual_microbatch_id=k)

            # First virtual stage no activation gradient tensor to send.
            if _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            if config.overlap_p2p_comm_warmup_flush:
                if not is_pp_last_stage(p2p_communicator.pp_group):
                    _, bwd_wait_handles = p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=False,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                else:
                    bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_handles = (
                        p2p_communicator.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            overlap_p2p_comm=True,
                        )
                    )

                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))
                if recv_next:
                    # SFT: use FIFO queue for cross-rank P2P gradient storage.
                    if is_sft_chunkpipe:
                        p2p_grad_queue[next_backward_model_chunk_id].append(
                            bwd_recv_buffer[k % bwd_recv_buffer_size])
                    else:
                        output_tensor_grads[next_backward_model_chunk_id].append(
                            bwd_recv_buffer[k % bwd_recv_buffer_size]
                        )
                    bwd_recv_buffer[(k + 1) % bwd_recv_buffer_size] = None

            else:
                output_tensor_grad = p2p_communicator.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape
                )

                if recv_next:
                    # SFT: use FIFO queue for cross-rank P2P gradient storage.
                    if is_sft_chunkpipe:
                        p2p_grad_queue[next_backward_model_chunk_id].append(output_tensor_grad)
                    else:
                        output_tensor_grads[next_backward_model_chunk_id][next_backward_microbatch_id].append(
                            output_tensor_grad)

        if send_prev_wait_handle is not None:
            send_prev_wait_handle.wait()

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
    nvtx_range_pop(suffix="cooldown")

    nvtx_range_push(suffix="misc")
    assert (
        not recv_prev_wait_handles
    ), 'recv_prev_wait_handles should be cleared at the end of a step'
    assert (
        not recv_next_wait_handles
    ), 'recv_next_wait_handles should be cleared at the end of a step'

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, is_pp_last_stage(p2p_communicator.pp_group), tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).

        config.finalize_model_grads_func(
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
        )

    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and config.cuda_graph_scope != "full_iteration"
    ):
        create_cudagraphs()
    nvtx_range_pop(suffix="misc")

    return forward_data_store


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    # Convention used in this function:
    # num_microbatches for number of microbatches per pipeline stage;
    # num_model_chunks for virtual pipeline size;
    # then total_num_microbatches = num_microbatches * num_model_chunks.
    # Their corresponding index variables are
    # microbatch_id in [0, num_microbatches)
    # model_chunk_id in [0, num_model_chunks)
    # virtual_microbatch_id in [0, total_num_microbatches)

    config = get_model_config(model[0])

    if config.enable_chunkpipe:
        return forward_backward_pipelining_with_interleaving_with_chunkpipe(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                decoder_seq_length=decoder_seq_length,
                forward_only=forward_only)
    
    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model[0])
        assert model_type != ModelType.encoder_and_decoder, (
            "encoder PP stages not yet supported when passing custom process groups. "
            "support coming soon!"
        )
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have a tp_group"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have a cp_group"
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it is used default "
            "`parallel_state.default_embedding_ranks` to create the process group. If you are "
            "using the default process group, please use `parallel_state.get_embedding_group()` "
            "to get the process group. If you don't need explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have a pos_embd. In previous version, it is used default "
            "`parallel_state.default_position_embedding_ranks` to create the process group."
            " If you are using the default process group, please use "
            "`parallel_state.get_position_embedding_group()` "
            "If you don't need pos_embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pp'), "pg_collection must have a pp_group"
        assert hasattr(pg_collection, 'dp_cp'), "pg_collection must have a dp_cp_group"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection"
            " provide none or provide all the process groups"
        )

    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for interleaved pipeline parallelism"

    if not forward_only and config.fine_grained_activation_offloading:
        fine_grained_offloading_reset()

    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        # vp is ignored for clear_embedding_activation_buffer
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
    # They will be re-enabled at the end of this function.
    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    forward_data_store = []
    output_tensor_grads = None
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
    else:
        output_tensor_grads = None

    pipeline_parallel_size = p2p_communicator.pp_group.size()
    pipeline_parallel_rank = p2p_communicator.pp_group.rank()

    if (
        config.microbatch_group_size_per_vp_stage > num_microbatches
        or config.microbatch_group_size_per_vp_stage < pipeline_parallel_size
    ):
        msg = (
            'The number of contiguous micro-batches in a virtual pipeline stage'
            f'should range in [PP={pipeline_parallel_size} , M={num_microbatches}]'
        )
        raise ValueError(msg)

    # If the final micro-batch group has fewer micro-batches than pipeline-parallel size,
    # the pipeline will have dependency bubbles.
    final_microbatch_group_size = num_microbatches % config.microbatch_group_size_per_vp_stage
    if 0 < final_microbatch_group_size < pipeline_parallel_size:
        msg = 'The remainder of M (the total micro-batches) divided by N (number of '
        msg += 'contiguous micro-batches in a virtual pipeline stage) should be 0, '
        msg += 'or larger than or equal to the pipeline-parallel size, but it is '
        msg += f'{final_microbatch_group_size}. '
        msg += 'Otherwise, it introduces dependency bubbles in the pipeline '
        msg += 'and reduces throughput.'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])

    # Determine hidden dimension for P2P communication
    # For hyper connections with multiple PP stages, use n-stream dimension
    hidden_dim = config.hidden_size
    if config.enable_hyper_connections and pipeline_parallel_size > 1:
        # For interleaved PP with hyper connections, all intermediate communications use n-stream
        # Note: This is a simplified approach - proper VPP support may need more complex logic
        hidden_dim = config.hidden_size * config.num_residual_streams

    tensor_shape = [seq_length, micro_batch_size, hidden_dim]
    tensor_shape[0] = tensor_shape[0] // cp_group.size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // tp_group.size()

    # Compute number of warmup and remaining microbatches.
    # seems only used for vpp
    num_model_chunks = len(model)
    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches,
        num_model_chunks,
        config.microbatch_group_size_per_vp_stage,
        forward_only=forward_only,
        overlap_moe_expert_parallel_comm=config.overlap_moe_expert_parallel_comm,
        p2p_communicator=p2p_communicator,
    )

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    # Create a tunable schedule lookup table.
    # The schedule lookup table uses the virtual_microbatch_id to find the corresponding
    # microbatch_id and model_chunk_id. For example, the tunable schedule table for
    # PP2 N3M5 with VP2 is constructed as below:
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    schedule_table = get_schedule_table(
        num_microbatches, len(model), config.microbatch_group_size_per_vp_stage
    )

    # Decouple individual lookup table for microbatch_id and model_chunk_id.
    # For example, the micro-batch table for PP2 N3M5 with VP2 is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # Similarly, the model chunk table is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    # Both tables are indexed with virtual_microbatch_id.
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    def get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
        """Helper method to count number of released (i.e. popped from input_tensors)
        microbatches for a model chunk."""
        if forward_only:  # Micro-batch is released after forward prop.
            return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
        else:  # Micro-batch is released after backward prop.
            # Zero backward prop in warmup.
            if virtual_microbatch_id < num_warmup_microbatches:
                return 0
            else:
                backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
                model_chunk_id = num_model_chunks - model_chunk_id - 1
                return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)

    def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
        else:
            return False

    def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
        """Determine if peers are sending, and where in data structure
        to put received tensors.
        Return a boolean if the pipeline stage expects to recv from peers, and the
        corresponding model_chunk_id for the received tensor.
        """
        recv = True
        # The leading pipeline stage is the first rank in fwd and the last rank in bwd.
        is_leading_pipeline_stage = (
            is_pp_first_stage(p2p_communicator.pp_group)
            if forward
            else is_pp_last_stage(p2p_communicator.pp_group)
        )

        last_model_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading_pipeline_stage:
            # The leading pipeline stage is ahead of the ending pipeline stage
            # (i.e. last rank in fwd and first rank in bwd) by (pipeline_parallel_size - 1).
            # Let's consider bwd as an example with PP 4:
            #       0 1 2 3 ...
            #     0 1 2 3 ...
            #   0 1 2 3 ...
            # 0 1 2 3 ...
            if virtual_microbatch_id < (pipeline_parallel_size - 1):
                # The ending stage has not produced any tensors, so no recv will be initiated.
                recv = False
                next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
            else:
                # Find the model chunk of the aligned microbatches in the ending stage.
                # For example, microbatch 0 in the ending stage is aligned with microbatch 3
                # in the leading stage.
                next_model_chunk_id = get_model_chunk_id(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )
            # Last model chunk in the final stage does not produce tensors.
            if next_model_chunk_id == last_model_chunk:
                recv = False
            if forward:
                # Model chunk id increases in forward.
                next_model_chunk_id += 1
            else:
                # Model chunk id decreases in backward.
                next_model_chunk_id -= 1
        else:
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)

        return recv, next_model_chunk_id

    def forward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id, microbatch_id):
        """Preprocess for forward_step_helper"""
        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
            if (
                param_sync_virtual_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
            ):
                param_sync_chunk_id = (
                    get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
                )
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if _is_vp_first_stage(vp_stage=model_chunk_id) and is_pp_first_stage(pp_group):
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)

        # For non-depth-first pipeline schedules, the first rank would buffer multiple received
        # activation tensors for a model chunk until accessed during warmup.
        # This input buffering is needed to overlap the computation with the receipt of
        # the next inputs. To index the proper buffered inputs for forword_step, we use
        # microbatch_id offset with number of released microbatches that have completed backprop.
        offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
        input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

        # add for config.reduce_variable_seq_shape_p2p_comm, remove padding.
        if config.reduce_variable_seq_shape_p2p_comm:
            input_tensor = p2p_comm_tensor_remove_padding(input_tensor,
                                                          config.p2p_comm_fixed_seq_lengths_per_rank)
            input_tensors[model_chunk_id][microbatch_id - offset] = input_tensor

        return input_tensor

    def forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens):
        """Postprocess for forward_step_helper"""
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens

        # If forward-only, no need to save tensors for a backward pass.
        if forward_only:
            # Release the tensor that have completed forward step.
            input_tensors[model_chunk_id].pop(0)
            output_tensors[model_chunk_id].pop()

        return

    def forward_step_helper(virtual_microbatch_id, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks"""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
        microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id, forward=True)

        input_tensor = forward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id, microbatch_id
        )

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
            vp_stage=model_chunk_id,
            is_last_stage=_is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group),
        )

        forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens)

        return output_tensor

    def backward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id):
        """Preprocess for backward_step_helper"""
        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
            virtual_microbatch_id
        ):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        # pylint: disable=E0606
        if _is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group):
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        # add for reduce_variable_seq_shape_p2p_comm, remove padding.
        if config.reduce_variable_seq_shape_p2p_comm:
            output_tensor_grad = p2p_comm_tensor_remove_padding(output_tensor_grad,
                                                                config.p2p_comm_fixed_seq_lengths_per_rank)

        return input_tensor, output_tensor, output_tensor_grad

    def backward_step_helper_postprocess(virtual_microbatch_id):
        """Postprocess for backward_step_helper"""
        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
            if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_virtual_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(
                    grad_sync_virtual_microbatch_id, forward=False
                )
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

    def backward_step_helper(virtual_microbatch_id):
        """Helper method to run backward step with model split into chunks"""
        nonlocal output_tensor_grads
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)

        input_tensor, output_tensor, output_tensor_grad = backward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id
        )

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        backward_step_helper_postprocess(virtual_microbatch_id)

        return input_tensor_grad

    def forward_backward_helper_wrapper(
        f_virtual_microbatch_id=None,
        b_virtual_microbatch_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        checkpoint_activations_microbatch=None,
    ):
        """
        wrap forward_helper, backward_helper, and combined_forward_backward_helper in a unified way
        """
        if config.overlap_moe_expert_parallel_comm and not forward_only:  # Combined 1F1B path
            return combined_1f1b_schedule_for_interleaved_pipelining(
                config,
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                forward_data_store,
                forward_step_helper_preprocess,
                forward_step_helper_postprocess,
                backward_step_helper_preprocess,
                backward_step_helper_postprocess,
                get_microbatch_id_in_model_chunk,
                get_model_chunk_id,
                partial(check_first_val_step, first_val_step, forward_only),
                is_first_microbatch_for_model_chunk,
                collect_non_loss_data,
                f_virtual_microbatch_id=f_virtual_microbatch_id,
                b_virtual_microbatch_id=b_virtual_microbatch_id,
                pre_forward=pre_forward,
                pre_backward=pre_backward,
                post_forward=post_forward,
                post_backward=post_backward,
            )
        else:  # Conventional interleaved 1F1B path
            forward_output_tensor = None
            backward_input_tensor_grad = None
            # forward pass
            if f_virtual_microbatch_id is not None:
                forward_model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
                if pre_forward is not None:
                    pre_forward()
                forward_output_tensor = forward_step_helper(
                    f_virtual_microbatch_id, checkpoint_activations_microbatch
                )
                if post_forward is not None:
                    forward_output_tensor = post_forward(forward_output_tensor)

            # Backward pass.
            if b_virtual_microbatch_id is not None:
                backward_model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
                if pre_backward is not None:
                    pre_backward()
                backward_input_tensor_grad = backward_step_helper(b_virtual_microbatch_id)
                if post_backward is not None:
                    backward_input_tensor_grad = post_backward(backward_input_tensor_grad)
            return forward_output_tensor, backward_input_tensor_grad

    # ==============================main logic=========================================
    _is_vp_first_stage = partial(
        is_vp_first_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    _is_vp_last_stage = partial(
        is_vp_last_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    pp_group = p2p_communicator.pp_group

    # Run warmup forward passes.
    nvtx_range_push(suffix="warmup")
    input_tensors[0].append(
        p2p_communicator.recv_forward(
            tensor_shape, _is_vp_first_stage(vp_stage=0) and is_pp_first_stage(pp_group)
        )
    )

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if is_pp_first_stage(p2p_communicator.pp_group):
        fwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if is_pp_last_stage(p2p_communicator.pp_group):
        bwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        bwd_recv_buffer_size = 1
    fwd_recv_buffer = [None] * fwd_recv_buffer_size
    bwd_recv_buffer = [None] * bwd_recv_buffer_size
    recv_prev_wait_handles = []
    send_next_wait_handle = None
    send_prev_wait_handle = None
    recv_next_wait_handles = []

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)

        if config.overlap_p2p_comm_warmup_flush:
            if (
                not (
                    _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group)
                )
                and k != 0
            ):
                assert recv_prev_wait_handles, (
                    f'pp rank {pipeline_parallel_rank}, iteration {k},'
                    'should have registered recv handle'
                )
                recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                recv_prev_wait_handle.wait()

        # Determine if tensor should be received from previous stage.
        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

        # No receive in last iteration when recv iteration k+1.
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Prefetch recv for iteration k+1 for non-first ranks.
        if config.overlap_p2p_comm_warmup_flush and not is_pp_first_stage(
            p2p_communicator.pp_group
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communicator.send_forward_recv_forward(
                    output_tensor=None,  # No output_tensor to send.
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )
            )

            if fwd_wait_recv_handles:
                recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        output_tensor, _ = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=k,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # Don't send tensor downstream if on last stage.
        if _is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group):
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm_warmup_flush:
            if (
                k == (num_warmup_microbatches - 1)
                and not config.overlap_p2p_comm
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communicator.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                    )
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape
                )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not is_pp_first_stage(p2p_communicator.pp_group):
                # Send only since recv prefetched.
                _, fwd_wait_handles = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=False, tensor_shape=tensor_shape, overlap_p2p_comm=True
                )
            else:  # No prefetch for first rank, so both send and recv initiated.
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(k + 1) % fwd_recv_buffer_size] = None

        if config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                if recv_next:
                    output_tensor_grads[num_model_chunks - 1].append(bwd_recv_buffer[-1])
    nvtx_range_pop(suffix="warmup")

    # Run 1F1B in steady state.
    nvtx_range_push(suffix="steady")
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        if config.overlap_p2p_comm:

            backward_k = k

            # Sync forward recv
            def pp_pre_forward(vp_stage=None):
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                if not (_is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_prev_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, fwd iteration {forward_k}, '
                            'should have registered recv handle'
                        )
                        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                        recv_prev_wait_handle.wait()
                    else:
                        if recv_prev_wait_handles is not None and recv_prev_wait_handles:
                            recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                            recv_prev_wait_handle.wait()

                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Async forward send / receive
            def pp_post_forward(output_tensor, vp_stage=None):
                nonlocal send_next_wait_handle
                nonlocal fwd_recv_buffer
                nonlocal fwd_wait_handles
                nonlocal recv_prev_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                # Last virtual stage no activation tensor to send.
                if _is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group):
                    output_tensor = None

                recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                    forward_k, forward=True
                )

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_next_wait_handle is not None:
                    send_next_wait_handle.wait()
                if fwd_wait_handles is not None:
                    send_next_wait_handle = (
                        fwd_wait_handles.pop("send_next")
                        if "send_next" in fwd_wait_handles
                        else None
                    )
                    if "recv_prev" in fwd_wait_handles:
                        recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
                # assert fwd_wait_handles is not None

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.
                if recv_prev:
                    input_tensors[next_forward_model_chunk_id].append(
                        fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                    )
                    fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None

                return output_tensor

            # Sync backward recv
            def pp_pre_backward(vp_stage=None):
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                if not (_is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_next_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, bwd iteration {backward_k}, '
                            'should have registered recv next handle'
                        )
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()
                    else:
                        if recv_next_wait_handles is not None and recv_next_wait_handles:
                            recv_next_wait_handle = recv_next_wait_handles.pop(0)
                            recv_next_wait_handle.wait()

            # Async backward send / receive
            def pp_post_backward(input_tensor_grad, vp_stage=None):
                nonlocal send_prev_wait_handle
                nonlocal bwd_wait_handles
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                # First virtual stage no activation gradient tensor to send.
                if _is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group):
                    input_tensor_grad = None

                recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                    backward_k, forward=False
                )

                (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[backward_k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(backward_k + 1) % bwd_recv_buffer_size] = None
                return input_tensor_grad

            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                pre_forward=pp_pre_forward,
                pre_backward=pp_pre_backward,
                post_forward=pp_post_forward,
                post_backward=pp_post_backward,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )

        else:  # No p2p overlap.
            backward_k = k
            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )
            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            if _is_vp_last_stage(vp_stage=forward_model_chunk_id) and is_pp_last_stage(pp_group):
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            if _is_vp_first_stage(vp_stage=backward_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (input_tensor, output_tensor_grad) = (
                p2p_communicator.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
    nvtx_range_pop(suffix="steady")

    # Run cooldown backward passes (flush out pipeline) for the last model chunk.
    nvtx_range_push(suffix="cooldown")
    curr_vp_stage = config.virtual_pipeline_model_parallel_size - 1
    if not forward_only:
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if are_all_microbatches_in_warmup:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communicator.recv_backward(
                    tensor_shape,
                    is_last_stage=(
                        _is_vp_last_stage(vp_stage=curr_vp_stage) and is_pp_last_stage(pp_group)
                    ),
                )
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            if (
                not (_is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group))
                and k != 0
            ):
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, backward iteration {k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                k, forward=False
            )

            if k == (total_num_microbatches - 1):
                recv_next = False

            # Prefetch recv for backward iteration k+1 for non last ranks.
            if config.overlap_p2p_comm_warmup_flush and not is_pp_last_stage(
                p2p_communicator.pp_group
            ):
                bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad=None,  # No input_tensor_grad to send.
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            _, input_tensor_grad = forward_backward_helper_wrapper(b_virtual_microbatch_id=k)

            # First virtual stage no activation gradient tensor to send.
            if _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            if config.overlap_p2p_comm_warmup_flush:
                if not is_pp_last_stage(p2p_communicator.pp_group):
                    _, bwd_wait_handles = p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=False,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                else:
                    bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_handles = (
                        p2p_communicator.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            overlap_p2p_comm=True,
                        )
                    )

                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(k + 1) % bwd_recv_buffer_size] = None

            else:
                output_tensor_grad = p2p_communicator.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape
                )

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

        if send_prev_wait_handle is not None:
            send_prev_wait_handle.wait()

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
    nvtx_range_pop(suffix="cooldown")

    nvtx_range_push(suffix="misc")
    assert (
        not recv_prev_wait_handles
    ), 'recv_prev_wait_handles should be cleared at the end of a step'
    assert (
        not recv_next_wait_handles
    ), 'recv_next_wait_handles should be cleared at the end of a step'

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, is_pp_last_stage(p2p_communicator.pp_group), tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).

        config.finalize_model_grads_func(
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
        )

    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and config.cuda_graph_scope != "full_iteration"
    ):
        create_cudagraphs()
    nvtx_range_pop(suffix="misc")

    return forward_data_store


def get_tensor_shapes(
    *,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    tp_group: torch.distributed.ProcessGroup,
    cp_group: torch.distributed.ProcessGroup,
    pp_rank: int = None,
    pp_size: int = None,
    is_recv: bool = True,
):
    """
    Determine right tensor sizes (based on position of rank with respect to split rank) and
    model size.

    For hyper connections (mHC), intermediate pipeline stages communicate n-stream tensors
    with dimension hidden_size * num_residual_streams.
    
    Args:
        is_recv: If True, compute shape for receiving; if False, for sending.
                 This matters for hyper connections where first/last stages have different
                 send/recv dimensions.
    """

    tensor_shapes = []
    # Use decoder_seq_length if provided, otherwise use seq_length
    effective_seq_length = decoder_seq_length if decoder_seq_length is not None else seq_length
    effective_seq_length = effective_seq_length // cp_group.size()

    if config.sequence_parallel:
        effective_seq_length = effective_seq_length // tp_group.size()

    # Determine hidden dimension based on hyper connections and pipeline stage
    hidden_dim = config.hidden_size
    if config.enable_hyper_connections and pp_rank is not None and pp_size is not None:
        # For hyper connections:
        # - recv: stages with rank > 0 receive n-stream (n*C) from previous stage
        # - send: stages with rank < pp_size-1 send n-stream (n*C) to next stage
        use_nstream = False
        if is_recv and pp_rank > 0:
            # Receiving from previous stage (which sends n*C)
            use_nstream = True
        elif not is_recv and pp_rank < pp_size - 1:
            # Sending to next stage (send n*C)
            use_nstream = True

        if use_nstream:
            hidden_dim = config.hidden_size * config.num_residual_streams

    tensor_shapes.append((effective_seq_length, micro_batch_size, hidden_dim))
    return tensor_shapes



def forward_backward_pipelining_without_interleaving_chunkpipe(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model)
        assert model_type != ModelType.encoder_and_decoder, (
            "encoder PP stages not yet supported when passing custom process groups. "
            "support coming soon!"
        )
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp_group"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp_group"
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it is used default "
            "`parallel_state.default_embedding_ranks` to create the process group. "
            " If you are using the default process group, please use "
            " `parallel_state.get_embedding_group()` "
            "If you don't need embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have a pos_embd. In previous version, it is used default "
            "`parallel_state.default_position_embedding_ranks` to create the process group. "
            " If you are using the default process group, please use  "
            " `parallel_state.get_position_embedding_group()` "
            "If you don't need pos_embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pp'), "pg_collection must have pp_group"
        assert hasattr(pg_collection, 'dp_cp'), "pg_collection must have dp_cp_group"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection "
            "provide none or provide all the process groups"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if not forward_only and config.fine_grained_activation_offloading:
        fine_grained_offloading_reset()

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # SFT dynamic group_size support for PP>1 chunkpipe
    is_sft_chunkpipe = getattr(config, 'sft_chunkpipe_mode', False)

    def _get_group_id_from_cache(mb_id, group_size_cache, chunk_num_per_seq):
        """Compute group_id from known group_size_cache.

        group_size_cache: dict, group_id -> group_size, filled by get_batch().
        When cache is empty, use chunk_num_per_seq as fallback (conservative).
        """
        if not group_size_cache:
            return mb_id // chunk_num_per_seq
        acc = 0
        for gid, gs in sorted(group_size_cache.items()):
            if mb_id < acc + gs:
                return gid
            acc += gs
        # fallback: group not yet discovered, offset by known groups
        return len(group_size_cache) + (mb_id - acc) // chunk_num_per_seq

    def _get_chunk_pos_from_cache(mb_id, group_id, group_size_cache, chunk_num_per_seq):
        """Get position of current chunk within its group (0-based)."""
        if not group_size_cache:
            return mb_id % chunk_num_per_seq
        acc = 0
        for gid, gs in sorted(group_size_cache.items()):
            if gid == group_id:
                return mb_id - acc
            acc += gs
        # fallback: group not yet discovered, offset by known groups
        return (mb_id - acc) % chunk_num_per_seq

    def _is_last_chunk_of_group_dynamic(mb_id, group_size_cache, chunk_num_per_seq):
        """Check if current chunk is the last in its group.

        If group_size is known, exact check; otherwise use modulo fallback
        (may delay cleanup but doesn't affect correctness).
        """
        acc = 0
        for gid, gs in sorted(group_size_cache.items()):
            if mb_id < acc + gs:
                return mb_id == acc + gs - 1
            acc += gs
        # fallback: group not yet discovered, offset by known groups
        return (mb_id - acc + 1) % chunk_num_per_seq == 0

    def _ensure_sft_chunk_info_after_forward(mb_id):
        """Record exact per-real-group chunk state after get_batch populated config."""
        nonlocal sft_current_composite_id
        components = list(getattr(config, 'chunkpipe_component_sizes', None) or [])
        if not components:
            components = [getattr(config, 'chunkpipe_current_group_size', None) or config.chunk_num_per_seq]

        if not sft_composite_cursor:
            sft_current_composite_id += 1
            for real_size in components:
                for chunk_idx in range(real_size):
                    sft_composite_cursor.append((chunk_idx, real_size))

        if sft_composite_cursor:
            sft_chunk_info[mb_id] = sft_composite_cursor.popleft()
            sft_mb_to_composite_id[mb_id] = sft_current_composite_id
        else:
            sft_chunk_info[mb_id] = (
                getattr(config, 'chunkpipe_chunk_idx_in_group', 0),
                getattr(config, 'chunkpipe_current_group_size', config.chunk_num_per_seq),
            )
            sft_mb_to_composite_id[mb_id] = sft_current_composite_id

    def _restore_sft_chunk_info_for_backward(mb_id):
        """Restore exact forward-time real-group state for backward."""
        if mb_id in sft_chunk_info:
            chunk_idx, real_size = sft_chunk_info[mb_id]
            config.chunkpipe_chunk_idx_in_group = chunk_idx
            config.chunkpipe_current_group_size = real_size
            return

        _gid = _get_group_id_from_cache(mb_id, group_size_cache, config.chunk_num_per_seq)
        config.chunkpipe_chunk_idx_in_group = _get_chunk_pos_from_cache(
            mb_id, _gid, group_size_cache, config.chunk_num_per_seq
        )
        config.chunkpipe_current_group_size = group_size_cache.get(_gid, config.chunk_num_per_seq)

    # group_size_cache: progressively records discovered group info
    # key = group_id (sequential), value = group_size
    group_size_cache = {}
    # Exact SFT composite replay state: mb_id -> (chunk_idx_in_group, real_group_size).
    sft_chunk_info = {}
    # Exact SFT composite ownership: mb_id -> composite_id.
    sft_mb_to_composite_id = {}
    sft_current_composite_id = -1
    # Pending expanded component states for the current composite.
    sft_composite_cursor = deque()

    if is_sft_chunkpipe:
        pass  # SFT: num_sequences determined dynamically, skip assertion
    else:
        assert num_microbatches % config.chunk_num_per_seq == 0, "num microbatches should be divided by num chunks"
        num_sequences = num_microbatches // config.chunk_num_per_seq

    # Compute number of warmup microbatches.
    # Same per-rank stagger as the standard 1F1B schedule (adjacent stages differ
    # by one), so the steady-state send/recv pairs of neighbouring stages match
    # within the same iteration. The extra chunk_num_per_seq - 1 forwards let all
    # chunks of the first sequence finish forward before its last chunk runs
    # backward; it is the same on every rank and does not change the pairing.
    num_warmup_microbatches = (
        p2p_communicator.pp_group.size() - p2p_communicator.pp_group.rank() - 1
    )
    num_warmup_microbatches += config.chunk_num_per_seq - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)

    rank = p2p_communicator.pp_group.rank()
    pp_size = p2p_communicator.pp_group.size()
    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_rank=rank,
        pp_size=pp_size,
        is_recv=True,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_rank=rank,
        pp_size=pp_size,
        is_recv=False,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # Input, output tensors only need to be saved when doing backward passes
    input_output_chunk_micro = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    chunkpipe_forward_microbatch = 0

    if not forward_only:
        input_output_chunk_micro = {}
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
        )
        config.chunkpipe_forward_microbatch = chunkpipe_forward_microbatch
        config.chunkpipe_forward = True
        if is_sft_chunkpipe:
            _gid = _get_group_id_from_cache(chunkpipe_forward_microbatch, group_size_cache, config.chunk_num_per_seq)
            config.chunkpipe_chunk_idx_in_group = _get_chunk_pos_from_cache(
                chunkpipe_forward_microbatch, _gid, group_size_cache, config.chunk_num_per_seq
            )
            if config.chunkpipe_chunk_idx_in_group > 0 and _gid in group_size_cache:
                config.chunkpipe_current_group_size = group_size_cache[_gid]
        else:
            config.chunkpipe_chunk_idx_in_group = chunkpipe_forward_microbatch % config.chunk_num_per_seq
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=chunkpipe_forward_microbatch,
            is_last_stage=is_pp_last_stage(p2p_communicator.pp_group),
        )
        p2p_communicator.send_forward(output_tensor, is_pp_last_stage(p2p_communicator.pp_group))
        total_num_tokens += num_tokens

        if is_sft_chunkpipe:
            _ensure_sft_chunk_info_after_forward(chunkpipe_forward_microbatch)
            _discovered_gs = getattr(config, 'chunkpipe_current_group_size', None)
            if _discovered_gs is not None and _discovered_gs > 0:
                _gid = _get_group_id_from_cache(chunkpipe_forward_microbatch, group_size_cache,
                                                config.chunk_num_per_seq)
                if _gid not in group_size_cache:
                    group_size_cache[_gid] = _discovered_gs

        if not forward_only:
            # let all chunks belong to the same sequence
            # compose a element of the input_output_chunk_micro
            if is_sft_chunkpipe:
                sequence_num = sft_mb_to_composite_id.get(chunkpipe_forward_microbatch)
                if sequence_num is None:
                    sequence_num = _get_group_id_from_cache(chunkpipe_forward_microbatch, group_size_cache,
                                                            config.chunk_num_per_seq)
            else:
                sequence_num = chunkpipe_forward_microbatch // config.chunk_num_per_seq
            if sequence_num not in input_output_chunk_micro:
                input_output_chunk_micro[sequence_num] = []
            chunks_infos = input_output_chunk_micro[sequence_num]
            tmp_tuple = (input_tensor, output_tensor, chunkpipe_forward_microbatch)
            chunks_infos.append(tmp_tuple)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
        else:
            if is_sft_chunkpipe:
                _is_last = _is_last_chunk_of_group_dynamic(chunkpipe_forward_microbatch, group_size_cache,
                                                           config.chunk_num_per_seq)
            else:
                _is_last = (chunkpipe_forward_microbatch + 1) % config.chunk_num_per_seq == 0
            if _is_last:
                # clear cache for all chunks belongs to the same sequence
                clear_key_value_cache(model, config.mtp_num_layers)
        chunkpipe_forward_microbatch += 1

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
        )

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        config.chunkpipe_forward_microbatch = chunkpipe_forward_microbatch
        config.chunkpipe_forward = True
        if is_sft_chunkpipe:
            _gid = _get_group_id_from_cache(chunkpipe_forward_microbatch, group_size_cache, config.chunk_num_per_seq)
            config.chunkpipe_chunk_idx_in_group = _get_chunk_pos_from_cache(
                chunkpipe_forward_microbatch, _gid, group_size_cache, config.chunk_num_per_seq
            )
            if config.chunkpipe_chunk_idx_in_group > 0 and _gid in group_size_cache:
                config.chunkpipe_current_group_size = group_size_cache[_gid]
        else:
            config.chunkpipe_chunk_idx_in_group = chunkpipe_forward_microbatch % config.chunk_num_per_seq

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=chunkpipe_forward_microbatch,
            is_last_stage=is_pp_last_stage(p2p_communicator.pp_group),
        )
        total_num_tokens += num_tokens

        if is_sft_chunkpipe:
            _ensure_sft_chunk_info_after_forward(chunkpipe_forward_microbatch)
            _discovered_gs = getattr(config, 'chunkpipe_current_group_size', None)
            if _discovered_gs is not None and _discovered_gs > 0:
                _gid = _get_group_id_from_cache(chunkpipe_forward_microbatch, group_size_cache,
                                                config.chunk_num_per_seq)
                if _gid not in group_size_cache:
                    group_size_cache[_gid] = _discovered_gs

        if forward_only:
            p2p_communicator.send_forward(
                output_tensor, is_pp_last_stage(p2p_communicator.pp_group)
            )
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
                )
            if is_sft_chunkpipe:
                _is_last = _is_last_chunk_of_group_dynamic(chunkpipe_forward_microbatch, group_size_cache,
                                                           config.chunk_num_per_seq)
            else:
                _is_last = (chunkpipe_forward_microbatch + 1) % config.chunk_num_per_seq == 0
            if _is_last:
                # clear cache for key&value for the same sequence
                clear_key_value_cache(model, config.mtp_num_layers)
        else:
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
            )

            # Add input_tensor and output_tensor to end of list.
            # let all chunks belong to the same sequence
            # compose a element of the input_output_chunk_micro
            if is_sft_chunkpipe:
                sequence_num = sft_mb_to_composite_id.get(chunkpipe_forward_microbatch)
                if sequence_num is None:
                    sequence_num = _get_group_id_from_cache(chunkpipe_forward_microbatch, group_size_cache,
                                                            config.chunk_num_per_seq)
            else:
                sequence_num = chunkpipe_forward_microbatch // config.chunk_num_per_seq
            if sequence_num not in input_output_chunk_micro:
                input_output_chunk_micro[sequence_num] = []
            chunks_infos = input_output_chunk_micro[sequence_num]
            tmp_tuple = (input_tensor, output_tensor, chunkpipe_forward_microbatch)
            chunks_infos.append(tmp_tuple)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # for all chunks belong to the same sequence,
            # backward is the reverse direction of the forward
            min_key = get_min_key(input_output_chunk_micro)
            chunks_infos = input_output_chunk_micro[min_key]
            if len(chunks_infos) == 0:
                del input_output_chunk_micro[min_key]
                min_key = get_min_key(input_output_chunk_micro)
                chunks_infos = input_output_chunk_micro[min_key]
            tmp_tuple = chunks_infos.pop()

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            config.chunkpipe_forward = False
            config.chunkpipe_backward_microbatch = tmp_tuple[2]

            if is_sft_chunkpipe:
                _restore_sft_chunk_info_for_backward(tmp_tuple[2])
            else:
                config.chunkpipe_chunk_idx_in_group = config.chunkpipe_backward_microbatch % config.chunk_num_per_seq

            input_tensor_grad = backward_step(
                tmp_tuple[0], tmp_tuple[1], output_tensor_grad, model_type, config
            )

            # remove caches for key & values
            remove_key_value_cache(model, tmp_tuple[2], config.mtp_num_layers)

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, is_pp_first_stage(p2p_communicator.pp_group)
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad,
                    recv_tensor_shapes,
                    is_pp_first_stage(p2p_communicator.pp_group),
                )
        chunkpipe_forward_microbatch += 1

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            # for all chunks belong to the same sequence,
            # backward is the reverse direction of the forward
            min_key = get_min_key(input_output_chunk_micro)
            chunks_infos = input_output_chunk_micro[min_key]
            if len(chunks_infos) == 0:
                del input_output_chunk_micro[min_key]
                min_key = get_min_key(input_output_chunk_micro)
                chunks_infos = input_output_chunk_micro[min_key]
            tmp_tuple = chunks_infos.pop()

            output_tensor_grad = p2p_communicator.recv_backward(
                send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
            )
            config.chunkpipe_forward = False
            config.chunkpipe_backward_microbatch = tmp_tuple[2]

            if is_sft_chunkpipe:
                _restore_sft_chunk_info_for_backward(tmp_tuple[2])
            else:
                config.chunkpipe_chunk_idx_in_group = config.chunkpipe_backward_microbatch % config.chunk_num_per_seq

            input_tensor_grad = backward_step(
                tmp_tuple[0], tmp_tuple[1], output_tensor_grad, model_type, config
            )

            # remove caches for key & values
            remove_key_value_cache(model, tmp_tuple[2], config.mtp_num_layers)

            p2p_communicator.send_backward(
                input_tensor_grad, is_pp_first_stage(p2p_communicator.pp_group)
            )

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, is_pp_last_stage(p2p_communicator.pp_group), tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        if config.calculate_per_token_loss:
            config.finalize_model_grads_func(
                [model],
                total_num_tokens,
                pg_collection=pg_collection,
            )
        else:
            config.finalize_model_grads_func(
                [model], None, pg_collection=pg_collection,
            )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and config.cuda_graph_scope != "full_iteration"
    ):
        create_cudagraphs()

    return forward_data_store


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    if config.enable_chunkpipe:
        return forward_backward_pipelining_without_interleaving_chunkpipe(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=num_microbatches,
                    seq_length=seq_length,
                    micro_batch_size=micro_batch_size,
                    decoder_seq_length=decoder_seq_length,
                    forward_only=forward_only,
                    collect_non_loss_data=collect_non_loss_data,
                    first_val_step=first_val_step,
                    adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
                    p2p_communicator=p2p_communicator,
                    pg_collection=pg_collection)

    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model)
        assert model_type != ModelType.encoder_and_decoder, (
            "encoder PP stages not yet supported when passing custom process groups. "
            "support coming soon!"
        )
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp_group"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp_group"
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it is used default "
            "`parallel_state.default_embedding_ranks` to create the process group. "
            " If you are using the default process group, please use "
            " `parallel_state.get_embedding_group()` "
            "If you don't need embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have a pos_embd. In previous version, it is used default "
            "`parallel_state.default_position_embedding_ranks` to create the process group. "
            " If you are using the default process group, please use  "
            " `parallel_state.get_position_embedding_group()` "
            "If you don't need pos_embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pp'), "pg_collection must have pp_group"
        assert hasattr(pg_collection, 'dp_cp'), "pg_collection must have dp_cp_group"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection "
            "provide none or provide all the process groups"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if not forward_only and config.fine_grained_activation_offloading:
        fine_grained_offloading_reset()

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        p2p_communicator.pp_group.size() - p2p_communicator.pp_group.rank() - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)

    rank = p2p_communicator.pp_group.rank()
    pp_size = p2p_communicator.pp_group.size()
    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_rank=rank,
        pp_size=pp_size,
        is_recv=True,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_rank=rank,
        pp_size=pp_size,
        is_recv=False,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
        )
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            is_last_stage=is_pp_last_stage(p2p_communicator.pp_group),
        )
        p2p_communicator.send_forward(output_tensor, is_pp_last_stage(p2p_communicator.pp_group))
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
        )

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            is_last_stage=is_pp_last_stage(p2p_communicator.pp_group),
        )
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(
                output_tensor, is_pp_last_stage(p2p_communicator.pp_group)
            )
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
                )
        else:
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, is_pp_first_stage(p2p_communicator.pp_group)
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad,
                    recv_tensor_shapes,
                    is_pp_first_stage(p2p_communicator.pp_group),
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communicator.recv_backward(
                send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
            )

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            p2p_communicator.send_backward(
                input_tensor_grad, is_pp_first_stage(p2p_communicator.pp_group)
            )

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, is_pp_last_stage(p2p_communicator.pp_group), tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and config.cuda_graph_scope != "full_iteration"
    ):
        create_cudagraphs()

    return forward_data_store
