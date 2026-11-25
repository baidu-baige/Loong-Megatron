# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_moe_residual_add import (
    can_use_fused_moe_residual_add,
    fused_moe_residual_add,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.utils import ScheduleNode, StageDispatchBwdGrad
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe.moe_layer import FlushPendingGradAccum, MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor
from megatron.core.typed_torch import apply_module, copy_signature


def _snapshot_dispatcher_state(dispatcher) -> dict | None:
    """Capture mutable routing metadata around a checkpoint recompute."""
    if dispatcher is None:
        return None
    return dict(dispatcher.__dict__)


def _restore_dispatcher_state(dispatcher, state: dict | None) -> None:
    if dispatcher is None or state is None:
        return
    dispatcher.__dict__.clear()
    dispatcher.__dict__.update(state)


def build_transformer_layer_callables(layer: TransformerLayer):
    """Create callables for transformer layer nodes.
    Divides the transformer layer's operations into a sequence of smaller, independent
    functions. This decomposition separates computation-heavy tasks (e.g., self-attention,
    MLP) from communication-heavy tasks (e.g., MoE's All-to-All).

    The six callables align with the schedule plan's slot order:
    1. pre_dispatch_computation (computation):
       attention -> pre-MLP layernorm -> router -> dispatch preprocess.
       For dense layers this is just the attention pass.
    2. moe_dispatch (communication): MoE dispatch All-to-All.
    3. mlp / moe_experts (computation): dense MLP or routed-experts compute.
    4. moe_combine (communication): MoE combine All-to-All.
    5. post_combine (computation): MoE combine postprocess and MLP residual.
    6. mtp_post_process (computation): always ``None`` here; only the MTP
       wrapper in ``common/fine_grained_callables.py`` fills this slot.

    By assigning these functions to different CUDA streams (e.g., a compute stream
    and a communication stream), the scheduler can overlap their execution, preventing
    tasks from competing for resources and hiding communication latency by running them
    in parallel with functions from other micro-batches.

    Args:
        layer: The transformer layer to build callables for.

    Returns:
        A tuple containing:
        - forward_funcs: List of 6 callables, one per slot in the schedule plan
          (pre_dispatch_computation, moe_dispatch, mlp, moe_combine,
          post_combine, mtp_post_process=None).
        - backward_dw: Dict mapping slot name to the delayed-wgrad callable
          (keys: "pre_dispatch_computation", "mlp").
    """

    is_moe = isinstance(layer.mlp, MoELayer)
    enable_deepep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend in ("deepep", "deepepv2")
    )
    enable_hybridep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "hybridep"
    )
    enable_ncclep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "ncclep"
    )

    def checkpoint_region(forward_func, *inputs):
        """Checkpoint one computation segment with the active precision recipe."""
        if layer.config.fp8 or layer.config.fp4:
            from megatron.core.extensions.transformer_engine import te_checkpoint

            return te_checkpoint(
                forward_func,
                False,
                tensor_parallel.random.get_cuda_rng_tracker,
                layer.pg_collection.tp,
                *inputs,
            )
        return tensor_parallel.checkpoint(forward_func, False, *inputs)

    def submodule_pre_dispatch_forward(node: ScheduleNode, hidden_states: torch.Tensor):
        """
        Performs the same attention forward logic as GPTModel and the forward pass for
        computations between attention and dispatch:
            pre mlp layernorm->router->dispatch preprocess
        """

        use_cuda_graph = (
            isinstance(layer, GraphableMegatronModule)
            and hasattr(layer, 'cuda_graphs')
            and layer.cuda_graphs
        )
        if use_cuda_graph:
            if (
                layer.a2a_overlap_attn_recompute
                or layer.a2a_overlap_post_attn_recompute
                or layer.a2a_overlap_mlp_recompute
            ):
                raise RuntimeError(
                    "A2A overlap node recompute selectors require the eager fine-grained "
                    "schedule and cannot be isolated from TE CUDA graph replay."
                )
            layer.set_te_cuda_graph_backward_dw_wrapper()
            forward_func = layer._te_cuda_graph_replay
        else:
            # wrapper function that keeps consistent api with cuda graph replay
            def forward_func(
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                rotary_pos_emb: Optional[Tensor] = None,
                rotary_pos_cos: Optional[Tensor] = None,
                rotary_pos_sin: Optional[Tensor] = None,
                packed_seq_params: Optional[PackedSeqParams] = None,
                sequence_len_offset: Optional[Tensor] = None,
                padding_mask: Optional[Tensor] = None,
            ):
                def attention_forward(attention_input):
                    attention_output, _ = layer._forward_attention(
                        hidden_states=attention_input,
                        attention_mask=attention_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        padding_mask=padding_mask,
                    )
                    return attention_output

                if layer.a2a_overlap_attn_recompute:
                    hidden_states = checkpoint_region(attention_forward, hidden_states)
                else:
                    hidden_states = attention_forward(hidden_states)
                if getattr(layer.config, 'moe_echo_expert_dispatch_overlap', False):
                    hidden_states = FlushPendingGradAccum.apply(hidden_states)
                if not isinstance(layer.mlp, MoELayer):
                    return hidden_states, None, None, None

                if layer.a2a_overlap_post_attn_recompute:
                    node.layer_state.mlp_norm_manager = None
                    post_attn_calls = [0]

                    def post_attn_forward(post_attention_input):
                        # ``dispatch_preprocess`` mutates the token dispatcher's routing
                        # metadata. During checkpoint recompute that mutation may run after
                        # the next microbatch has already advanced the dispatcher state.
                        # Restore the state that was current when recompute started.
                        is_recompute = post_attn_calls[0] > 0
                        post_attn_calls[0] += 1
                        dispatcher_state = (
                            dict(layer.mlp.token_dispatcher.__dict__) if is_recompute else None
                        )
                        try:
                            pre_mlp_layernorm_output = apply_module(layer.pre_mlp_layernorm)(
                                post_attention_input
                            )
                            if layer._pre_mlp_layernorm_returns_residual:
                                pre_mlp_layernorm_output, post_attention_input = (
                                    pre_mlp_layernorm_output
                                )
                            probs, routing_map = layer.mlp.route(
                                pre_mlp_layernorm_output,
                                padding_mask,
                                packed_seq_params=packed_seq_params,
                            )
                            local_tokens, probs = layer.mlp.preprocess(
                                pre_mlp_layernorm_output,
                                probs,
                                routing_map,
                                padding_mask,
                            )
                            return (
                                post_attention_input,
                                pre_mlp_layernorm_output,
                                local_tokens,
                                probs,
                            )
                        finally:
                            _restore_dispatcher_state(
                                layer.mlp.token_dispatcher, dispatcher_state
                            )

                    (
                        hidden_states,
                        pre_mlp_layernorm_output,
                        local_tokens,
                        probs,
                    ) = checkpoint_region(post_attn_forward, hidden_states)
                else:
                    mlp_norm_manager = off_interface(
                        layer.offload_mlp_norm, hidden_states, "mlp_norm"
                    )
                    node.layer_state.mlp_norm_manager = mlp_norm_manager
                    if layer.recompute_pre_mlp_layernorm:
                        layer.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                            retain_input_tensors=layer._pre_mlp_layernorm_returns_residual
                        )
                        with mlp_norm_manager as hidden_states:
                            pre_mlp_layernorm_output = layer.pre_mlp_norm_checkpoint.checkpoint(
                                apply_module(layer.pre_mlp_layernorm), hidden_states
                            )
                    else:
                        with mlp_norm_manager as hidden_states:
                            pre_mlp_layernorm_output = apply_module(layer.pre_mlp_layernorm)(
                                hidden_states
                            )

                    # When using fused residual norm (e.g. TEFusedResidualRMSNorm),
                    # the layernorm returns (normalized_output, residual). Unpack
                    # and use the fused residual for the downstream BDA connection.
                    if layer._pre_mlp_layernorm_returns_residual:
                        pre_mlp_layernorm_output, hidden_states = pre_mlp_layernorm_output

                if layer.config.moe_enable_echo:
                    node.layer_state.pre_mlp_layernorm_output = pre_mlp_layernorm_output
                    shared_expert_output = (
                        None
                        if layer.a2a_overlap_mlp_recompute
                        else layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
                    )
                    return hidden_states, None, None, shared_expert_output

                shared_expert_output = (
                    layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
                    if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap
                    else None
                )
                # The shared expert is computed eagerly here, matching the
                # reference MoELayer path. The A2A-overlap MLP checkpoint only
                # replays routed experts; replaying shared experts as well
                # introduces BF16 gradient ULP drift.
                node.layer_state.pre_mlp_layernorm_output = None

                if layer.a2a_overlap_post_attn_recompute:
                    return hidden_states, local_tokens, probs, shared_expert_output

                # Same routing inputs as the eager MoELayer.forward path: the padding mask is
                # consumed by the router (dropless HybridEP excludes padded tokens).
                probs, routing_map = layer.mlp.route(
                    pre_mlp_layernorm_output, padding_mask, packed_seq_params=packed_seq_params
                )
                local_tokens, probs = layer.mlp.preprocess(
                    pre_mlp_layernorm_output, probs, routing_map, padding_mask
                )
                return hidden_states, local_tokens, probs, shared_expert_output

        hidden_states, local_tokens, probs, shared_expert_output = forward_func(
            hidden_states=hidden_states,
            attention_mask=node.chunk_state.attention_mask,
            rotary_pos_emb=node.chunk_state.rotary_pos_emb,
            rotary_pos_cos=node.chunk_state.rotary_pos_cos,
            rotary_pos_sin=node.chunk_state.rotary_pos_sin,
            packed_seq_params=node.chunk_state.packed_seq_params,
            sequence_len_offset=node.chunk_state.sequence_len_offset,
            padding_mask=getattr(node.chunk_state, "padding_mask", None),
        )
        if not isinstance(layer.mlp, MoELayer):
            return hidden_states

        pre_mlp_layernorm_output = getattr(node.layer_state, 'pre_mlp_layernorm_output', None)
        if layer.config.moe_enable_echo:
            if pre_mlp_layernorm_output is None:
                raise RuntimeError("ECHO routing requires the pre-MLP layernorm output.")
            routing_map, probs, expert_offloading_map = layer.mlp._echo_route(
                pre_mlp_layernorm_output
            )
            local_tokens, probs = layer.mlp.preprocess(
                pre_mlp_layernorm_output, probs, routing_map
            )
            node.layer_state.echo_expert_offloading_map = expert_offloading_map

        node.layer_state.pre_mlp_layernorm_output = None

        # Detach here for mlp_bda residual connection
        node.layer_state.residual = node.detach(hidden_states)
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            # Detach here for shared expert connection in moe_combine
            node.layer_state.shared_expert_output = node.detach(shared_expert_output)

        return local_tokens, probs

    def submodule_dispatch_forward(
        node: ScheduleNode, local_tokens: torch.Tensor, probs: torch.Tensor
    ):
        """
        Dispatches tokens to the experts based on the router output.
        """
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep or enable_ncclep:
            metadata = token_dispatcher.metadata
            # update token_probs to be the detached version, prevents
            # backward graph from connecting to pre_dispatch_computation submodule
            metadata.token_probs = probs

        dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs)

        if enable_ncclep and layer.config.moe_ncclep_zero_copy:
            # Insert an identity node as the sole consumer of the dispatch output, so the
            # dispatch-backward gets the symm buffer instead of a non-symm AccumulateGrad clone.
            # Must stay inside this node's graph segment (before the next node detaches it).
            dispatched_tokens = StageDispatchBwdGrad.apply(dispatched_tokens, token_dispatcher)

        # `dispatched_probs` is needed by backward pass of swiglu, therefore it's
        # passed to moe_forward within `layer_state` to avoid the free_input process
        # of the input tensors.
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    def submodule_moe_forward(node: ScheduleNode, dispatched_tokens: torch.Tensor):
        """
        Run forward pass for computations between dispatch and combine:
            post dispatch->experts->combine preprocess
        """
        dispatched_probs = node.layer_state.dispatched_probs
        pre_mlp_layernorm_output = getattr(node.layer_state, 'pre_mlp_layernorm_output', None)
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep or enable_ncclep:
            metadata = token_dispatcher.metadata
            # update dispatched_probs to be detached version, prevents
            # backward graph from connecting to dispatch submodule
            metadata.dispatched_probs = dispatched_probs

        fc1_expert_dispatch_metadata = None
        fc2_expert_dispatch_metadata = None
        if layer.config.moe_enable_echo:
            expert_offloading_map = node.layer_state.echo_expert_offloading_map
            fc1_expert_dispatch_metadata, fc2_expert_dispatch_metadata = (
                layer.mlp._prepare_echo_expert_dispatch(expert_offloading_map)
            )
            node.layer_state.echo_expert_offloading_map = None

        def routed_expert_forward(routed_tokens, routed_probs):
            if layer.config.moe_enable_echo:
                return layer.mlp.echo_routed_experts_compute(
                    routed_tokens,
                    routed_probs,
                    fc1_expert_dispatch_metadata,
                    fc2_expert_dispatch_metadata,
                )
            expert_output, _ = layer.mlp.routed_experts_compute(routed_tokens, routed_probs)
            return expert_output

        if layer.a2a_overlap_mlp_recompute:
            recompute_shared_experts = False

            def checkpointed_experts(
                routed_tokens,
                routed_tokens_per_expert,
                routed_probs,
                shared_expert_input=None,
            ):
                shared_expert_output = (
                    layer.mlp.shared_experts_compute(shared_expert_input)
                    if recompute_shared_experts
                    else None
                )
                if layer.config.moe_enable_echo:
                    expert_output, _ = layer.mlp._run_echo_experts(
                        routed_tokens,
                        routed_tokens_per_expert,
                        routed_probs,
                        fc1_expert_dispatch_metadata,
                        fc2_expert_dispatch_metadata,
                    )
                else:
                    expert_output, _ = layer.mlp._run_experts(
                        routed_tokens, routed_tokens_per_expert, routed_probs
                    )
                return expert_output, shared_expert_output

            checkpointed_experts_calls = [0]
            dispatcher_state = None

            def recompute_safe_experts(*args):
                nonlocal dispatcher_state
                is_recompute = checkpointed_experts_calls[0] > 0
                checkpointed_experts_calls[0] += 1
                if is_recompute:
                    dispatcher_state = _snapshot_dispatcher_state(layer.mlp.token_dispatcher)
                try:
                    return checkpointed_experts(*args)
                finally:
                    _restore_dispatcher_state(layer.mlp.token_dispatcher, dispatcher_state)
                    if is_recompute:
                        dispatcher_state = None

            permuted_tokens, tokens_per_expert, permuted_probs = (
                layer.mlp._dispatch_postprocess_for_experts(dispatched_tokens, dispatched_probs)
            )
            checkpoint_inputs = (permuted_tokens, tokens_per_expert, permuted_probs)
            if recompute_shared_experts:
                checkpoint_inputs += (pre_mlp_layernorm_output,)
            expert_output, shared_expert_output = checkpoint_region(
                recompute_safe_experts, *checkpoint_inputs
            )
            expert_output = layer.mlp.token_dispatcher.combine_preprocess(expert_output)
            if recompute_shared_experts:
                node.layer_state.shared_expert_output = node.detach(shared_expert_output)
            node.layer_state.pre_mlp_layernorm_output = None
        else:
            expert_output = routed_expert_forward(dispatched_tokens, dispatched_probs)

        # For HybridEP and NCCL EP, tokens_per_expert is generated on comm stream, as the
        # input to `routed_experts_compute`, a ref is needed to prevent it from being freed.
        if enable_hybridep or enable_ncclep:
            tokens_per_expert = token_dispatcher.metadata.get_number_of_tokens_per_expert()
            node.layer_state.tokens_per_expert = tokens_per_expert

        if (
            layer.recompute_pre_mlp_layernorm
            and not layer.a2a_overlap_post_attn_recompute
        ):
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of expert_output
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

        return expert_output

    def submodule_combine_forward(node: ScheduleNode, output: torch.Tensor):
        """Run the MoE token-combine communication."""
        return layer.mlp.combine(output)

    def submodule_post_combine_forward(node: ScheduleNode, output: torch.Tensor):
        """
        # Run the remaining computation in the transformer layer after token combine.
        # The `mlp_bda` computation is placed after `mlp.combine` due to data dependency.
        # This ordering is also critical for pipeline performance. Starting the `mlp.combine`
        # communication at first allows it to be overlapped with computation from another
        # microbatch. If `mlp_bda` were to run first, it would compete for SM resources
        # with another microbatch's computation and expose the communication.
        """
        residual = node.layer_state.residual
        shared_expert_output = getattr(node.layer_state, 'shared_expert_output', None)
        # Without dropout, the shared-expert add of `postprocess` and the residual add of
        # `mlp_bda` run as one kernel that is bit-identical to the two adds.
        fuse_residual_add = (
            shared_expert_output is not None
            and not layer.a2a_overlap_mlp_recompute
            and layer.mlp_bda is get_bias_dropout_add
            and (layer.hidden_dropout == 0.0 or not layer.training)
        )
        output = layer.mlp.postprocess(output, None if fuse_residual_add else shared_expert_output)
        # Check the kernel's requirements on the postprocessed output: `postprocess` restores the
        # token order and the [s, b, h] shape of the combined tokens.
        should_fuse_residual_add = fuse_residual_add and can_use_fused_moe_residual_add(
            output, shared_expert_output, residual
        )
        if fuse_residual_add and not should_fuse_residual_add:
            output = output + shared_expert_output

        if hasattr(layer, 'cuda_graphs') and layer.cuda_graphs:
            layer.mlp.cudagraph_tensor_store.clear()
        with layer.bias_dropout_add_exec_handler():
            if should_fuse_residual_add:
                hidden_states = fused_moe_residual_add(output, shared_expert_output, residual)
            else:
                hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                    (output, None), residual, layer.hidden_dropout
                )
        # Delay the offload of the mlp norm until after the mlp_bda has been computed
        # because the residual is needed in the mlp_bda.
        mlp_norm_manager = getattr(node.layer_state, 'mlp_norm_manager', None)
        if mlp_norm_manager is not None:
            hidden_states = mlp_norm_manager.group_offload(
                hidden_states, forced_released_tensors=[residual]
            )
            node.layer_state.mlp_norm_manager = None
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # Need to record tensors created on comp stream to comm stream
        node.layer_state.residual.record_stream(torch.cuda.current_stream())
        if shared_expert_output is not None:
            shared_expert_output.record_stream(torch.cuda.current_stream())

        # release tensor reference after use
        node.layer_state.residual = None
        node.layer_state.shared_expert_output = None

        # final layer norm from decoder
        final_layernorm = node.chunk_state.model.decoder.final_layernorm
        if not node.is_mtp and final_layernorm and node.is_last_layer:
            output = final_layernorm(output)
            output = make_viewless_tensor(inp=output, requires_grad=True, keep_graph=True)
        return output

    @copy_signature(layer._forward_mlp, handle_first_dst_param='preserve')
    def mlp_wrapper(node: ScheduleNode, *args, **kwargs):
        """Wrapper for Dense forward."""
        return layer._forward_mlp(*args, **kwargs)

    def raise_not_implemented(*args):
        """Raise NotImplementedError for Dense layer."""
        raise NotImplementedError("This callable is not implemented for Dense layer.")

    # Build forward and backward callable functions
    pre_dispatch_func = submodule_pre_dispatch_forward
    dispatch_func = submodule_dispatch_forward if is_moe else raise_not_implemented
    mlp_func = submodule_moe_forward if is_moe else mlp_wrapper
    combine_func = submodule_combine_forward if is_moe else raise_not_implemented
    post_combine_func = submodule_post_combine_forward if is_moe else raise_not_implemented

    layer.init_backward_dw_wrapper()

    forward_funcs = [
        pre_dispatch_func,
        dispatch_func,
        mlp_func,
        combine_func,
        post_combine_func,
        None,
    ]
    backward_dw = {"pre_dispatch_computation": layer.backward_dw_wrapper, "mlp": layer.mlp}
    return forward_funcs, backward_dw
