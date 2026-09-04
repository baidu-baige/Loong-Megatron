# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Expert module implementations for mixture-of-experts layers."""

import copy
import itertools
from copy import deepcopy
from functools import partial, wraps
from math import ceil
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.core import parallel_state, tensor_parallel
from megatron.core.activations import squared_relu
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.jit import jit_fuser
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.mlp import MLP, MLPSubmodules, apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    make_sharded_object_for_checkpoint,
    sharded_state_dict_default,
)

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import Fp8Padding, Fp8Unpadding

    HAVE_TE = True

except ImportError:

    HAVE_TE = False


# TODO(Hepteract): delete the usage of the global parallel_state.
# Currently we still have to use the global parallel_state in expert_dist_ckpt_decorator(),
# in order to set sub-module's process group while getting sharded_state_dict.
# After sub-module's refactoring is done, we can pass pg_collection to sub-module
# and delete the function expert_dist_ckpt_decorator.
def expert_dist_ckpt_decorator(func):
    """Decorator of shared_state_dict in expert layer for distributed checkpoint.

    Since !1940, the TP size for Expert layer can be different with Attention.
    To make distributed checkpoint work in such cases, we use a decorator to
    replace the default TP parallel states with expert-TP parallel states.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Store original states
        original_rank = parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK
        original_size = parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
        original_group = parallel_state._TENSOR_MODEL_PARALLEL_GROUP
        try:
            # Set new states
            parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = (
                parallel_state.get_expert_tensor_parallel_rank()
            )
            parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = (
                parallel_state.get_expert_tensor_parallel_world_size()
            )
            parallel_state._TENSOR_MODEL_PARALLEL_GROUP = (
                parallel_state.get_expert_tensor_parallel_group()
            )

            # Execute the function
            result = func(*args, **kwargs)
        finally:
            # Restore original states
            parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = original_rank
            parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = original_size
            parallel_state._TENSOR_MODEL_PARALLEL_GROUP = original_group
        return result

    return wrapper


class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            @jit_fuser
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func
        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and self.config.fp8:
            raise ValueError("moe_act recompute for fp8 cannot work with the legacy GroupedMLP.")

        @jit_fuser
        def activation_func_with_probs(x, probs):
            dtype = x.dtype
            res = self.activation_func(x) * probs
            return res.to(dtype)

        self.activation_func_with_probs = activation_func_with_probs

        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.tp_group = pg_collection.expt_tp
        # use pg_collection.expt_dp_group as data parallel group in this module.
        self.dp_group = pg_collection.expt_dp
        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition, self.config.hidden_size, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1, config.init_method, partition_dim=1, is_expert=True
                )
                _initialize_affine_weight_gpu(
                    self.weight2, config.output_layer_init_method, partition_dim=0, is_expert=True
                )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

        def remove_extra_states_check(self, incompatible_keys):
            """
            Remove _extra_state from unexpected keys.
            These keys are for dist ckpt compatibility with SequentialMLP.
            """
            keys = deepcopy(incompatible_keys.unexpected_keys)
            for key in keys:
                if '_extra_state' in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        fc1_expert_dispatch_event=None,
        fc2_expert_dispatch_event=None,
    ):
        """Forward step of the GroupedMLP."""
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )
            if self.activation_recompute:
                intermediate_parallel = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                h = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, h, permuted_probs.unsqueeze(-1)
                )
                fc2_output = torch.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
                fc2_output = torch.matmul(h, w2)

        return fc2_output, None

    @expert_dist_ckpt_decorator
    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        Maps local expert to global experts.
        The sharded_state_dict for the weight parts are compatible with the SequentialMLP,
        whereas the optimizer states are not due to the limitation from weight transposing.
        That is, for finetuning scenario, the checkpoint is compatible with the SequentialMLP.

        When `singleton_local_shards` metadata flag is True, experts are broken down into
        separate tensors and stored under separate global keys. Additionally, similarly to MLP,
        layers with GLU activations are broken down into separate `w` and `v` tensors.
        """
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()
        dp_rank = self.dp_group.rank()
        num_global_experts = ep_size * self.num_local_experts
        local_expert_indices_offset = ep_rank * self.num_local_experts

        prepend_axis_num = len(sharded_offsets)
        replica_id = (0, 0, dp_rank)

        local_ffn_dim_size = (
            self.weight2.numel() // self.num_local_experts // self.config.hidden_size
        )

        def _break_into_individual_experts(
            experts_ten: torch.Tensor,
            key: str,
            tp_offset: Tuple[int, int, int],
            replica_id: ReplicaId,
        ):
            """Breaks experts into individual tensors and stores them under separate global keys"""
            experts_state = []
            assert len(experts_ten) == self.num_local_experts, (
                experts_ten.shape,
                self.num_local_experts,
            )
            for local_expert_idx, expert_ten in enumerate(experts_ten):
                global_expert_idx = local_expert_indices_offset + local_expert_idx
                expert_key = key.replace(
                    f'{prefix}experts.', f'{prefix}experts.{global_expert_idx}.'
                )
                experts_state.append(
                    ShardedTensor.from_rank_offsets(
                        expert_key,
                        expert_ten.contiguous(),
                        *sharded_offsets,
                        tp_offset,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    )
                )
            return experts_state

        @torch.no_grad()
        def sh_ten_build_fn(
            key: str,
            t: torch.Tensor,
            replica_id: ReplicaId,
            flattened_range: Optional[slice],
            tp_axis: int,
            with_glu: bool,
        ):
            # TODO: write a generic implementation to cover both cases with and without GLU
            if tp_axis == 1:
                # weight1
                if with_glu:
                    last_dim_size = local_ffn_dim_size * 2
                else:
                    last_dim_size = local_ffn_dim_size
                real_shape = (self.num_local_experts, self.config.hidden_size, last_dim_size)
            elif tp_axis == 0:
                # weight2
                real_shape = (self.num_local_experts, local_ffn_dim_size, self.config.hidden_size)
                assert with_glu == False
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if flattened_range is None:
                # weights
                t = t.view(real_shape).transpose(-1, -2)
                # change tp_axis due to the transposing
                tp_axis = 1 - tp_axis
                if with_glu:
                    assert tp_axis == 0, tp_axis
                    if singleton_local_shards:
                        w_tensor, v_tensor = torch.chunk(t, 2, -2)
                        w_key = f'{key}_w'
                        v_key = f'{key}_v'
                        sub_states = {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': {
                                'w': _break_into_individual_experts(
                                    w_tensor,
                                    w_key,
                                    (prepend_axis_num, tp_rank, tp_size),
                                    replica_id,
                                ),
                                'v': _break_into_individual_experts(
                                    v_tensor,
                                    v_key,
                                    (prepend_axis_num, tp_rank, tp_size),
                                    replica_id,
                                ),
                            },
                        }
                    else:
                        local_tensors = torch.chunk(t, 2, -2)
                        sub_states = [
                            ShardedTensor.from_rank_offsets(
                                key,
                                local_tensors[0].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                            ShardedTensor.from_rank_offsets(
                                key,
                                local_tensors[1].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_size + tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                        ]
                else:
                    if singleton_local_shards:
                        sub_states = {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': _break_into_individual_experts(
                                t, key, (prepend_axis_num + tp_axis, tp_rank, tp_size), replica_id
                            ),
                        }
                    else:
                        sub_states = ShardedTensor.from_rank_offsets(
                            key,
                            t.contiguous(),
                            *sharded_offsets,
                            (prepend_axis_num, ep_rank, ep_size),
                            (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                            replica_id=replica_id,
                            prepend_axis_num=prepend_axis_num,
                        )
            else:
                if singleton_local_shards:
                    raise NotImplementedError(
                        'flattened_range not supported for'
                        ' GroupedMLP with singleton_local_shards'
                    )
                # flattened optmizer states
                # the non-flattened weight shape is [local_expert_num, hidden_size, ffn_size]
                #
                # For the case without GLU, it is straightforward, we just need to split each
                # expert along the dim-0.
                #
                # For the case with GLU, we need to split the experts along dim-0 and split the
                # two tensors for GLU along dim-2.
                # To split along the non-first dim, we need to chunk the tensor into small pieces,
                # since they belong to different tenors and are interleaved in the flattened space.
                # Refer to the below sketch graph.
                # |................|           |........|........|
                # |............FFFF|           |........|....BBBB|
                # |FFFFFFFFFFFFFFFF|     ->    |AAAAAAAA|BBBBBBBB|
                # |FFFFFFFFFFFFFFFF|           |AAAAAAAA|BBBBBBBB|
                # |FF..............|           |AA......|........|
                # |................|           |........|........|
                #
                # But too many chunks have severe performance issues. We merge these chunks during
                # the save process along with some length information and recover them during the
                # load process.
                assert t.ndim == 1, (key, t.shape)
                if with_glu:
                    non_flat_local_shape = (1, self.config.hidden_size, local_ffn_dim_size)
                    chunk_numel = local_ffn_dim_size
                    sub_states = []
                    start_pos = 0
                    for local_expert_idx in range(self.num_local_experts):
                        first_glu_idx = -1
                        w_start_range = -1
                        v_start_range = -1
                        w_tensors = []
                        v_tensors = []
                        w_lens = []
                        v_lens = []
                        expert_global_idx = local_expert_indices_offset + local_expert_idx
                        for input_dim_idx in range(self.config.hidden_size):
                            for glu_idx in range(2):
                                local_idx = (
                                    local_expert_idx * self.config.hidden_size * 2
                                    + input_dim_idx * 2
                                    + glu_idx
                                )
                                if (
                                    flattened_range.start < chunk_numel * (local_idx + 1)
                                    and flattened_range.stop > chunk_numel * local_idx
                                ):
                                    if first_glu_idx == -1:
                                        first_glu_idx = glu_idx
                                    end_pos = min(
                                        flattened_range.stop,
                                        chunk_numel * (local_idx + 1) - flattened_range.start,
                                    )
                                    local_tensor = t[start_pos:end_pos]
                                    local_flattened_range = slice(
                                        max(0, flattened_range.start - chunk_numel * local_idx),
                                        min(
                                            chunk_numel,
                                            flattened_range.stop - chunk_numel * local_idx,
                                        ),
                                    )
                                    assert (
                                        len(local_tensor)
                                        == local_flattened_range.stop - local_flattened_range.start
                                    )
                                    start_pos += len(local_tensor)
                                    if glu_idx == 0:
                                        w_tensors.append(local_tensor)
                                        w_lens.append(len(local_tensor))
                                        if w_start_range == -1:
                                            w_start_range = max(
                                                0, flattened_range.start - chunk_numel * local_idx
                                            )
                                    else:
                                        v_tensors.append(local_tensor)
                                        v_lens.append(len(local_tensor))
                                        if v_start_range == -1:
                                            v_start_range = max(
                                                0, flattened_range.start - chunk_numel * local_idx
                                            )
                        sub_states.append(
                            {
                                'w_tensors': ShardedTensor.from_rank_offsets_flat(
                                    key,
                                    (
                                        torch.cat(w_tensors, -1)
                                        if len(w_tensors) > 0
                                        else torch.Tensor()
                                    ),
                                    non_flat_local_shape,
                                    *sharded_offsets,
                                    (
                                        prepend_axis_num,
                                        expert_global_idx,  # pylint: disable=E0606
                                        num_global_experts,
                                    ),
                                    (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size * 2),
                                    replica_id=replica_id,
                                    prepend_axis_num=prepend_axis_num,
                                    flattened_range=slice(
                                        w_start_range, w_start_range + sum(w_lens)
                                    ),
                                ),
                                'w_lens': LocalNonpersistentObject(w_lens),
                                'v_tensors': ShardedTensor.from_rank_offsets_flat(
                                    key,
                                    (
                                        torch.cat(v_tensors, -1)
                                        if len(v_tensors) > 0
                                        else torch.Tensor()
                                    ),
                                    non_flat_local_shape,
                                    *sharded_offsets,
                                    (prepend_axis_num, expert_global_idx, num_global_experts),
                                    (
                                        prepend_axis_num + 1 + tp_axis,
                                        tp_rank + tp_size,
                                        tp_size * 2,
                                    ),
                                    replica_id=replica_id,
                                    prepend_axis_num=prepend_axis_num,
                                    flattened_range=slice(
                                        v_start_range, v_start_range + sum(v_lens)
                                    ),
                                ),
                                'v_lens': LocalNonpersistentObject(v_lens),
                                'first_glu_idx': LocalNonpersistentObject(first_glu_idx),
                            }
                        )
                else:
                    non_flat_local_shape = (
                        real_shape[0] // self.num_local_experts,
                        *real_shape[1:],
                    )
                    chunk_numel = local_ffn_dim_size * self.config.hidden_size
                    sub_states = []
                    start_pos = 0
                    for local_expert_idx in range(self.num_local_experts):
                        if (
                            flattened_range.start < chunk_numel * (local_expert_idx + 1)
                            and flattened_range.stop > chunk_numel * local_expert_idx
                        ):
                            end_pos = min(
                                flattened_range.stop,
                                chunk_numel * (local_expert_idx + 1) - flattened_range.start,
                            )
                            local_tensor = t[start_pos:end_pos]
                            local_flattened_range = slice(
                                max(0, flattened_range.start - chunk_numel * local_expert_idx),
                                min(
                                    chunk_numel,
                                    flattened_range.stop - chunk_numel * local_expert_idx,
                                ),
                            )
                            assert (
                                len(local_tensor)
                                == local_flattened_range.stop - local_flattened_range.start
                            )
                            start_pos += len(local_tensor)
                            expert_global_idx = local_expert_indices_offset + local_expert_idx
                            sub_states.append(
                                ShardedTensor.from_rank_offsets_flat(
                                    key,
                                    local_tensor,
                                    non_flat_local_shape,
                                    *sharded_offsets,
                                    (prepend_axis_num, expert_global_idx, num_global_experts),
                                    (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                                    replica_id=replica_id,
                                    prepend_axis_num=prepend_axis_num,
                                    flattened_range=local_flattened_range,
                                )
                            )
            return sub_states

        @torch.no_grad()
        def sh_ten_merge_fn(sub_state_dict, tp_axis: int, with_glu: bool):
            if tp_axis == 1:
                # weight1
                weight_shape = (self.config.hidden_size, -1)
            elif tp_axis == 0:
                # weight2
                weight_shape = (-1, self.config.hidden_size)
                assert with_glu == False
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if isinstance(sub_state_dict, list) and isinstance(sub_state_dict[0], dict):
                # flattened tensor with glu
                res = []
                for local_expert_dict in sub_state_dict:
                    w_tensors = torch.split(
                        local_expert_dict['w_tensors'], local_expert_dict['w_lens']
                    )
                    v_tensors = torch.split(
                        local_expert_dict['v_tensors'], local_expert_dict['v_lens']
                    )
                    first_glu_idx = local_expert_dict['first_glu_idx']
                    if first_glu_idx == 0:
                        res += [
                            x for x in itertools.chain(*itertools.zip_longest(w_tensors, v_tensors))
                        ]
                    else:
                        res += [
                            x for x in itertools.chain(*itertools.zip_longest(v_tensors, w_tensors))
                        ]
                return torch.cat(res)
            elif isinstance(sub_state_dict, list) and sub_state_dict[0].ndim == 1:
                # flattened tensor without glu
                return torch.cat(sub_state_dict)
            else:
                if isinstance(sub_state_dict, dict):
                    assert sub_state_dict['singleton_local_shards']
                    if with_glu:
                        assert isinstance(sub_state_dict['data'], dict)
                        sub_state_dict = torch.cat(
                            (
                                torch.stack(sub_state_dict['data']['w']),
                                torch.stack(sub_state_dict['data']['v']),
                            ),
                            dim=-2,
                        )
                    else:
                        assert isinstance(sub_state_dict['data'], list)
                        sub_state_dict = torch.stack(sub_state_dict['data'])
                else:
                    if with_glu:
                        sub_state_dict = torch.cat(sub_state_dict, -2)
                return sub_state_dict.transpose(-1, -2).reshape(weight_shape)

        state_dict = self.state_dict(prefix='', keep_vars=True)
        for name, tensor in state_dict.items():
            if name == 'weight1':
                tp_axis = 1
                with_glu = self.config.gated_linear_unit
                wkey = f'{prefix}experts.linear_fc1.weight'
            else:
                tp_axis = 0
                with_glu = False
                wkey = f'{prefix}experts.linear_fc2.weight'

            this_replica_id = list(copy.deepcopy(replica_id))
            flattened_range = None

            sharded_state_dict[f'{prefix}{name}'] = ShardedTensorFactory(
                wkey,
                tensor,
                partial(sh_ten_build_fn, tp_axis=tp_axis, with_glu=with_glu),
                partial(sh_ten_merge_fn, tp_axis=tp_axis, with_glu=with_glu),
                tuple(this_replica_id),
                flattened_range=flattened_range,
            )

        replica_id = (0, tp_rank, dp_rank)
        # Add fake _extra_state to be compatible with SequentialMLP
        for expert_local_idx in range(self.num_local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            if singleton_local_shards:
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )
            for mod in ['linear_fc1', 'linear_fc2']:
                if singleton_local_shards:
                    expert_key = f'{prefix}experts.{expert_global_idx}.{mod}._extra_state'
                else:
                    expert_key = f'{prefix}experts.{mod}._extra_state'
                sharded_state_dict[f'{prefix}expert{expert_global_idx}.{mod}._extra_state'] = (
                    make_sharded_object_for_checkpoint(
                        None, expert_key, expert_sharded_offsets, replica_id
                    )
                )

        return sharded_state_dict

    def backward_dw(self):
        """Performs backward pass for weight gradients in Experts.
        Empty implementation for compatibility with SequentialMLP and TEGroupedMLP.
        """
        pass


class DummyFunction(torch.autograd.Function):
    """Autograd placeholder for empty expert weight tensors."""

    @staticmethod
    def forward(ctx, x, weight_shape):
        """Create a placeholder weight tensor for forward computation."""
        ctx.input = x
        dummy_weight = torch.empty(weight_shape, dtype=x.dtype, device=x.device)
        # dummy_weight = torch.zeros(weight_shape, dtype=x.dtype, device=x.device)
        return dummy_weight

    @staticmethod
    def backward(ctx, grad_output):
        """Return a zero gradient for the placeholder input."""
        return torch.zeros_like(ctx.input), None


class TEGroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.input_size = getattr(self.config, "moe_latent_size", None) or self.config.hidden_size
        assert not (
            self.config.add_bias_linear and config.bias_dropout_fusion
        ), "bias_dropout_fusion is not supported in TEGroupedMLP when add_bias_linear=True"

        self.ep_group = pg_collection.ep

        # Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.moe_ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        # TODO(Hepteract): pass pg_collection to submodule after refactoring Linear modules
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.num_local_experts,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=True,
            tp_comm_buffer_name='fc1',
            tp_group=pg_collection.expt_tp,
        )
        self.fc1_weight_shape = self.linear_fc1.weight0.shape

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = build_module(submodules.activation_func, config=self.config)
        else:
            self.activation_func = self.config.activation_func

        # TODO(Hepteract): pass pg_collection to submodule after refactoring Linear modules
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.num_local_experts,
            self.config.moe_ffn_hidden_size,
            self.input_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name='fc2',
            tp_group=pg_collection.expt_tp,
        )
        self.fc2_weight_shape = self.linear_fc2.weight0.shape

        self.offload_expert_fc1 = (
            self.config.fine_grained_activation_offloading
            and "expert_fc1" in self.config.offload_modules
        )

        self.offload_moe_act = (
            self.config.fine_grained_activation_offloading
            and "moe_act" in self.config.offload_modules
        )

        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and self.config.fp8:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc2)

        # This is to avoid the CPU overhead of multiple d2h copies
        if self.offload_expert_fc1 and not (self.config.fp8 or self.config.fp4):
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc1)

        if self.config.fp8:
            assert HAVE_TE, "FP8 requires TE."
            self.fp8_padding = Fp8Padding(self.num_local_experts)
            self.fp8_unpadding = Fp8Unpadding(self.num_local_experts)

    @staticmethod
    def _apply_bias(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
        """Apply per-expert bias scaled by routed probabilities."""
        if bias_parallel is None:
            return intermediate_parallel
        shape = intermediate_parallel.shape
        return (
            torch.cat(
                [
                    t + b * p
                    for t, b, p in zip(
                        torch.split(intermediate_parallel.view(-1, shape[-1]), tokens_per_expert),
                        bias_parallel,
                        torch.split(permuted_probs, tokens_per_expert),
                    )
                ]
            )
            .view(shape)
            .to(intermediate_parallel.dtype)
        )

    def free_expert_parameters(self, expert_indices: List[int]):
        """Free echo expert parameters."""
        to_free_weight_names = [f'weight{i}' for i in expert_indices]
        for module in [self.linear_fc1, self.linear_fc2]:
            # Clear all parameters in the module
            for name, param in list(module.named_parameters()):
                if name in to_free_weight_names:
                    delattr(module, name)
                    module._parameters.pop(name, None)

    def get_expert_weights(
        self, module, expert_indices: List[int]
    ) -> List[torch.Tensor]:
        """Get the weights of the experts."""
        assert module in ["fc1", "fc2"], f"Invalid module: {module}"
        expert_layer = self.linear_fc1 if module == "fc1" else self.linear_fc2
        weight_list = []
        for i in expert_indices:
            weight = getattr(expert_layer, f"weight{i}")
            weight_list.append(weight)
        return weight_list

    def set_expert_weights(
        self,
        module,
        expert_weights: List[torch.Tensor],
        expert_indices: List[int],
    ):
        """Set the weights of the experts."""
        expert_layer = self.linear_fc1 if module == "fc1" else self.linear_fc2
        weight_shape = self.fc1_weight_shape if module == "fc1" else self.fc2_weight_shape
        for i, expert_index in enumerate(expert_indices):
            if expert_weights[i].numel() == 0:
                setattr(
                    expert_layer,
                    f"weight{expert_index}",
                    DummyFunction.apply(expert_weights[i], weight_shape),
                )
            else:
                setattr(
                    expert_layer,
                    f"weight{expert_index}",
                    expert_weights[i],
                )

    def _apply_activation(
        self,
        fc1_output: torch.Tensor,
        bias_parallel: Optional[torch.Tensor],
        permuted_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply activation function (and optional bias) to FC1 output.

        Mirrors the ``bias_act_func`` logic inside ``TEGroupedMLP.forward`` but as a
        reusable method so that ``forward_with_dispatch_overlap`` can call it on the
        home and echo halves independently.

        Args:
            fc1_output: Output of linear_fc1, shape [T, ffn_hidden].
            bias_parallel: Per-expert bias from linear_fc1 (None when add_bias_linear=False).
            permuted_probs: Router probabilities, shape [T, 1].

        Returns:
            Activated hidden states, same shape as fc1_output (halved on last dim when GLU).
        """
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                fc1_output = fc1_output + bias_parallel
            fc1_output = self.activation_func(fc1_output)
            if permuted_probs is not None:
                original_dtype = fc1_output.dtype
                fc1_output = (fc1_output * permuted_probs).to(original_dtype)
        elif self.config.bias_activation_fusion:
            if self.activation_func == F.silu and self.config.gated_linear_unit:
                fc1_output = weighted_bias_swiglu_impl(
                    fc1_output,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                )
            elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                fc1_output = weighted_bias_quick_geglu_impl(
                    fc1_output,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                    self.config.glu_linear_offset,
                    self.config.activation_func_clamp_value,
                )
            else:
                raise ValueError(
                    "Only support fusion of swiglu and quick_gelu in TEGroupedMLP."
                )
        elif self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu:
            assert bias_parallel is None
            fc1_output = weighted_squared_relu_impl(fc1_output, permuted_probs)
        else:
            if self.config.gated_linear_unit:
                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )
                fc1_output = glu(fc1_output)
            else:
                fc1_output = self.activation_func(fc1_output)
            original_dtype = fc1_output.dtype
            fc1_output = (fc1_output * permuted_probs).to(original_dtype)
        return fc1_output

    def forward_with_dispatch_overlap(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        num_home_experts: int,
        home_expert_indices: List[int],
        echo_expert_indices: List[int],
        expert_dispatcher,
        fc1_dispatch_metadata,
        fc2_dispatch_metadata,
        a2a_stream: "torch.cuda.Stream",
        fc1_event: "torch.cuda.Event",
        fc2_event: "torch.cuda.Event",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward with home GEMM overlapping echo expert weight dispatch."""
        # ── Sanity checks ──────────────────────────────────────────────────────
        if self.offload_expert_fc1 or self.offload_moe_act:
            raise RuntimeError(
                "forward_with_dispatch_overlap does not support fine-grained activation "
            )
        if self.activation_recompute:
            raise RuntimeError(
                "forward_with_dispatch_overlap does not support moe_act activation recompute. "
            )

        # ── Preparation ────────────────────────────────────────────────────────
        # tokens_per_expert is already a CPU tensor (guaranteed by all dispatcher
        # implementations), so .cpu() is a no-op and there is no D2H sync here.
        tpe_list = tokens_per_expert.long().cpu().tolist()
        num_echo_experts = len(echo_expert_indices)
        received_num_tokens = permuted_local_hidden_states.shape[0]
        permuted_local_hidden_states = permuted_local_hidden_states.contiguous()

        if self.config.moe_received_token_capacity is not None:
            actual_total = sum(tpe_list)
            permuted_local_hidden_states = permuted_local_hidden_states[:actual_total]
            permuted_probs = permuted_probs[:actual_total]

        # Split tokens / probs into home and echo halves (zero-copy views).
        home_token_count = sum(tpe_list[:num_home_experts])
        home_tokens = permuted_local_hidden_states[:home_token_count]
        echo_tokens = permuted_local_hidden_states[home_token_count:]
        home_tpe    = tpe_list[:num_home_experts]
        echo_tpe    = tpe_list[num_home_experts:]
        home_probs  = permuted_probs[:home_token_count].unsqueeze(-1)
        echo_probs  = permuted_probs[home_token_count:].unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert self.config.moe_router_topk == 1, \
                "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            home_tokens = (home_probs * home_tokens).to(home_tokens.dtype)
            home_probs  = torch.ones_like(home_probs)
            if echo_tokens.numel() > 0:
                echo_tokens = (echo_probs * echo_tokens).to(echo_tokens.dtype)
                echo_probs  = torch.ones_like(echo_probs)

        # TE GroupedLinear requires exactly num_local_experts entries in m_splits.
        # Home-only call: real counts for home slots, 0 for echo slots (kernel skips m=0).
        # Echo-only call: 0 for home slots, real counts for echo slots.
        home_tpe_padded = home_tpe + [0] * num_echo_experts
        echo_tpe_padded = [0] * num_home_experts + echo_tpe

        # Lazy-init placeholder tensors for echo weight slots (once per module lifetime).
        # TE reads weight{i} pointers for all num_gemms slots even when m=0; echo weights
        # are freed at init time, so we keep persistent empty tensors as stand-ins.
        if echo_expert_indices and not hasattr(self, '_echo_fc1_placeholders'):
            ref_w1 = getattr(self.linear_fc1, 'weight0')
            self._echo_fc1_placeholders = [
                torch.empty(self.fc1_weight_shape, device=ref_w1.device, dtype=ref_w1.dtype)
                for _ in range(num_echo_experts)
            ]
            ref_w2 = getattr(self.linear_fc2, 'weight0')
            self._echo_fc2_placeholders = [
                torch.empty(self.fc2_weight_shape, device=ref_w2.device, dtype=ref_w2.dtype)
                for _ in range(num_echo_experts)
            ]

        # ══ FC1: dispatch on stream A  ∥  home GEMM on default stream ══════════
        with torch.cuda.nvtx.range("Get_expert_weights"):
            if echo_expert_indices:
                self.set_expert_weights("fc1", self._echo_fc1_placeholders, echo_expert_indices)
            home_fc1_weights = self.get_expert_weights("fc1", home_expert_indices)
        # Preprocess (ravel/stack/routing_map expand) on default stream — avoids
        # HBM bandwidth contention with the dispatch communication kernel.
        with torch.cuda.nvtx.range("fc1_echo_dispatch_preprocess"):
            fc1_preprocessed = expert_dispatcher.expert_dispatch_preprocess(
                fc1_dispatch_metadata, *home_fc1_weights)
        with torch.cuda.stream(a2a_stream):                          # stream A
            # Wait for default stream (token dispatch + preprocess) before launching
            # expert weight dispatch, avoiding NIC bandwidth contention.
            a2a_stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.nvtx.range("fc1_echo_dispatch"):
                dispatched_fc1_weights = expert_dispatcher.expert_dispatch_communication(
                    fc1_dispatch_metadata, fc1_preprocessed, *home_fc1_weights)
                fc1_event.record()
        with torch.cuda.nvtx.range("fc1_compute"):
            home_fc1_out, home_bias1 = self.linear_fc1(home_tokens, home_tpe_padded)  # default stream, overlaps
            home_act = self._apply_activation(home_fc1_out, home_bias1, home_probs)

            if echo_expert_indices:
                torch.cuda.current_stream().wait_event(fc1_event)
                self.set_expert_weights("fc1", list(dispatched_fc1_weights), echo_expert_indices)
                # Home slots have m=0 in echo_tpe_padded.  Replace with no-grad
                # Parameters so TE skips wgrad allocation for those slots in
                # the echo FC1 backward pass.  Must be Parameter (not plain tensor)
                # because TE's GroupedLinear stores home weights in _parameters.
                self.set_expert_weights(
                    "fc1",
                    [Parameter(w.detach(), requires_grad=False) for w in home_fc1_weights],
                    home_expert_indices,
                )
                echo_fc1_out, echo_bias1 = self.linear_fc1(echo_tokens, echo_tpe_padded)
                # Restore original Parameters so the optimizer can update them.
                self.set_expert_weights("fc1", home_fc1_weights, home_expert_indices)
                echo_act = self._apply_activation(echo_fc1_out, echo_bias1, echo_probs)

        # ══ FC2: dispatch on stream A  ∥  home GEMM on default stream ══════════
        if echo_expert_indices:
            self.set_expert_weights("fc2", self._echo_fc2_placeholders, echo_expert_indices)
        home_fc2_weights = self.get_expert_weights("fc2", home_expert_indices)
        # Preprocess on default stream before switching to dispatch stream.
        with torch.cuda.nvtx.range("fc2_echo_dispatch_preprocess"):
            fc2_preprocessed = expert_dispatcher.expert_dispatch_preprocess(
                fc2_dispatch_metadata, *home_fc2_weights)
        with torch.cuda.stream(a2a_stream):                          # stream A
            # fc2 dispatch runs after fc1 dispatch on stream A (serialized).
            # Must wait for default stream again: fc2 preprocess ran on the
            # default stream after the fc1 dispatch wait_stream, so stream A
            # needs a new wait to see the fc2 preprocess tensors.
            a2a_stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.nvtx.range("fc2_echo_dispatch"):
                dispatched_fc2_weights = expert_dispatcher.expert_dispatch_communication(
                    fc2_dispatch_metadata, fc2_preprocessed, *home_fc2_weights)
                fc2_event.record()
        with torch.cuda.nvtx.range("fc2_compute"):
            home_fc2_out, home_output_bias = self.linear_fc2(home_act, home_tpe_padded)  # default stream, overlaps

            if echo_expert_indices:
                torch.cuda.current_stream().wait_event(fc2_event)
                self.set_expert_weights("fc2", list(dispatched_fc2_weights), echo_expert_indices)
                # Same as FC1: replace home slots (m=0) with no-grad Parameters.
                self.set_expert_weights(
                    "fc2",
                    [Parameter(w.detach(), requires_grad=False) for w in home_fc2_weights],
                    home_expert_indices,
                )
                echo_fc2_out, echo_output_bias = self.linear_fc2(echo_act, echo_tpe_padded)
                # Restore original Parameters for the optimizer.
                self.set_expert_weights("fc2", home_fc2_weights, home_expert_indices)

        if echo_expert_indices:
            if home_output_bias is not None or echo_output_bias is not None:
                raise RuntimeError(
                    "forward_with_dispatch_overlap does not support add_bias_linear=True. "
                    "Disable moe_echo_expert_dispatch_overlap when add_bias_linear is enabled."
                )
            output = torch.cat([home_fc2_out, echo_fc2_out], dim=0)
        else:
            if home_output_bias is not None:
                raise RuntimeError(
                    "forward_with_dispatch_overlap does not support add_bias_linear=True. "
                    "Disable moe_echo_expert_dispatch_overlap when add_bias_linear is enabled."
                )
            output = home_fc2_out

        if self.config.moe_received_token_capacity is not None:
            pad_len = received_num_tokens - output.shape[0]
            if pad_len > 0:
                output = torch.cat(
                    [output, torch.zeros(pad_len, output.shape[1],
                                         dtype=output.dtype, device=output.device)],
                    dim=0,
                )

        return output, None

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        fc1_expert_dispatch_event=None,
        fc2_expert_dispatch_event=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        tokens_per_expert = tokens_per_expert.long().cpu().tolist()
        # When selective FP8 is active and this module was initialized without
        # FP8 (decided at init time), skip FP8 padding/unpadding since the
        # actual GEMMs will run in BF16.
        need_fp8_padding = self.config.fp8 and (
            not getattr(self.linear_fc1, "_selective_fp8_disabled", False)
        )
        if need_fp8_padding:
            actual_tokens_per_expert = tokens_per_expert
        use_hybrid_ep_dispatcher = (
            self.config.moe_flex_dispatcher_backend == "hybridep"
            and self.config.moe_token_dispatcher_type == "flex"
        )

        if self.config.moe_received_token_capacity is not None:
            # Capture the received token count before truncation; used below to pad
            # the output back to this length.
            received_num_tokens = permuted_local_hidden_states.shape[0]
            permuted_local_hidden_states = permuted_local_hidden_states.contiguous()
            permuted_local_hidden_states = permuted_local_hidden_states[: sum(tokens_per_expert)]
            permuted_probs = permuted_probs[: sum(tokens_per_expert)]

        if need_fp8_padding and not use_hybrid_ep_dispatcher:
            permuted_local_hidden_states, tokens_per_expert = self.fp8_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.fp8_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert
            )
        else:
            permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if self.offload_expert_fc1:
            permuted_local_hidden_states = fine_grained_offloading_group_start(
                permuted_local_hidden_states, name="expert_fc1"
            )
        with get_fine_grained_offloading_context(self.offload_expert_fc1):
            fc1_output, bias_parallel = self.linear_fc1(
                permuted_local_hidden_states, tokens_per_expert
            )
        if self.offload_expert_fc1:
            fc1_output, bias_parallel = fine_grained_offloading_group_commit(
                fc1_output,
                bias_parallel,
                name="expert_fc1",
                forced_released_tensors=[permuted_local_hidden_states],
            )

        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            if self.config.use_te_activation_func:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                intermediate_parallel = self.activation_func(intermediate_parallel)
                if permuted_probs is not None:
                    original_dtype = intermediate_parallel.dtype
                    intermediate_parallel = intermediate_parallel * permuted_probs
                    intermediate_parallel = intermediate_parallel.to(original_dtype)
            elif self.config.bias_activation_fusion:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                        self.config.activation_func_clamp_value,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu in TEGroupedMLP."
                    )
            elif (
                self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu
            ):
                assert bias_parallel is None
                intermediate_parallel = weighted_squared_relu_impl(
                    intermediate_parallel, permuted_probs
                )
            else:
                if self.config.gated_linear_unit:

                    def glu(x):
                        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                        if (val := self.config.activation_func_clamp_value) is not None:
                            x_glu = x_glu.clamp(min=None, max=val)
                            x_linear = x_linear.clamp(min=-val, max=val)
                        return self.config.activation_func(x_glu) * (
                            x_linear + self.config.glu_linear_offset
                        )

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
            return intermediate_parallel

        if self.offload_moe_act:
            fc1_output = fine_grained_offloading_group_start(fc1_output, name="moe_act")

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with get_fine_grained_offloading_context(self.offload_moe_act):
                bias_act_output = self.activation_checkpoint.checkpoint(
                    bias_act_func, fc1_output, bias_parallel, permuted_probs
                )
        else:
            with get_fine_grained_offloading_context(self.offload_moe_act):
                bias_act_output = bias_act_func(fc1_output, bias_parallel, permuted_probs)

        output, output_bias = self.linear_fc2(bias_act_output, tokens_per_expert)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)
        if self.offload_moe_act:
            (output,) = fine_grained_offloading_group_commit(
                output, name="moe_act", forced_released_tensors=[fc1_output]
            )
            
        # upad and concat the output
        if need_fp8_padding and not use_hybrid_ep_dispatcher:
            output = self.fp8_unpadding(output, actual_tokens_per_expert)

        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)
        output_bias = None
        if self.config.moe_received_token_capacity is not None:
            output = torch.cat(
                [
                    output,
                    torch.zeros(
                        received_num_tokens - output.shape[0],
                        output.shape[1],
                        dtype=output.dtype,
                        device=output.device,
                    ),
                ],
                dim=0,
            )

        return output, output_bias

    @expert_dist_ckpt_decorator
    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Maps local expert to global experts.
        The sharded state dict is interchangable with SequentialMLP's.
        """
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        num_local_experts = self.config.num_moe_experts // self.ep_group.size()
        for name, module in self._modules.items():
            sub_sd = sharded_state_dict_default(module, f'{name}.', sharded_offsets, metadata)
            if name == 'linear_fc1' and self.config.gated_linear_unit:
                num_global_experts = self.ep_group.size() * num_local_experts
                local_expert_indices_offset = self.ep_group.rank() * num_local_experts
                ep_axis = len(sharded_offsets)
                for i in range(num_local_experts):
                    if singleton_local_shards:
                        new_sharded_offsets = sharded_offsets
                    else:
                        new_sharded_offsets = (
                            *sharded_offsets,
                            (ep_axis, local_expert_indices_offset + i, num_global_experts),
                        )
                    for k in (f'{name}.weight{i}', f'{name}.bias{i}'):
                        if k in sub_sd:
                            sub_sd[k] = apply_swiglu_sharded_factory(
                                sub_sd[k], new_sharded_offsets, singleton_local_shards
                            )
            if singleton_local_shards:
                replace_prefix_for_sharding(sub_sd, '', f'{prefix}experts.')
            else:
                # Add prefix here to match sequential's keys
                replace_prefix_for_sharding(sub_sd, f'{name}.', f'{prefix}experts.{name}.')
            sharded_state_dict.update({f"{prefix}{k}": v for k, v in sub_sd.items()})
        return sharded_state_dict

    def backward_dw(self):
        """Performs backward pass for weight gradients in TEGroupedMLP.

        This method executes the backward pass for weight gradients by calling
        backward_dw() on the linear layers in reverse order (fc2 followed by fc1).
        If an error occurs during execution, it is caught and re-raised with a
        descriptive message.
        """
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):

        if config.moe_ffn_hidden_size == config.ffn_hidden_size:
            super().__init__(config=config)
        else:
            # Local SequentialMLP can still be used here by overriding the ffn_hidden_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.ffn_hidden_size = config.moe_ffn_hidden_size
            super().__init__(config=sequential_mlp_config)

        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_dp_group as data parallel group in this module.
        # TODO (Hepteract): expt_dp wont be needed here once distributed checkpoint is refactored
        self.dp_group = pg_collection.expt_dp

        for _ in range(self.num_local_experts):
            expert = MLP(
                self.config,
                submodules,
                ffn_hidden_size=self.config.moe_ffn_hidden_size,
                is_expert=True,
                tp_group=pg_collection.expt_tp,
            )
            self.local_experts.append(expert)

        self.fc1_weight_shape = self.local_experts[0].linear_fc1.weight.shape
        self.fc2_weight_shape = self.local_experts[0].linear_fc2.weight.shape

    def _pad_tensor_for_fp8(self, hidden, probs):
        """Padding tensor shape to multiples of 16/32."""
        actual_num_tokens = hidden.shape[0]
        divisor = get_fp8_align_size(self.config.fp8_recipe)
        padded_num_tokens = ceil(actual_num_tokens / divisor) * divisor - actual_num_tokens
        if padded_num_tokens > 0:
            pad_tensor = torch.zeros(
                padded_num_tokens, hidden.shape[1], dtype=hidden.dtype, device=hidden.device
            )
            hidden = torch.cat((hidden, pad_tensor), dim=0)
            pad_probs = torch.zeros(padded_num_tokens, dtype=probs.dtype, device=probs.device)
            probs = torch.cat((probs, pad_probs), dim=0)
        return hidden, probs

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        fc1_expert_dispatch_event=None,
        fc2_expert_dispatch_event=None,
    ):
        """Forward step of the SequentialMLP."""

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if self.num_local_experts == 1:
            if self.config.fp8:
                hidden, probs = self._pad_tensor_for_fp8(
                    permuted_local_hidden_states, permuted_probs
                )
                output, output_bias = self.local_experts[0](hidden, probs)
                output = output[: permuted_local_hidden_states.shape[0]]
            else:
                output, output_bias = self.local_experts[0](
                    permuted_local_hidden_states, permuted_probs
                )

            return output, output_bias
        else:
            tokens_per_expert = tokens_per_expert.tolist()
            tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)
            probs_list = torch.split(permuted_probs, tokens_per_expert)

            output_local_list = []

            for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
                if self.config.fp8:
                    hidden, probs = self._pad_tensor_for_fp8(tokens, probs)
                    output, output_bias = expert(hidden, probs)
                    output = output[: tokens.shape[0]]
                else:
                    output, output_bias = expert(tokens, probs)
                output_local_list.append(output)

            output_local = torch.cat(output_local_list, dim=0)
            output_bias_local = None
            # Note: if bias is enabled on experts, it is already added to the output at this point
            return output_local, output_bias_local

    def backward_dw(self):
        """Backward pass for weight gradients in SequentialMLP."""
        for expert in self.local_experts:
            expert.backward_dw()

    @expert_dist_ckpt_decorator
    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Maps local expert to global experts."""
        sharded_state_dict = {}
        num_global_experts = self.config.num_moe_experts
        num_local_experts = num_global_experts // self.ep_group.size()
        local_expert_indices_offset = self.ep_group.rank() * num_local_experts

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)

        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            if singleton_local_shards:
                expert_sharded_prefix = f'{prefix}experts.{expert_global_idx}.'
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_prefix = f'{prefix}experts.'
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'

                sh_ten.replica_id = (*replica_id[:2], self.dp_group.rank())

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict

    def free_expert_parameters(self, expert_indices: List[int]):
        """Free the parameters of the experts."""
        for expert_idx in expert_indices:
            expert = self.local_experts[expert_idx]
            for layer_name in ['linear_fc1', 'linear_fc2']:
                layer = getattr(expert, layer_name)
                # Clear all parameters in the layer
                for param_name in list(layer._parameters.keys()):
                    if getattr(layer, param_name) is not None:
                        delattr(layer, param_name)
                        layer._parameters.pop(param_name, None)


    def get_expert_weights(
        self, module, expert_indices: List[int]
    ) -> List[torch.Tensor]:
        """Get the weights of the experts."""
        assert module in ["fc1", "fc2"], f"Invalid module: {module}"
        expert_layer_name = "linear_fc1" if module == "fc1" else "linear_fc2"
        weight_list = []
        for i in expert_indices:
            weight = getattr(self.local_experts[i], expert_layer_name).weight
            weight_list.append(weight)
        return weight_list

    def set_expert_weights(
        self,
        module,
        expert_weights: List[torch.Tensor],
        expert_indices: List[int],
    ):
        """Set the weights of the experts."""
        expert_layer_name = "linear_fc1" if module == "fc1" else "linear_fc2"
        for i, expert_index in enumerate(expert_indices):
            if expert_weights[i].numel() == 0:
                getattr(self.local_experts[expert_index], expert_layer_name).weight = DummyFunction.apply(
                    expert_weights[i], 
                    self.fc1_weight_shape if module == "fc1" else self.fc2_weight_shape
                )
            else:
                getattr(self.local_experts[expert_index], expert_layer_name).weight = expert_weights[i]
