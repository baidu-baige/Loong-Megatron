# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import partial
from typing import Optional, Protocol

import torch

from megatron.core import tensor_parallel, utils
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.inference.moe import InferenceGroupedGemmBackend
from megatron.core.inference.moe.flashinfer_mxfp8 import require_flashinfer_routed_mxfp8
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection, resolve_gtp_remat_group
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    GLOBAL_MOE_ROUTING_TRACKER,
    MoECudaGraphPartialCaptureSignal,
    MoECudaGraphTensorStore,
    get_default_pg_collection,
    initialize_cuda_monitoring,
    maybe_skip_or_early_return_by_cudagraph,
    monitor_max_dispatcher_tokens,
    monitor_max_memory_usage,
    write_monitor_data_to_file,
)
from megatron.core.transformer.moe.offloading_planner import gen_offloading_plan
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEElasticExpertDispatcher,
    MoEFlexTokenDispatcher,
    MoESyncFreeElasticExpertDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.moe.token_dispatcher_inference import (
    NCCLAllGatherDispatcher,
    NVLSAllGatherVDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import internal_api, nvtx_range_pop, nvtx_range_push

try:
    import flashinfer  # pylint: disable=unused-import

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

if HAVE_FLASHINFER:
    try:
        import flashinfer_cubin  # pylint: disable=unused-import
        import flashinfer_jit_cache  # pylint: disable=unused-import

        HAVE_FLASHINFER_CUBIN_AND_JIT_CACHE = True
    except ImportError:
        HAVE_FLASHINFER_CUBIN_AND_JIT_CACHE = False

try:
    import triton  # pylint: disable=unused-import

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TELinear, TENorm, te_checkpoint
else:
    TELinear, TENorm, te_checkpoint = None, None, None


class ExpertsInterface(Protocol):
    """Interface for the experts used in an MoELayer."""

    def forward(
        self,
        dispatched_input: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        /,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the experts layer."""
        ...

    def backward_dw(self) -> None:
        """Backward pass to compute weight gradients for the experts."""
        ...


class ExpertsBuilder(Protocol):
    """Protocol for building the experts used in an MoELayer."""

    def __call__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        /,
        *,
        pg_collection: ProcessGroupCollection | None,
        name: str | None = None,
    ) -> ExpertsInterface: ...


class SharedExpertsInterface(Protocol):
    """Interface for the shared experts used in an MoELayer."""

    def forward(self, hidden_states: torch.Tensor, /) -> torch.Tensor:
        """Forward pass of the shared experts."""
        ...

    def backward_dw(self) -> None:
        """Backward pass to compute weight gradients for the shared experts."""
        ...


class SharedExpertsBuilder(Protocol):
    """Protocol for building the shared experts used in an MoELayer."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None,
        gate: bool,
        name: str | None = None,
    ) -> SharedExpertsInterface: ...


class RouterInterface(Protocol):
    """Interface for the router used in an MoELayer."""

    def forward(
        self,
        input: torch.Tensor,
        /,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the router.

        Returns:
            A tuple of (probabilities, routing_map).
        """
        ...

    def set_layer_number(self, layer_number: int) -> None:
        """Set the layer number for the router.

        Called from transformer_layer during initialization.
        """
        ...


class RouterBuilder(Protocol):
    """Protocol for building a Router."""

    def __call__(
        self,
        /,
        *,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None,
        is_mtp_layer: bool = False,
        hash_moe_layer_threshold: int | None = None,
    ) -> RouterInterface: ...


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: ExpertsBuilder
    shared_experts: SharedExpertsBuilder | None = None
    router: RouterBuilder = TopKRouter


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        if self.config.moe_enable_echo:
            self.num_local_total_experts = (
                self.config.num_moe_experts + self.config.moe_num_echo_experts
            ) // ep_size
        else:
            self.num_local_total_experts = self.config.num_moe_experts // ep_size
        self.num_home_experts = self.config.num_moe_experts // ep_size
        # Preserve the pre-ECHO meaning of num_local_experts: home experts only.
        self.num_local_experts = self.num_home_experts
        local_expert_indices_offset = ep_rank * self.num_local_total_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_total_experts)
        ]
        max_expert_id = self.config.num_moe_experts + (
            self.config.moe_num_echo_experts if self.config.moe_enable_echo else 0
        )
        assert all(map(lambda x: x < max_expert_id, self.local_expert_indices))
        self.router: RouterInterface = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class FlushPendingGradAccum(torch.autograd.Function):
    """Identity forward op that flushes deferred ECHO weight gradients in backward."""

    @staticmethod
    def forward(ctx, x):
        """Return the input unchanged."""
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """Add pending echo expert gradients to their home expert main_grad."""
        pending = getattr(MoELayer, 'pending_expert_wgrads', None)
        if pending:
            with torch.cuda.stream(MoELayer.moe_a2a_stream):
                MoELayer.moe_a2a_stream.wait_event(MoELayer.grad_combine_event)
                MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                for weight, wgrad in pending:
                    weight.main_grad.add_(wgrad)
            pending.clear()
        return grad_output


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        hash_moe_layer_threshold: Optional[int] = None,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        self.submodules = not_none(submodules)
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        # If using mcore cudagraphs, recompute is handled by transformer_layer.MoETransformerLayer
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective'
            and "moe" in config.recompute_modules
            and config.cuda_graph_impl != 'local'
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )
        self.routed_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "routed_experts" in config.recompute_modules
            and "moe" not in config.recompute_modules
        )

        self.tp_group = pg_collection.tp

        # Initialize router.
        self.router = self.submodules.router(
            config=self.config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
        )
        if layer_number is not None:
            self.router.set_layer_number(layer_number)
        self.tp_group = pg_collection.tp

        # Initialize latent projections.
        if self.config.moe_latent_size:
            assert HAVE_TE, "TransformerEngine is required for MoE latent projections."
            if self.config.transformer_impl == "inference_optimized":
                from megatron.core.tensor_parallel.inference_layers import InferenceLinear

                linear_cls = InferenceLinear
            else:
                linear_cls = TELinear
            gtp_remat_group = (
                resolve_gtp_remat_group(pg_collection, is_expert=False)
                if "moe_latent_proj" in self.config.gtp_remat_opt_in_modules
                else None
            )
            linear_gtp_kwargs = {}
            if linear_cls is TELinear:
                linear_gtp_kwargs = {
                    "gtp_remat_group": gtp_remat_group,
                    "gtp_replica_group": pg_collection.dp_cp,
                }
            self.fc1_latent_proj = linear_cls(
                self.config.hidden_size,
                self.config.moe_latent_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
                name=(name + ".fc1_latent_proj") if name is not None else None,
                **linear_gtp_kwargs,
            )
            if self.config.moe_use_norm_before_up_proj:
                self.fc2_norm = TENorm(
                    config=self.config,
                    hidden_size=self.config.moe_latent_size,
                    eps=self.config.layernorm_epsilon,
                )
            self.fc2_latent_proj = linear_cls(
                self.config.moe_latent_size,
                self.config.hidden_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
                name=(name + ".fc2_latent_proj") if name is not None else None,
                **linear_gtp_kwargs,
            )
            if linear_cls is TELinear:
                # The duplicated operation has no TP execution group. TELinear uses
                # `_tp_group` only to encode the owning TP replica coordinate in checkpoints.
                self.fc1_latent_proj._tp_group = pg_collection.tp
                self.fc2_latent_proj._tp_group = pg_collection.tp

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_total_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_total_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_total_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        if config.moe_enable_echo:
            if config.moe_echo_expert_dispatcher_type == "hybridep":
                self.expert_dispatcher = MoESyncFreeElasticExpertDispatcher(
                    config=config, pg_collection=pg_collection
                )
            elif config.moe_echo_expert_dispatcher_type == "alltoall":
                self.expert_dispatcher = MoEElasticExpertDispatcher(
                    config=config, pg_collection=pg_collection
                )
            else:
                raise ValueError(
                    f"Unsupported expert dispatcher type: {config.moe_echo_expert_dispatcher_type}"
                )
            num_echo_local_experts = config.moe_num_echo_experts // utils.get_pg_size(
                self.ep_group
            )
            echo_config = replace(
                self.config,
                gradient_accumulation_fusion=False,
                moe_enable_echo=False,
            )
            self.experts = self.submodules.experts(
                num_echo_local_experts + self.num_home_experts,
                echo_config,
                pg_collection=pg_collection,
                name=(name + ".experts") if name is not None else None,
            )
            self.echo_expert_indices = list(
                range(self.num_home_experts, num_echo_local_experts + self.num_home_experts)
            )
            self.home_expert_indices = list(range(self.num_home_experts))
            self.experts.free_expert_parameters(self.echo_expert_indices)
        else:
            self.experts = self.submodules.experts(
                self.num_local_experts,
                self.config,
                pg_collection=pg_collection,
                name=(name + ".experts") if name is not None else None,
            )

        # Initialize shared experts
        if self.use_shared_expert:
            assert (
                self.submodules.shared_experts is not None
            ), "Shared experts builder is not provided in the module spec."
            self.shared_experts = self.submodules.shared_experts(
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
                name=(name + ".shared_experts") if name is not None else None,
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

        if config.enable_moe_mem_monitor:
            initialize_cuda_monitoring(
                tp_rank=utils.get_pg_rank(pg_collection.tp),
                pp_rank=utils.get_pg_rank(pg_collection.pp),
                ep_rank=utils.get_pg_rank(pg_collection.ep),
                mem_monitor_force_print_token_threshold=(
                    config.moe_mem_monitor_force_print_token_threshold
                ),
            )

        # Inference-optimized mode setup
        if config.transformer_impl == "inference_optimized":
            if config.inference_grouped_gemm_backend == InferenceGroupedGemmBackend.FLASHINFER:
                assert HAVE_FLASHINFER, (
                    "inference_grouped_gemm_backend='flashinfer' requires flashinfer-python. "
                    "Install flashinfer-python or set "
                    "inference_grouped_gemm_backend to 'torch' or 'vllm'."
                )
                fp8_recipe = getattr(config.fp8_recipe, "value", config.fp8_recipe)
                if config.fp8 and fp8_recipe == "mxfp8":
                    require_flashinfer_routed_mxfp8()

                # Verify that pre-compiled FlashInfer CUTLASS kernels are available
                # when using the FlashInfer backend. The flashinfer-jit-cache package
                # must be installed ahead of time to avoid a multi-minute JIT
                # compilation step at runtime.
                from megatron.core.inference.utils import check_flashinfer_jit_cache_installed

                check_flashinfer_jit_cache_installed()
            elif config.inference_grouped_gemm_backend == InferenceGroupedGemmBackend.TORCH:
                assert hasattr(torch.nn.functional, 'grouped_mm') or hasattr(
                    torch, '_grouped_mm'
                ), (
                    "inference_grouped_gemm_backend='torch' requires "
                    "torch.nn.functional.grouped_mm (> torch 2.10) or torch._grouped_mm (<= 2.10)."
                )
            elif config.inference_grouped_gemm_backend == InferenceGroupedGemmBackend.VLLM:
                assert HAVE_TRITON, (
                    "inference_grouped_gemm_backend='vllm' requires Triton. "
                    "Install triton (pip install triton)."
                )
            self._setup_inference_mode(pg_collection)

        # Cudagraph tensor store for resuming the forward pass from the end of the cudagraph.
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()
        self.fwd_execution_map = ["route", "expert_compute", "postprocess"]

        if self.config.moe_echo_expert_dispatch_overlap and not hasattr(
            MoELayer, 'moe_a2a_stream'
        ):
            MoELayer.moe_a2a_stream = torch.cuda.Stream()
            MoELayer.fc1_expert_dispatch_event = torch.cuda.Event()
            MoELayer.fc2_expert_dispatch_event = torch.cuda.Event()
            MoELayer.token_dispatch_event = torch.cuda.Event()
            MoELayer.token_combine_event = torch.cuda.Event()
            MoELayer.grad_combine_event = torch.cuda.Event()
            MoELayer.dispatch_bwd_event = torch.cuda.Event()
            MoELayer.combine_bwd_event = torch.cuda.Event()
            MoELayer.pending_expert_wgrads = []

        # Setup events and streams for delayed wgrad computation.
        self.setup_delayed_wgrad_for_dispatch_backward_overlap()

    def _setup_inference_mode(self, pg_collection):
        """Set up inference-optimized token dispatcher.

        Called from __init__ when config.transformer_impl == "inference_optimized".
        Stores the training dispatcher and creates the inference dispatcher selected
        by config.inference_moe_token_dispatcher_type ('nccl' or 'nvls').
        The active dispatcher is selected at the start of `forward` based on
        `InferenceMode.is_active()`.
        """
        dispatcher_type = self.config.inference_moe_token_dispatcher_type
        dispatcher_cls = (
            NVLSAllGatherVDispatcher if dispatcher_type == 'nvls' else NCCLAllGatherDispatcher
        )

        self._training_token_dispatcher = self.token_dispatcher
        self._inference_token_dispatcher = dispatcher_cls(
            self.num_local_experts,
            self.local_expert_indices,
            config=self.config,
            pg_collection=pg_collection,
        )

        # Wire shared-expert overlap into the inference dispatcher (NVLS only).
        # The dispatcher launches the shared-expert forward on SharedExpertMLP.stream
        # concurrently with AGV+experts+RSV and adds it back in combine_postprocess.
        if (
            dispatcher_type == 'nvls'
            and self.use_shared_expert
            and self.config.moe_shared_expert_overlap
        ):
            self._inference_token_dispatcher.set_shared_experts(self.shared_experts)
            # With MoE latent projections, the shared expert must run on the full
            # hidden_states (pre-latent) and its output added post-fc2_latent_proj.
            # The dispatcher only sees latent-dim tensors, so we move the launch+add
            # into preprocess/postprocess on the layer and tell the dispatcher to
            # skip its own internal launch+add.
            if self.config.moe_latent_size:
                self._inference_token_dispatcher._external_shared_expert_launch = True
        # Inference only: side-stream shared-expert output for latent-MoE + NVLS overlap
        # (preprocess launches on SharedExpertMLP.stream; postprocess joins+adds).
        self._latent_shared_expert_output: Optional[torch.Tensor] = None

    def select_token_dispatcher(self) -> None:
        """Select the inference token dispatcher if active, otherwise use training dispatcher."""
        if not hasattr(self, "_inference_token_dispatcher"):
            return
        if InferenceMode.is_active():
            self.token_dispatcher = self._inference_token_dispatcher
            self.shared_expert_overlap = self._inference_token_dispatcher.shared_experts is not None
        else:
            self.token_dispatcher = self._training_token_dispatcher
            self.shared_expert_overlap = self.config.moe_shared_expert_overlap

    def setup_delayed_wgrad_for_dispatch_backward_overlap(self):
        """Initializes CUDA events and streams for overlapping expert
        weight gradient computation with dispatch backward.
        """
        self._delayed_wgrad_event: Optional[torch.cuda.Event] = None
        self._delayed_wgrad_stream: Optional[torch.cuda.Stream] = None
        if self.config.overlap_dispatch_backward_with_experts_wgrad:
            self._delayed_wgrad_event = torch.cuda.Event()
            self._delayed_wgrad_stream = torch.cuda.Stream(device="cuda")

    @maybe_skip_or_early_return_by_cudagraph("route")
    def route(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Compute token routing for preprocessing.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. The input padding mask is batch-first;
        routing consumes it sequence-first to align with ``hidden_states``.
        """
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()
        probs, routing_map = apply_module(self.router)(
            hidden_states, padding_mask, input_ids=input_ids, packed_seq_params=packed_seq_params
        )
        return probs, routing_map

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Preprocess token routing for dispatch.

        This method preprocesses the hidden states and routing probabilities for the token
        dispatcher. The optional padding mask is the batch-first mask given to ``route``; it is
        passed through untouched and the dispatcher decides whether its backend excludes padded
        rows.
        """
        # Latent-MoE + NVLS-inference shared-expert overlap: launch the shared
        # expert on its side stream BEFORE fc1_latent_proj so it sees the full
        # hidden_states. The corresponding join+add runs in postprocess after
        # fc2_latent_proj. Skipped on the training / NCCL paths.
        if (
            self.config.moe_latent_size
            and self.shared_expert_overlap
            and isinstance(self.token_dispatcher, NVLSAllGatherVDispatcher)
        ):
            stream = SharedExpertMLP.stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._latent_shared_expert_output = apply_module(self.shared_experts)(hidden_states)
        elif self.config.moe_latent_size:
            if self.shared_expert_overlap:
                if self.training:
                    raise AssertionError(
                        "Shared expert overlap with MoE latent projections is not supported "
                        "during training. Disable moe_shared_expert_overlap."
                    )
                raise AssertionError(
                    "Shared expert overlap with MoE latent projections requires the NVLS "
                    "inference dispatcher. Either disable moe_shared_expert_overlap or set "
                    "inference_moe_token_dispatcher_type='nvls'."
                )
        # Project the hidden_states from hidden dimension down to latent dimension.
        if self.config.moe_latent_size:
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        if isinstance(self.token_dispatcher, MoEFlexTokenDispatcher):
            # Only the flex dispatcher takes the mask (dropless HybridEP excludes padded rows).
            hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
                hidden_states, routing_map, probs, padding_mask=padding_mask
            )
        else:
            hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
                hidden_states, routing_map, probs
            )
        return hidden_states, probs

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        if self.config.overlap_dispatch_backward_with_experts_wgrad:
            hidden_states = _RegisterDelayedWgradForExperts.apply(self, hidden_states)
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    @maybe_skip_or_early_return_by_cudagraph("shared_experts_compute")
    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8 or self.config.fp4:
                    shared_expert_output = te_checkpoint(
                        apply_module(self.shared_experts),
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        self.tp_group,
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        apply_module(self.shared_experts), False, hidden_states
                    )
            else:
                shared_expert_output = apply_module(self.shared_experts)(hidden_states)

        return shared_expert_output

    @internal_api
    def _dispatch_postprocess_for_experts(
        self, hidden_states: torch.Tensor, probs: torch.Tensor
    ):
        """Prepare permuted expert inputs from dispatched tokens."""
        if self.config.overlap_dispatch_backward_with_experts_wgrad:
            hidden_states = _RecordExpertDgradCompletion.apply(
                self._delayed_wgrad_event, hidden_states
            )
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        if self.config.enable_moe_mem_monitor:
            monitor_max_memory_usage()
            monitor_max_dispatcher_tokens(tokens_per_expert)
            if self.config.moe_mem_monitor_log is not None:
                write_monitor_data_to_file(
                    self.config.moe_mem_monitor_log,
                    self.config.print_moe_mem_monitor_interval,
                )
        return dispatched_input, tokens_per_expert, permuted_probs

    def _run_experts(
        self,
        dispatched_input: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Run only the routed expert computation on already-permuted tokens."""
        if hasattr(self, "_inference_token_dispatcher") and InferenceMode.is_active():
            routing_map = self.token_dispatcher.routing_map
            expert_output, mlp_bias = apply_module(self.experts)(
                dispatched_input, tokens_per_expert, permuted_probs, routing_map=routing_map
            )
        else:
            # NCCL-EP zero-copy: experts write fc2 output and fc1 dgrad straight into the combine /
            # dispatch symm buffers. Passed only when set (non-TEGroupedMLP experts don't accept
            # these kwargs).
            output_buffer, grad_input_buffer = self.token_dispatcher.get_expert_zero_copy_buffers()
            expert_kwargs = {}
            if output_buffer is not None:
                expert_kwargs["output_buffer"] = output_buffer
            if grad_input_buffer is not None:
                expert_kwargs["grad_input_buffer"] = grad_input_buffer
            expert_output, mlp_bias = apply_module(self.experts)(
                dispatched_input, tokens_per_expert, permuted_probs, **expert_kwargs
            )
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        return expert_output, mlp_bias

    @internal_api
    def routed_experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self._dispatch_postprocess_for_experts(hidden_states, probs)
        )
        expert_output, mlp_bias = self._run_experts(
            dispatched_input, tokens_per_expert, permuted_probs
        )
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def combine(self, output: torch.Tensor):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication).
        """
        output = self.token_dispatcher.token_combine(output)
        return output

    def postprocess(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Project the output back from latent dimension to hidden dimension after combine
        in latent dimension if needed. Combine expert output with shared_experts if needed.

        _latent_shared_expert_output is inference-only (latent-MoE + NVLS dispatcher with
        shared-expert overlap). It is populated in preprocess and joined here, after
        fc2_latent_proj, so the dimensions match the full hidden dim."""

        output = self.token_dispatcher.combine_postprocess(output)
        if self.config.moe_latent_size:
            if self.config.moe_use_norm_before_up_proj:
                output = apply_module(self.fc2_norm)(output)
            output, _ = self.fc2_latent_proj(output)

        if shared_expert_output is not None:
            output = output + shared_expert_output
        elif (
            isinstance(self.token_dispatcher, NVLSAllGatherVDispatcher)
            and self._latent_shared_expert_output is not None
        ):
            # This codepath is for inference-only shared-expert overlap of latent MoEs.
            # Must happen post-fc2_latent_proj so dimensions match.
            torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)
            output = output + self._latent_shared_expert_output
            self._latent_shared_expert_output = None
        return output

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """This method is a combined method of route and preprocess. Deprecated."""

        probs, routing_map = self.route(hidden_states)
        hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map)
        return hidden_states, probs, residual

    def _echo_route(self, hidden_states: torch.Tensor):
        """Route tokens and build the ECHO overflow-assignment plan."""
        with torch.cuda.nvtx.range("router"):
            probs, routing_map = self.router(hidden_states)

        with torch.cuda.nvtx.range("rerouting"):
            tokens_per_expert_current_ep_rank = routing_map.sum(dim=0)
            if self.config.moe_echo_expert_dispatch_overlap and hasattr(MoELayer, "moe_a2a_stream"):
                torch.cuda.current_stream().wait_stream(MoELayer.moe_a2a_stream)
            tokens_per_expert_per_ep_rank = gather_from_sequence_parallel_region(
                tokens_per_expert_current_ep_rank, group=self.ep_group
            ).reshape(self.ep_group.size(), self.config.num_moe_experts)

            # Generate offloading plan to redistribute tokens to echo experts
            num_spare_experts_per_ep_rank = self.config.moe_num_echo_experts // self.ep_group.size()
            if self.config.moe_echo_algorithm == "greedy":
                if num_spare_experts_per_ep_rank == 1:
                    assignment_algorithm = "approx_bin_packing"
                else:
                    assignment_algorithm = "one_shot_greedy"
            else:
                assignment_algorithm = "sinkhorn"
            rerouting_map, rerouted_probs, expert_offloading_map = gen_offloading_plan(
                routing_map,
                probs,
                tokens_per_expert_per_ep_rank,
                self.ep_group.rank(),
                num_ep_ranks=self.ep_group.size(),
                num_spare_experts_per_ep_rank=num_spare_experts_per_ep_rank,
                assignment_algorithm=assignment_algorithm,
            )
            if self.config.moe_echo_dump_dir is not None:
                GLOBAL_MOE_ROUTING_TRACKER.set_rank_info(self.ep_group)
                num_offloaded_experts_per_rank = expert_offloading_map.reshape(self.ep_group.size(), -1).sum(dim=-1)
                GLOBAL_MOE_ROUTING_TRACKER.add_data(
                    self.layer_number,
                    "num_offloaded_experts_per_rank",
                    num_offloaded_experts_per_rank,
                )
                rerouted_tokens_per_rank = rerouting_map.sum(dim=0).reshape(self.ep_group.size(), -1).sum(dim=-1)
                torch.distributed.all_reduce(
                    rerouted_tokens_per_rank,
                    group=self.ep_group,
                    op=torch.distributed.ReduceOp.SUM,
                )
                GLOBAL_MOE_ROUTING_TRACKER.add_data(
                    self.layer_number,
                    "rerouted_tokens_per_rank",
                    rerouted_tokens_per_rank,
                )

            if self.config.moe_echo_log_file is not None:
                self._echo_log(
                    tokens_per_expert_per_ep_rank,
                    rerouting_map,
                    expert_offloading_map,
                )
                GLOBAL_MOE_ROUTING_TRACKER.add_data(
                    self.layer_number,
                    "tokens_per_expert_per_ep_rank",
                    tokens_per_expert_per_ep_rank,
                )
                tokens_per_rank = tokens_per_expert_per_ep_rank.reshape(
                    self.ep_group.size(),
                    self.ep_group.size(),
                    -1,
                ).sum(dim=[0, 2])
                GLOBAL_MOE_ROUTING_TRACKER.add_data(
                    self.layer_number, "tokens_per_rank", tokens_per_rank
                )
        return rerouting_map, rerouted_probs, expert_offloading_map

    def _echo_preprocess(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        dispatch_stream: torch.cuda.Stream | None = None,
    ):
        """Preprocess rerouted tokens for the token dispatcher."""

        def preprocess():
            with torch.cuda.nvtx.range("token_dispatch_preprocess"):
                return self.preprocess(hidden_states, probs, routing_map)

        if dispatch_stream is None:
            return preprocess()

        with torch.cuda.stream(dispatch_stream):
            dispatch_stream.wait_stream(torch.cuda.current_stream())
            return preprocess()

    def _prepare_echo_expert_dispatch(self, expert_offloading_map: torch.Tensor):
        """Prepare ECHO expert-weight metadata and optionally dispatch the weights."""
        fc1_expert_dispatch_metadata = self.expert_dispatcher.preprocess(expert_offloading_map)
        fc2_expert_dispatch_metadata = self.expert_dispatcher.preprocess(expert_offloading_map)
        fc1_expert_dispatch_metadata.buffer_idx = 0
        fc2_expert_dispatch_metadata.buffer_idx = 1

        if not self.config.moe_echo_expert_dispatch_overlap:
            with torch.cuda.nvtx.range("expert_dispatch"):
                fc1_expert_weights = self.experts.get_expert_weights(
                    "fc1", self.home_expert_indices
                )
                dispatched_fc1_weights = self.expert_dispatcher.expert_dispatch(
                    fc1_expert_dispatch_metadata, *fc1_expert_weights
                )
                self.experts.set_expert_weights(
                    "fc1", dispatched_fc1_weights, self.echo_expert_indices
                )

                fc2_expert_weights = self.experts.get_expert_weights(
                    "fc2", self.home_expert_indices
                )
                dispatched_fc2_weights = self.expert_dispatcher.expert_dispatch(
                    fc2_expert_dispatch_metadata, *fc2_expert_weights
                )
                self.experts.set_expert_weights(
                    "fc2", dispatched_fc2_weights, self.echo_expert_indices
                )
        return fc1_expert_dispatch_metadata, fc2_expert_dispatch_metadata

    def _run_echo_experts(
        self,
        dispatched_input: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        fc1_expert_dispatch_metadata,
        fc2_expert_dispatch_metadata,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run ECHO experts on already-permuted tokens."""
        if self.config.moe_echo_expert_dispatch_overlap:
            expert_output, mlp_bias = self.experts.forward_with_dispatch_overlap(
                dispatched_input,
                tokens_per_expert,
                permuted_probs,
                len(self.home_expert_indices),
                self.home_expert_indices,
                self.echo_expert_indices,
                self.expert_dispatcher,
                fc1_expert_dispatch_metadata,
                fc2_expert_dispatch_metadata,
                MoELayer.moe_a2a_stream,
                MoELayer.fc1_expert_dispatch_event,
                MoELayer.fc2_expert_dispatch_event,
            )
        else:
            expert_output, mlp_bias = apply_module(self.experts)(
                dispatched_input, tokens_per_expert, permuted_probs
            )
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        return expert_output, mlp_bias

    def _echo_experts_from_permuted(
        self,
        dispatched_input: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        fc1_expert_dispatch_metadata,
        fc2_expert_dispatch_metadata,
    ) -> torch.Tensor:
        """Run ECHO experts and prepare their output for a single token combine."""
        expert_output, _ = self._run_echo_experts(
            dispatched_input,
            tokens_per_expert,
            permuted_probs,
            fc1_expert_dispatch_metadata,
            fc2_expert_dispatch_metadata,
        )
        return self.token_dispatcher.combine_preprocess(expert_output)

    def echo_routed_experts_compute(
        self,
        dispatched_input: torch.Tensor,
        dispatched_probs: torch.Tensor,
        fc1_expert_dispatch_metadata,
        fc2_expert_dispatch_metadata,
    ) -> torch.Tensor:
        """Post-process ECHO tokens, run experts, and prepare one combine input."""
        permuted_tokens, tokens_per_expert, permuted_probs = (
            self._dispatch_postprocess_for_experts(dispatched_input, dispatched_probs)
        )
        return self._echo_experts_from_permuted(
            permuted_tokens,
            tokens_per_expert,
            permuted_probs,
            fc1_expert_dispatch_metadata,
            fc2_expert_dispatch_metadata,
        )

    def echo_forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer with echo experts.

        This implements the echo expert logic where overflow tokens are offloaded
        to spare/echo experts for better load balancing.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        residual = hidden_states
        rerouting_map, rerouted_probs, expert_offloading_map = self._echo_route(hidden_states)
        fc1_expert_dispatch_metadata, fc2_expert_dispatch_metadata = (
            self._prepare_echo_expert_dispatch(expert_offloading_map)
        )

        shared_expert_output = self.shared_experts_compute(residual)

        if self.config.moe_echo_expert_dispatch_overlap:
            hidden_states, probs = self._echo_preprocess(
                hidden_states,
                rerouted_probs,
                rerouting_map,
                dispatch_stream=MoELayer.moe_a2a_stream,
            )
        else:
            hidden_states, probs = self._echo_preprocess(
                hidden_states, rerouted_probs, rerouting_map
            )

        # Step 4: Token dispatch
        def dispatch_and_compute(hidden_states, probs):
            """Dispatch tokens, run experts, and combine outputs."""
            if self.config.moe_echo_expert_dispatch_overlap:
                with torch.cuda.stream(MoELayer.moe_a2a_stream):
                    # dispatch_preprocess already ran on moe_a2a_stream; wait_stream ensures
                    # any default-stream work submitted between Step 3 and here is also visible.
                    MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                    with torch.cuda.nvtx.range("token_dispatch"):
                        dispatched_input, probs = self.dispatch(hidden_states, probs)
                        dispatched_input, tokens_per_expert, permuted_probs = (
                            self.token_dispatcher.dispatch_postprocess(dispatched_input, probs)
                        )
                    MoELayer.token_dispatch_event.record()
                # Let default stream use dispatched tokens for GEMM only after A2A completes.
                torch.cuda.default_stream().wait_event(MoELayer.token_dispatch_event)
            else:
                with torch.cuda.nvtx.range("token_dispatch"):
                    dispatched_input, probs = self.dispatch(hidden_states, probs)
                    dispatched_input, tokens_per_expert, permuted_probs = (
                        self.token_dispatcher.dispatch_postprocess(dispatched_input, probs)
                    )

            with torch.cuda.nvtx.range("expert_compute"):
                output = self._echo_experts_from_permuted(
                    dispatched_input,
                    tokens_per_expert,
                    permuted_probs,
                    fc1_expert_dispatch_metadata,
                    fc2_expert_dispatch_metadata,
                )

            if self.config.moe_echo_expert_dispatch_overlap:
                with torch.cuda.stream(MoELayer.moe_a2a_stream):
                    MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                    with torch.cuda.nvtx.range("token_combine"):
                        output = self.combine(output)
                    MoELayer.token_combine_event.record()
                torch.cuda.default_stream().wait_event(MoELayer.token_combine_event)
            else:
                with torch.cuda.nvtx.range("token_combine"):
                    output = self.combine(output)
            return output, None, shared_expert_output

        if self.moe_layer_recompute:
            output, mlp_bias, shared_expert_output = tensor_parallel.checkpoint(
                partial(dispatch_and_compute), False, hidden_states, probs
            )
        else:
            output, mlp_bias, shared_expert_output = dispatch_and_compute(hidden_states, probs)

        output = self.postprocess(output, shared_expert_output)

        return output, mlp_bias

    def _echo_log(
        self,
        tokens_per_expert_per_ep_rank: torch.Tensor,
        rerouting_map: torch.Tensor,
        expert_offloading_map: torch.Tensor,
    ):
        """Log echo expert stats for the current step and layer.

        All EP ranks must call this together (collective ops inside).
        Only EP rank 0 writes the result to file.
        """
        from megatron.training.global_vars import get_args
        args = get_args()
        iteration = args.curr_iteration

        # Step filter — consistent across all ranks, no collective ops yet
        if self.config.moe_echo_log_steps is not None:
            log_steps = {int(s) for s in self.config.moe_echo_log_steps.split(',')}
            if iteration not in log_steps:
                return

        # Layer filter — consistent across all ranks
        if self.config.moe_echo_log_layers is not None:
            log_layers = {int(s) for s in self.config.moe_echo_log_layers.split(',')}
            if self.layer_number not in log_layers:
                return

        # ---- Collective operation: all EP ranks must reach here together ----
        # Each rank has its local rerouting_map; sum and all_reduce to get global counts
        ep_size = self.ep_group.size()
        num_experts = self.config.num_moe_experts
        num_home_experts_per_rank = num_experts // ep_size
        num_echo_slots = expert_offloading_map.shape[1] if expert_offloading_map.ndim == 2 else 1
        num_echo_slots_per_rank = max(1, num_echo_slots // ep_size)
        # rerouting_map is the postprocessed tensor from gen_offloading_plan, whose column layout
        # interleaves home experts and echo slots per EP rank:
        #   [home_r0..., echo_r0..., home_r1..., echo_r1..., ...]
        # Each rank occupies `section` consecutive columns.
        section = num_home_experts_per_rank + num_echo_slots_per_rank

        local_after = rerouting_map.sum(dim=0).float().to(tokens_per_expert_per_ep_rank.device)
        torch.distributed.all_reduce(local_after, group=self.ep_group)
        after_counts = local_after.long().cpu()

        # ---- Only rank 0 formats and writes from here ----
        if self.ep_group.rank() != 0:
            return

        tokens_per_expert = tokens_per_expert_per_ep_rank.sum(dim=0).cpu()

        rank_loads_before = [
            tokens_per_expert[r * num_home_experts_per_rank:(r + 1) * num_home_experts_per_rank].sum().item()
            for r in range(ep_size)
        ]
        total_tokens = sum(rank_loads_before)
        avg_rank_load = total_tokens // ep_size
        expert_hot_threshold = avg_rank_load // num_home_experts_per_rank

        rank_loads_after = []
        for r in range(ep_size):
            # postprocessed layout: each rank occupies `section` consecutive columns
            start = r * section
            load = after_counts[start:start + section].sum().item()
            rank_loads_after.append(load)

        def imbalance_pct(loads):
            avg = sum(loads) / len(loads)
            return (max(loads) - avg) / avg * 100 if avg > 0 else 0.0

        imb_before = imbalance_pct(rank_loads_before)
        imb_after = imbalance_pct(rank_loads_after)
        total_rerouted = sum(
            after_counts[r * section + num_home_experts_per_rank:r * section + section].sum().item()
            for r in range(ep_size)
        )

        SEP = "=" * 72
        lines = [
            SEP,
            f"[Echo Expert Stats] L{self.layer_number} | step={iteration} | "
            f"EP={ep_size} ranks | {num_echo_slots_per_rank} echo slots/rank",
            SEP,
            f"[1/4] BEFORE echo routing  "
            f"(rank-hot threshold={avg_rank_load} tokens, "
            f"expert-hot threshold={expert_hot_threshold} tokens)",
            f"  EP rank  load (tot)   per-expert load  (* = hot)",
            f"  {'-' * 64}",
        ]
        for r in range(ep_size):
            load = rank_loads_before[r]
            hot_tag = " HOT" if load > avg_rank_load else "    "
            start = r * num_home_experts_per_rank
            parts = []
            for i in range(num_home_experts_per_rank):
                eidx = start + i
                cnt = tokens_per_expert[eidx].item()
                star = "*" if cnt > expert_hot_threshold else " "
                parts.append(f"E{eidx}{star}: {cnt}")
            lines.append(f"  rank {r}  {load:>9}{hot_tag}  |  {'  '.join(parts)}")
        lines += [
            f"  {'-' * 64}",
            f"  avg={avg_rank_load}  max={max(rank_loads_before)}  "
            f"min={min(rank_loads_before)}  imbalance={imb_before:.1f}%",
            "",
            "[2/4] Echo cloning plan",
        ]
        offload_map_cpu = expert_offloading_map.cpu()
        plan_count = 0
        planned_echo_slots = set()
        for home_local_idx in range(offload_map_cpu.shape[0]):
            for echo_slot_local in range(offload_map_cpu.shape[1]):
                if not offload_map_cpu[home_local_idx, echo_slot_local]:
                    continue
                home_rank = home_local_idx // num_home_experts_per_rank
                home_global_idx = home_rank * num_home_experts_per_rank + (home_local_idx % num_home_experts_per_rank)
                home_load = tokens_per_expert[home_global_idx].item()
                spillover = max(0, home_load - expert_hot_threshold)
                echo_rank = echo_slot_local // num_echo_slots_per_rank if num_echo_slots_per_rank > 0 else 0
                echo_local_in_rank = echo_slot_local % max(num_echo_slots_per_rank, 1)
                # fix: use postprocessed interleaved layout to find the correct column
                echo_col = echo_rank * section + num_home_experts_per_rank + echo_local_in_rank
                rerouted = after_counts[echo_col].item() if echo_col < len(after_counts) else 0
                planned_echo_slots.add(echo_slot_local)
                plan_count += 1
                lines.append(
                    f"  E{home_global_idx} (rank {home_rank}, local {home_local_idx % num_home_experts_per_rank}) "
                    f"| load={home_load} | spillover={spillover}  =>  "
                    f"echo slot {echo_slot_local} (rank {echo_rank}, local {echo_local_in_rank}) "
                    f"=>  {rerouted} tokens rerouted"
                )
        # Also show echo slots activated by depth_first_allocation (not in expert_offloading_map)
        num_echo_slots_total = offload_map_cpu.shape[1] if offload_map_cpu.ndim == 2 else 1
        for echo_slot_local in range(num_echo_slots_total):
            if echo_slot_local in planned_echo_slots:
                continue
            echo_rank = echo_slot_local // num_echo_slots_per_rank if num_echo_slots_per_rank > 0 else 0
            echo_local_in_rank = echo_slot_local % max(num_echo_slots_per_rank, 1)
            echo_col = echo_rank * section + num_home_experts_per_rank + echo_local_in_rank
            rerouted = after_counts[echo_col].item() if echo_col < len(after_counts) else 0
            if rerouted > 0:
                plan_count += 1
                lines.append(
                    f"  [unplanned/DFS] echo slot {echo_slot_local} "
                    f"(rank {echo_rank}, local {echo_local_in_rank}) "
                    f"=>  {rerouted} tokens rerouted"
                )
        if plan_count == 0:
            lines.append("  (no cloning this step)")
        lines += ["", "[3/4] AFTER echo routing  (projected)",
                  "  EP rank  load (tot)  Δ  per-expert load (home + echo slots)",
                  f"  {'-' * 72}"]
        for r in range(ep_size):
            load_after = rank_loads_after[r]
            delta = load_after - rank_loads_before[r]
            delta_str = f"+{delta}" if delta >= 0 else str(delta)
            # fix: rank r's home experts start at col r*section in the postprocessed layout
            rank_start = r * section
            parts = [f"E{r * num_home_experts_per_rank + i}: {after_counts[rank_start + i].item()}"
                     for i in range(num_home_experts_per_rank)]
            for slot in range(num_echo_slots_per_rank):
                col = rank_start + num_home_experts_per_rank + slot
                if col < len(after_counts) and after_counts[col].item() > 0:
                    parts.append(f"Echo{slot}: {after_counts[col].item()}")
            lines.append(f"  rank {r}  {load_after:>9} {delta_str:>7}  |  {'  '.join(parts)}")
        lines += [
            f"  {'-' * 72}",
            f"  avg={avg_rank_load}  max={max(rank_loads_after)}  "
            f"min={min(rank_loads_after)}  imbalance={imb_after:.1f}%",
            "",
            "[4/4] Load-balance improvement",
            f"  imbalance  before: {imb_before:.1f}%  ->  after: {imb_after:.1f}%  "
            f"({imb_after - imb_before:+.1f} pp)",
            f"  max rank load      before: {max(rank_loads_before)}  ->  "
            f"after: {max(rank_loads_after)}  ({max(rank_loads_after) - max(rank_loads_before):+d} tokens)",
            f"  tokens rerouted: {total_rerouted} / {total_tokens} "
            f"({total_rerouted / total_tokens * 100:.1f}%)" if total_tokens > 0 else
            f"  tokens rerouted: {total_rerouted} / {total_tokens}",
            SEP,
            "",
        ]
        with open(self.config.moe_echo_log_file, "a") as f:
            f.write("\n".join(lines))

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor shape [seq_length, bsz, hidden_size].
            padding_mask (torch.Tensor, optional): Boolean mask indicating padding positions.
                                                   Shape [bsz, seq_length]. True = padding,
                                                   False = valid. Defaults to None.
            input_ids (torch.Tensor, optional): Token IDs with shape
                [batch_size, seq_length]. Required by hash routing.
        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )
        self.select_token_dispatcher()

        if self.config.moe_enable_echo:
            if padding_mask is not None:
                raise ValueError("ECHO MoE does not support padding_mask.")
            if intermediate_tensors is not None:
                raise ValueError("ECHO MoE does not support intermediate_tensors.")
            return self.echo_forward(hidden_states)

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states, intermediate_tensors=None, padding_mask=None):
            try:
                if "route" in self.fwd_execution_map:
                    shared_expert_output = self.shared_experts_compute(hidden_states)
                    probs, routing_map = self.route(
                        hidden_states, padding_mask, input_ids, packed_seq_params=packed_seq_params
                    )
                    hidden_states, probs = self.preprocess(
                        hidden_states, probs, routing_map, padding_mask
                    )

                    if intermediate_tensors is not None:
                        return hidden_states, probs, shared_expert_output

            except MoECudaGraphPartialCaptureSignal as e:
                # This signal is raised from the maybe_skip_or_early_return_by_cudagraph decorator.
                # It means we should early-return from the MoE layer forward pass.
                # This happens when we are partially capturing the CUDA graph of the MoE layer,
                # like cuda_graph_modules=["moe_router", "moe_preprocess"].
                # We need to return the intermediate tensors as CUDA graph outputs.
                return e.get_early_return_outputs(hidden_states, shared_expert_output)

            if "expert_compute" in self.fwd_execution_map:
                if intermediate_tensors is not None:
                    hidden_states, probs = intermediate_tensors

                dispatched_input, probs = self.dispatch(hidden_states, probs)
                output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
                assert (
                    mlp_bias is None
                ), f"mlp_bias is not supported for {type(self.token_dispatcher)}"
                output = self.combine(output)

                if intermediate_tensors is not None:
                    return output, mlp_bias

            if "postprocess" in self.fwd_execution_map:
                if intermediate_tensors is not None:
                    output, shared_expert_output = intermediate_tensors

                output = self.postprocess(output, shared_expert_output)

                if intermediate_tensors is not None:
                    return output

            return output, mlp_bias

        def routed_experts_forward(hidden_states, shared_expert_output=None):
            probs, routing_map = self.route(
                hidden_states, padding_mask, input_ids, packed_seq_params=packed_seq_params
            )
            hidden_states, probs = self.preprocess(
                hidden_states, probs, routing_map, padding_mask
            )
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
            assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
            output = self.combine(output)
            return output, mlp_bias

        if self.moe_layer_recompute and self.training:
            if self.config.fp8 or self.config.fp4:
                outputs = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.tp_group,
                    hidden_states,
                    intermediate_tensors,
                    padding_mask,
                )
            else:
                outputs = tensor_parallel.checkpoint(
                    custom_forward, False, hidden_states, intermediate_tensors, padding_mask
                )
        elif (
            self.routed_experts_recompute
            and self.training
            and torch.is_grad_enabled()
            and intermediate_tensors is None
            and self.fwd_execution_map == ["route", "expert_compute", "postprocess"]
        ):
            shared_expert_output = self.shared_experts_compute(hidden_states)
            checkpoint_args = (
                (hidden_states,)
                if shared_expert_output is None
                else (hidden_states, shared_expert_output)
            )
            if self.config.fp8 or self.config.fp4:
                outputs = te_checkpoint(
                    routed_experts_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.tp_group,
                    *checkpoint_args,
                )
            else:
                outputs = tensor_parallel.checkpoint(
                    routed_experts_forward,
                    False,
                    *checkpoint_args,
                )
            output, mlp_bias = outputs
            output = self.postprocess(output, shared_expert_output)
            outputs = (output, mlp_bias)
        else:
            outputs = custom_forward(hidden_states, intermediate_tensors, padding_mask)

        return outputs

    def backward_dw(self, routed_experts: bool = True, shared_experts: bool = False):
        """Compute weight gradients for experts and shared experts."""
        from megatron.core.pipeline_parallel.utils import get_comm_stream

        # TODO(Wohox): replace the "routed_experts" and "shared_experts" arguments with better
        # naming to better explain that they are actually from different fine-grained callables,
        # or use scanning to decide which backward_dw should be called.
        if routed_experts:
            self.experts.backward_dw()
            if self.config.moe_latent_size and self.config.overlap_moe_expert_parallel_comm:
                # TODO(Wohox): fc2_latent_proj forward and backward are executed in comm stream,
                # so we execute its backward_dw in the comm stream too. But this may harm the
                # EP overlap performance. Better to check if there is a better way to handle this.
                comm_stream = get_comm_stream()
                with torch.cuda.stream(comm_stream):
                    self.fc2_latent_proj.backward_dw()
        if shared_experts:
            if self.use_shared_expert and not self.shared_expert_overlap:
                self.shared_experts.backward_dw()
            if self.config.moe_latent_size and self.config.overlap_moe_expert_parallel_comm:
                self.fc1_latent_proj.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8/fp4."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)


class _RecordExpertDgradCompletion(torch.autograd.Function):
    """Autograd function that records a CUDA event when expert data gradients finish.

    Placed in the forward graph just before the expert computation so that during
    the backward pass, when the expert dgrad completes, we record an event. The
    subsequent ``_RegisterDelayedWgradForExperts`` waits on this event before
    launching the delayed wgrad computation on a separate CUDA stream.
    """

    @staticmethod
    def forward(ctx, event: torch.cuda.Event, *inputs):
        """Forward pass that stores the event and passes through inputs unchanged."""
        ctx.event = event
        return inputs[0] if len(inputs) == 1 else inputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass that records the event when expert dgrad completes."""
        ctx.event.record(torch.cuda.current_stream())
        ctx.event = None
        return (None,) + grad_outputs


class _RegisterDelayedWgradForExperts(torch.autograd.Function):
    """Autograd function that orchestrates delayed wgrad computation for MoE experts.

    Placed in the forward graph at the dispatch boundary. During the backward pass,
    this function:
      1. Records an event on the current (backward) stream to signal the dgrad is done.
      2. Executes the delayed wgrad computation on a dedicated CUDA stream.
      3. Waits for the wgrad computation to complete.
      4. Invokes the registered gradient processing callback (e.g., FSDP reduce-scatter).
    """

    @staticmethod
    def forward(ctx, module: MoELayer, *inputs):
        """Forward pass that stores the MoE module and passes through inputs unchanged."""
        ctx.module = module
        return inputs[0] if len(inputs) == 1 else inputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass that executes delayed wgrad computation on a separate stream."""
        module = ctx.module
        event = module._delayed_wgrad_event
        wgrad_stream = module._delayed_wgrad_stream

        wgrad_stream.wait_event(event)
        with torch.cuda.stream(wgrad_stream):
            nvtx_range_push("delayed_expert_wgrad")
            module.backward_dw(routed_experts=True, shared_experts=False)
            nvtx_range_pop("delayed_expert_wgrad")
            event.record(wgrad_stream)

        torch.cuda.current_stream().wait_event(event)

        for param in module.parameters():
            if getattr(param, "post_wgrad_grad_acc_hook", None) is not None:
                param.post_wgrad_grad_acc_hook()

        ctx.module = None
        return (None,) + grad_outputs
