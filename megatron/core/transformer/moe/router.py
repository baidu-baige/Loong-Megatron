# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Optional

import torch

from megatron.core.tensor_parallel import reduce_from_tensor_model_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    ProcessGroupCollection,
    apply_random_logits,
    apply_router_token_dropping,
    compute_routing_scores_for_aux_loss,
    router_gating_linear,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.tp_cp_group = pg_collection.tp_cp
        self.tp_dp_cp_group = pg_collection.tp_dp_cp

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if self.config.add_bias_linear:
            self.bias = torch.nn.Parameter(
                torch.empty((self.config.num_moe_experts), dtype=torch.float32)
            )
        else:
            self.bias = None
        # If calculate per token loss, we need to scale up moe aux loss by the number of tokens.
        # So we need to know if the model is configured to calculate per token loss.
        self.calculate_per_token_loss = self.config.calculate_per_token_loss
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the router parameters."""
        if self.config.perform_initialization:
            self.config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=self.config.params_dtype)
        setattr(self.weight, 'sequence_parallel', self.config.sequence_parallel)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype=self.config.params_dtype)
            setattr(self.bias, 'sequence_parallel', self.config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts.

    The workflow of TopKRouter is as follows:
    (1) Calculate the logits by the router gating network.
    (2) Calculate the routing probabilities and map for top-k selection with score function.
    (3) [Optional] Apply token dropping to top-k expert selection.
    (4) [Optional] Apply the auxiliary load balancing loss for the given scores and routing map.

    Naming convention:
        logits: The output logits by the router gating network.
        scores: The scores after score function used to select the experts and calculate aux loss.
        probs: The topk weights used to combined the experts' outputs.
        routing_map: The masked routing map between tokens and experts.
    """

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

        # Initialize global tokens per expert for global aux loss
        if self.get_aux_loss_coeff("global_aux_loss") > 0:
            self.register_buffer(
                'global_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'ga_steps',
                torch.tensor(0, dtype=torch.float32, device=torch.cuda.current_device()),
                persistent=False,
            )
        else:
            self.global_tokens_per_expert = None
            self.ga_steps = None

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def get_aux_loss_coeff(self, aux_loss_type: str) -> float:
        """Return the aux loss coeff for the given auxiliary loss type.
        If the auxiliary loss type is not found, return 0.0.
        """
        if isinstance(self.routing_type, str):
            if self.routing_type == aux_loss_type:
                return self.config.moe_aux_loss_coeff
        if isinstance(self.routing_type, list):
            try:
                idx = self.routing_type.index(aux_loss_type)
                return self.config.moe_aux_loss_coeff[idx]
            except ValueError:
                return 0.0
        return 0.0

    def is_aux_loss_enabled(self) -> bool:
        """Check if the auxiliary loss is enabled."""
        for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
            if self.get_aux_loss_coeff(aux_loss_type) > 0:
                return True
        return False

    def _apply_aux_loss(
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply the auxiliary loss for the given scores and routing map."""
        aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
        if aux_loss_coeff == 0:
            return probs
        
        num_tokens = routing_map.shape[0]
        if not self.config.enable_chunkpipe:
            tokens_per_expert = routing_map.sum(dim=0)
        else:
            if self.config.sft_chunkpipe_mode:
                # SFT: scheduler sets chunk index directly
                effective_group = self.config.chunkpipe_current_group_size
                chunk_index = self.config.chunkpipe_chunk_idx_in_group
                microbatch_key = self.config.chunkpipe_backward_microbatch - chunk_index
            else:
                # Pretrain: compute from global counter (all groups same size)
                effective_group = self.config.chunk_num_per_seq
                chunk_index = self.config.chunkpipe_backward_microbatch % effective_group
                microbatch_key = self.config.chunkpipe_backward_microbatch // effective_group
            num_tokens = num_tokens * effective_group
            ftp_map = getattr(self, '_chunkpipe_full_tokens_per_expert_map', {})
            tokens_per_expert = ftp_map.get(microbatch_key, None)
            if tokens_per_expert is None:
                tokens_per_expert = routing_map.reshape(num_tokens, -1).sum(dim=0)
            else:
                # Clone to prevent in-place all-reduce from corrupting the cached tensor.
                tokens_per_expert = tokens_per_expert.clone()
            # Clean up this microbatch's entry after the last backward chunk (chunk_index == 0)
            if chunk_index == 0 and microbatch_key in ftp_map:
                del ftp_map[microbatch_key]

        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_cp_group
        )

        total_num_tokens = num_tokens * self.tp_cp_group.size()

        aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=aux_loss_coeff,
            fused=self.config.moe_router_fusion,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs, aux_loss_coeff, aux_loss, "load_balancing_loss", self.tp_cp_group
        )
        return probs

    def _apply_seq_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        seq_length: int,
        bsz: int,
    ):
        """Apply the sequence-level auxiliary loss for the given scores and routing map.

        To calculate the sequence-level aux loss, we reshape the batch_size dimension to
        experts dimension. The resulted loss by switch_load_balancing_loss_func is equal
        to the sum of aux loss for each sequence in the batch. And then we divide the aux
        loss by the batch size to get averaged aux loss.

        When chunkpipe is enabled, sequences are split into chunks and each chunk is
        forwarded independently. To compute the exact same loss as the non-chunked case,
        we accumulate scores and tokens_per_expert across all chunks of a sequence, and
        compute the loss only on the last chunk using the full-sequence statistics.
        """
        seq_aux_loss_coeff = self.get_aux_loss_coeff("seq_aux_loss")
        if seq_aux_loss_coeff == 0:
            return probs
        if self.config.enable_chunkpipe and self.config.chunk_num_per_seq > 1:
            return self._apply_seq_aux_loss_chunkpipe(
                probs, scores_for_aux_loss, routing_map, seq_length, bsz, seq_aux_loss_coeff
            )

        scores_for_aux_loss = scores_for_aux_loss.reshape(seq_length, -1)
        tokens_per_expert = routing_map.reshape(seq_length, -1).sum(dim=0)
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_cp_group
        )

        total_num_tokens = seq_length * self.tp_cp_group.size()

        aux_loss = (
            switch_load_balancing_loss_func(
                probs=scores_for_aux_loss,
                tokens_per_expert=tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=seq_aux_loss_coeff,
                fused=self.config.moe_router_fusion,
            )
            / bsz
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs, seq_aux_loss_coeff, aux_loss, "seq_load_balancing_loss", self.tp_cp_group
        )
        return probs


    def _accumulate_chunkpipe_tokens_per_expert(
        self, routing_map: torch.Tensor, seq_length: int, bsz: int
    ):
        """Accumulate tokens_per_expert across chunks during the original forward pass.

        This runs under torch.no_grad() (inside CheckpointFunction.forward) so that
        the full-sequence tokens_per_expert is available for each chunk's backward
        recomputation to compute per-chunk partial aux_loss with correct gradients.

        Values are stored per-microbatch to handle 1F1B pipeline interleaving where
        multiple microbatches' forwards may complete before any backward starts.
        """
        chunkpipe_fwd_mb = self.config.chunkpipe_forward_microbatch
        if self.config.sft_chunkpipe_mode:
            # SFT: scheduler sets chunk index directly
            chunk_index = self.config.chunkpipe_chunk_idx_in_group
            microbatch_key = chunkpipe_fwd_mb - chunk_index
        else:
            # Pretrain: compute from global counter (all groups same size)
            chunk_num = self.config.chunk_num_per_seq
            chunk_index = chunkpipe_fwd_mb % chunk_num
            microbatch_key = chunkpipe_fwd_mb // chunk_num

        tokens_per_expert_chunk = routing_map.reshape(seq_length, -1).sum(dim=0)

        if not hasattr(self, '_chunkpipe_full_tokens_per_expert_map'):
            self._chunkpipe_full_tokens_per_expert_map = {}

        if chunk_index == 0:
            # First chunk in forward order: initialize for this microbatch
            self._chunkpipe_full_tokens_per_expert_map[microbatch_key] = (
                tokens_per_expert_chunk.detach().clone()
            )
        else:
            self._chunkpipe_full_tokens_per_expert_map[microbatch_key].add_(
                tokens_per_expert_chunk.detach()
            )

    def _apply_seq_aux_loss_chunkpipe(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        seq_length: int,
        bsz: int,
        seq_aux_loss_coeff: float,
    ):
        """Compute seq_aux_loss equivalent to the non-chunked case under chunkpipe.

        The non-chunked loss for sample b is:
            L_b = Σ_e (Σ_t P[b,t,e]) * (Σ_t F[b,t,e]) * E * coeff / (topk * S²)
        where S is the full sequence length.

        The loss can be decomposed into per-chunk contributions:
            L = Σ_c L_c, where L_c = coeff * Σ_e (Σ_{t∈chunk_c} P[t,e]) * F_total[e] / ...

        Each chunk computes its partial loss L_c using:
            - Current chunk's scores (WITH gradient) for Σ_{t∈chunk_c} P[t,e]
            - Full-sequence tokens_per_expert F_total (pre-accumulated during original
              forward, detached) as the coefficient
            - total_num_tokens = full_seq_length (S)
        """
        if self.config.sft_chunkpipe_mode:
            # SFT: scheduler sets chunk index directly
            effective_group = self.config.chunkpipe_current_group_size
            chunk_index = self.config.chunkpipe_chunk_idx_in_group
            microbatch_key = self.config.chunkpipe_backward_microbatch - chunk_index
        else:
            # Pretrain: compute from global counter (all groups same size)
            effective_group = self.config.chunk_num_per_seq
            chunk_index = self.config.chunkpipe_backward_microbatch % effective_group
            microbatch_key = self.config.chunkpipe_backward_microbatch // effective_group
        scores_chunk = scores_for_aux_loss.reshape(seq_length, -1)

        # Look up full-sequence tokens_per_expert pre-accumulated during original forward.
        # During backward recomputation, use chunkpipe_backward_microbatch to find the
        # correct microbatch's accumulated tokens_per_expert.
        ftp_map = getattr(self, '_chunkpipe_full_tokens_per_expert_map', {})
        tokens_per_expert_chunk = ftp_map.get(microbatch_key, None)
        # Clean up this microbatch's entry after the last backward chunk (chunk_index == 0)
        if chunk_index == 0 and microbatch_key in ftp_map:
            del ftp_map[microbatch_key]

        if tokens_per_expert_chunk is None:
            # Fallback: if accumulation didn't happen, use current chunk's tokens_per_expert
            tokens_per_expert_chunk = routing_map.reshape(seq_length, -1).sum(dim=0)
        else:
            # Clone to prevent in-place all-reduce from corrupting the cached tensor,
            # which is reused by subsequent backward chunks.
            tokens_per_expert_chunk = tokens_per_expert_chunk.clone()
        full_tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert_chunk, self.tp_cp_group
        ).detach()
        # Use full-sequence parameters for correct scaling
        full_seq_length = seq_length * effective_group
        total_num_tokens = full_seq_length * self.tp_cp_group.size()

        # Compute this chunk's partial aux_loss contribution:
        # L_c = coeff * Σ_e [ (Σ_{t∈chunk} scores[t,e]) * F_total[e] ] / (topk * T² * bsz)
        aux_loss = (
            switch_load_balancing_loss_func(
                probs=scores_chunk,
                tokens_per_expert=full_tokens_per_expert.detach(),
                total_num_tokens=total_num_tokens,
                topk=self.topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=seq_aux_loss_coeff,
                fused=self.config.moe_router_fusion,
            )
            / bsz
        )

        probs = self.attach_and_log_load_balancing_loss(
            probs, seq_aux_loss_coeff, aux_loss, "seq_load_balancing_loss", self.tp_cp_group
        )

        return probs

    def _apply_global_aux_loss(
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply the global auxiliary loss for the given scores and routing map."""
        global_aux_loss_coeff = self.get_aux_loss_coeff("global_aux_loss")
        if global_aux_loss_coeff == 0:
            return probs

        num_tokens = scores_for_aux_loss.shape[0]
        if not self.config.enable_chunkpipe:
            tokens_per_expert = routing_map.sum(dim=0)
        else:
            if self.config.sft_chunkpipe_mode:
                # SFT: scheduler sets chunk index directly
                effective_group = self.config.chunkpipe_current_group_size
                chunk_index = self.config.chunkpipe_chunk_idx_in_group
                microbatch_key = self.config.chunkpipe_backward_microbatch - chunk_index
            else:
                # Pretrain: compute from global counter (all groups same size)
                effective_group = self.config.chunk_num_per_seq
                chunk_index = self.config.chunkpipe_backward_microbatch % effective_group
                microbatch_key = self.config.chunkpipe_backward_microbatch // effective_group
            num_tokens = num_tokens * effective_group
            ftp_map = getattr(self, '_chunkpipe_full_tokens_per_expert_map', {})
            tokens_per_expert = ftp_map.get(microbatch_key, None)
            if tokens_per_expert is None:
                tokens_per_expert = routing_map.reshape(num_tokens, -1).sum(dim=0)
            else:
                # Clone to prevent in-place all-reduce from corrupting the cached tensor.
                tokens_per_expert = tokens_per_expert.clone()
            # Clean up this microbatch's entry after the last backward chunk (chunk_index == 0)
            if chunk_index == 0 and microbatch_key in ftp_map:
                del ftp_map[microbatch_key]

        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_dp_cp_group
        )
        if not self.config.enable_chunkpipe \
            or chunk_index == effective_group - 1:
            self.global_tokens_per_expert += tokens_per_expert
            self.ga_steps += 1
        averated_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps

        total_num_tokens = num_tokens * self.tp_dp_cp_group.size()

        global_aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=averated_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=global_aux_loss_coeff,
            fused=self.config.moe_router_fusion,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs,
            global_aux_loss_coeff,
            global_aux_loss,
            "global_load_balancing_loss",
            self.tp_dp_cp_group,
        )
        return probs

    def attach_and_log_load_balancing_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
        aux_loss_name: str,
        reduce_group: torch.distributed.ProcessGroup,
    ):
        """Attach aux loss function to activation and add to logging."""
        # TODO (zijiey): fix the per_layer_logging for MTP, currently it will incorrectly
        # add the aux loss logging value to other layer's since it is difficult to get the
        # correct layer_number for MTP. It does not affect the correctness of the calculation
        # results and the reduced load_balancing_loss logging value.
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        if self.config.enable_chunkpipe:
            effective_group = self.config.chunkpipe_current_group_size or self.config.chunk_num_per_seq

        # For SFT chunkpipe: the DataLoader already produces chunk-level micro-batches, so
        # get_num_microbatches() naturally returns 4N (chunk-inflated). The tracker's
        # loss_scale = 1/(4N) causes each per-chunk partial loss to be weighted 4x too small.
        # We must scale the logged value by effective_group to match the non-chunked baseline.
        #
        # For pretrain chunkpipe: get_num_microbatches() is NOT chunk-inflated (returns N),
        # because training_utils.py inflates the loop count separately. The tracker's
        # loss_scale = 1/N combined with 4N router calls already sums correctly without scaling.
        if self.config.enable_chunkpipe and self.config.sft_chunkpipe_mode:
            log_loss = aux_loss * effective_group
        else:
            log_loss = aux_loss
        save_to_aux_losses_tracker(
            aux_loss_name,
            log_loss / aux_loss_coeff,
            self.layer_number,
            num_layers,
            reduce_group=reduce_group,
        )

        # Scale the gradient contribution by effective_group for chunkpipe (both SFT and pretrain).
        # Each chunk only has 1/effective_group of the tokens; the gradient must reflect the full
        # sequence to match the non-chunked training dynamics.
        if self.config.enable_chunkpipe:
            aux_loss = aux_loss * effective_group

        if self.calculate_per_token_loss:
            # Scale the aux_loss by the number of tokens.
            # The expected final scaling for aux_loss gradients is 1/(num_micro_batches * dp_size).
            # After commit 02648000, Megatron started using the number of total tokens to scale
            # gradients under the argument of calculate_per_token_loss,
            # which scales both the main_loss gradient and aux_loss gradient by
            # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads function.
            # To correct this scaling, we need to scale the aux_loss by num_local_tokens here.
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * activation.shape[0])
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training and torch.is_grad_enabled():
            # Skip Z loss calculations when using torch.no_grad() or checkpointing.
            moe_z_loss_coeff = self.config.moe_z_loss_coeff / self.tp_cp_group.size()
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            scale_up = 1.0
            if self.calculate_per_token_loss:
                # The expected final scaling for z_loss gradients is
                # 1/(num_micro_batches * dp_size).
                # After commit 02648000, Megatron started using the number of total tokens
                # to scale gradients under the argument of calculate_per_token_loss,
                # which scales both the main_loss gradient and z_loss gradient by
                # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads().
                # To correct this scaling, we need to scale the z_loss by num_local_tokens here.
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss * logits.shape[0])
            else:
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)

            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, dtype=input.dtype, device=input.device),
                    torch.tensor(1.0 + eps, dtype=input.dtype, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        # Calculate probs and routing_map for token dispatching
        if self.routing_type == "sinkhorn":
            probs, routing_map = self.sinkhorn_load_balancing(logits)
        else:
            probs, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                fused=self.config.moe_router_fusion,
            )

        # Apply token dropping to probs and routing_map.
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Pre-accumulate tokens_per_expert during original forward (under no_grad)
        # for chunkpipe seq_aux_loss. This runs before activation recomputation so that
        # the full-sequence tokens_per_expert is available during each chunk's backward.
        # Use routing_map from compute_routing_scores_for_aux_loss (without expert_bias
        # and group_topk) to match what baseline _apply_seq_aux_loss uses.
        if (self.training and not torch.is_grad_enabled()
                and self.config.enable_chunkpipe and self.config.chunk_num_per_seq > 1):
            routing_map_for_accum, _ = compute_routing_scores_for_aux_loss(
                logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
            )
            self._accumulate_chunkpipe_tokens_per_expert(routing_map_for_accum, seq_length, bsz)

        # Apply each aux loss type and attach aux loss autograd function to probs
        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            # Calculate scores and routing_map for aux loss
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
            )
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
            probs = self._apply_global_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss
            )

        # Update expert bias and tokens_per_expert
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        return probs, routing_map

    def reset_global_aux_loss_tracker(self):
        """Reset the global aux loss tracker."""
        if self.global_tokens_per_expert is not None:
            self.global_tokens_per_expert.zero_()
            self.ga_steps.zero_()

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            # Apply force load balancing with random logits for benchmark
            logits = apply_random_logits(logits)

        probs, routing_map = self.routing(logits)

        return probs, routing_map

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)
