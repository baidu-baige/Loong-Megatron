# Copyright (c) 2025, NVIDIA CORPORATION and Alibaba PAI. All rights reserved.
from collections import defaultdict
from typing import Dict

import torch

_CPU_ADAM_STATE_KEYS = {"exp_avg", "exp_avg_sq", "adamw_exp_avg", "adamw_exp_avg_sq"}

from megatron.core.fp8_utils import (
    dequantize_fp8_tensor,
    get_fp8_cpu_offload_proxy_info,
    get_fp8_cpu_offload_proxy_numel,
)

_FP8_WRITEBACK_STAGE_BYTES = 512 * 1024 * 1024


def _param_generator(cpu_optimizer):
    for group in cpu_optimizer.param_groups:
        for param in group["params"]:
            yield param


class HybridDeviceOptimizer(torch.optim.Optimizer):
    """
    HybridDeviceOptimizer is a custom optimizer designed to facilitate
    hybrid parameter updates across GPU and CPU. This optimizer allows
    users to adjust the fraction of parameters updated on the CPU and
    GPU through the `offload_fraction` parameter.

    It supports bf16 mixed-precision training. Additionally, the optimizer
    implements overlapping operations for improved performance, including
    gradient transfer from device to host (D2H) and parameter transfer
    from host to device (H2D).

    Example:
        from transformer_engine.pytorch.optimizers import FusedAdam as GPUAdam
        from torch.optim import AdamW as CPUAdam
        optimizer = HybridDeviceOptimizer(
            param_groups,
            cpu_optimizer_cls=CPUAdam,
            gpu_optimizer_cls=GPUAdam,
            offload_fraction=0.5,
            param_update_in_fp32=True,
            overlap_cpu_optimizer_d2h_h2d=True,
        )
        optimizer.step()

    Note:
        This optimizer is particularly useful in scenarios where memory
        constraints are present or when leveraging both CPU and GPU resources
        can lead to performance improvements.
    """

    def __init__(
        self,
        params,
        offload_fraction=0.5,
        cpu_optimizer_cls=None,
        gpu_optimizer_cls=None,
        param_update_in_fp32: bool = False,
        pin_cpu_grads: bool = True,
        pin_cpu_params: bool = True,
        overlap_cpu_optimizer_d2h_h2d: bool = True,
        grad_streaming: bool = False,
        grad_streaming_bucket_bytes: int = 4 * 1024 * 1024 * 1024,
        contiguous_state: bool = False,
        **kwargs,
    ):
        super(HybridDeviceOptimizer, self).__init__(
            params,
            defaults={
                "offload_fraction": offload_fraction,
                "cpu_optimizer_cls": cpu_optimizer_cls,
                "gpu_optimizer_cls": gpu_optimizer_cls,
                "param_update_in_fp32": param_update_in_fp32,
                "pin_cpu_grads": pin_cpu_grads,
                "pin_cpu_params": pin_cpu_params,
                "overlap_cpu_optimizer_d2h_h2d": overlap_cpu_optimizer_d2h_h2d,
                "grad_streaming": grad_streaming,
                "grad_streaming_bucket_bytes": grad_streaming_bucket_bytes,
                "contiguous_state": contiguous_state,
                **kwargs,
            },
        )

        self.offload_fraction = offload_fraction
        self.cpu_optimizer_cls = cpu_optimizer_cls
        self.gpu_optimizer_cls = gpu_optimizer_cls
        self.pin_cpu_grads = pin_cpu_grads
        self.pin_cpu_params = pin_cpu_params
        self.grad_streaming = grad_streaming
        self.grad_streaming_bucket_bytes = grad_streaming_bucket_bytes
        self.overlap_cpu_optimizer_d2h_h2d = overlap_cpu_optimizer_d2h_h2d or grad_streaming
        self.param_update_in_fp32 = param_update_in_fp32
        self.contiguous_state = contiguous_state
        self.state_arenas = None
        self._arena_slices = None
        self.sub_optimizer_kwargs = kwargs
        self._fp8_writeback_waves = None
        self._fp8_writeback_group = None

        self._init_sub_optimizers()
        self._register_load_state_dict_hooks()

    def _get_high_prec_param_shard_for_fp8_proxy(self, proxy_param):
        """Return this proxy's current high-precision model-param shard."""
        info = get_fp8_cpu_offload_proxy_info(proxy_param)
        assert info is not None, "Expected an FP8 CPU-offload proxy with metadata."
        model_param = info.blockwise_fp8_model_param
        start_offset = info.start_offset
        shard_numel = info.shard_numel

        high_precision_init_val = None
        if hasattr(model_param, "get_high_precision_init_val"):
            high_precision_init_val = model_param.get_high_precision_init_val()
        if high_precision_init_val is not None:
            return high_precision_init_val.view(-1)[start_offset : start_offset + shard_numel]

        return dequantize_fp8_tensor(model_param).view(-1)[
            start_offset : start_offset + shard_numel
        ]

    def _build_fp8_cpu_offload_master_param_shard(self, proxy_param):
        """Build the optimizer-owned CPU FP32 master param shard for an FP8 proxy."""
        info = get_fp8_cpu_offload_proxy_info(proxy_param)
        assert info is not None, "Expected an FP8 CPU-offload proxy with metadata."
        model_param_shard = self._get_high_prec_param_shard_for_fp8_proxy(proxy_param)
        master_param = model_param_shard.detach().to(
            device="cpu", dtype=torch.float32, copy=True
        ).contiguous()
        if self.pin_cpu_params:
            master_param = master_param.pin_memory()

        model_param = info.blockwise_fp8_model_param
        if hasattr(model_param, "clear_high_precision_init_val"):
            model_param.clear_high_precision_init_val()
        return master_param

    def set_fp8_cpu_offload_writeback_plan(
        self, model_params, data_parallel_group, stage_bytes=_FP8_WRITEBACK_STAGE_BYTES
    ):
        """Register the rank-invariant plan for FP8 master write-back."""
        model_params = list(model_params)
        waves = []
        current_wave = []
        current_bytes = 0
        for model_param in model_params:
            nbytes = model_param.numel() * 4
            if current_wave and current_bytes + nbytes > stage_bytes:
                waves.append(current_wave)
                current_wave = []
                current_bytes = 0
            current_wave.append(model_param)
            current_bytes += nbytes
        if current_wave:
            waves.append(current_wave)

        self._fp8_writeback_waves = waves
        self._fp8_writeback_group = data_parallel_group

    def _collect_fp8_offload_master_shards(self):
        """Map FP8 model parameters to their local CPU master shard."""
        shards = {}
        for cpu_param, gpu_param in self.cpu_copys_map_gpu_param.items():
            info = get_fp8_cpu_offload_proxy_info(gpu_param)
            if info is not None:
                shards[info.blockwise_fp8_model_param] = (cpu_param, info.start_offset)
        return shards

    def _writeback_fp8_cpu_offload_masters(self):
        """Quantize CPU FP32 master shards once per step in rank-invariant waves."""
        from megatron.core.fp8_utils import quantize_param_shard

        shards = self._collect_fp8_offload_master_shards()
        if self._fp8_writeback_waves is None:
            assert not shards, (
                "FP8 CPU-offload master shards exist but no write-back plan was registered."
            )
            return

        for wave in self._fp8_writeback_waves:
            main_params = []
            start_offsets = []
            staged_masters = []
            for model_param in wave:
                entry = shards.get(model_param)
                if entry is None:
                    main_params.append(None)
                    start_offsets.append(None)
                    continue
                master_param, start_offset = entry
                if not master_param.is_cuda:
                    master_param = master_param.to(model_param.device, non_blocking=True)
                    staged_masters.append(master_param)
                main_params.append(master_param)
                start_offsets.append(start_offset)

            quantize_param_shard(
                wave, main_params, start_offsets, self._fp8_writeback_group
            )
            del staged_masters

    def _set_gpu_fp32_grads(self):
        """Copy GPU model gradients to their FP32 optimizer parameters."""
        if not self.param_update_in_fp32:
            return
        for param in self.param_to_fp32_param:
            if param in self.gpu_params_map_cpu_copy:
                continue
            fp32_param = self.param_to_fp32_param[param]
            grad = getattr(param, "decoupled_grad", param.grad)
            if grad is not None:
                fp32_param.grad = grad.to(fp32_param.dtype)
                fp32_param.requires_grad = True
            else:
                fp32_param.grad = None
                fp32_param.requires_grad = False

    def _set_sub_optimizer_grads(self):
        self._set_gpu_fp32_grads()

        # Sync the grads from GPU to CPU.
        for optimizer in self.cpu_optimizers:
            for param in _param_generator(optimizer):
                gpu_param = self.cpu_copys_map_gpu_param[param]
                grad = getattr(gpu_param, "decoupled_grad", gpu_param.grad)
                if grad is None:
                    param.grad = None
                    param.requires_grad = False
                    continue

                param.requires_grad = False
                if param not in self.cpu_copy_map_grad:
                    self.cpu_copy_map_grad[param] = torch.empty(
                        param.shape, dtype=param.dtype, pin_memory=self.pin_cpu_grads, device="cpu"
                    )

                param.grad = self.cpu_copy_map_grad[param]
                self.cpu_copy_map_grad[param].data.copy_(grad, non_blocking=True)
            self._cpu_optimizer_map_data_event[optimizer] = self._d2h_stream.record_event()

    def _build_grad_streaming_plan(self):
        """Partition CPU optimizers into bounded transfer waves."""
        alignment = 16
        waves = []
        current_wave = []
        current_bytes = 0
        arena_bytes = 0
        for optimizer in self.cpu_optimizers:
            optimizer_params = []
            optimizer_bytes = 0
            for param in _param_generator(optimizer):
                nbytes = param.numel() * param.element_size()
                aligned_bytes = (nbytes + alignment - 1) // alignment * alignment
                optimizer_params.append((param, optimizer_bytes, nbytes))
                optimizer_bytes += aligned_bytes

            if current_wave and current_bytes + optimizer_bytes > self.grad_streaming_bucket_bytes:
                waves.append(current_wave)
                arena_bytes = max(arena_bytes, current_bytes)
                current_wave = []
                current_bytes = 0

            optimizer_params = [
                (param, current_bytes + offset, nbytes)
                for param, offset, nbytes in optimizer_params
            ]
            current_wave.append((optimizer, optimizer_params))
            current_bytes += optimizer_bytes

        if current_wave:
            waves.append(current_wave)
            arena_bytes = max(arena_bytes, current_bytes)
        return waves, arena_bytes

    def _ensure_grad_streaming_buffers(self):
        """Allocate the two pinned gradient staging arenas once."""
        if getattr(self, "_gs_waves", None) is not None:
            return
        self._gs_waves, arena_bytes = self._build_grad_streaming_plan()
        self._gs_arenas = (
            [
                torch.empty(
                    arena_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=self.pin_cpu_grads,
                )
                for _ in range(2)
            ]
            if arena_bytes > 0
            else []
        )
        self._gs_events = [None, None]

    def _gs_issue_wave(self, wave_index):
        """Enqueue asynchronous D2H copies for one gradient wave."""
        if wave_index >= len(self._gs_waves):
            return
        arena = self._gs_arenas[wave_index % 2]
        with torch.cuda.stream(self._d2h_stream):
            for _, optimizer_params in self._gs_waves[wave_index]:
                for param, byte_offset, nbytes in optimizer_params:
                    gpu_param = self.cpu_copys_map_gpu_param[param]
                    grad = getattr(gpu_param, "decoupled_grad", gpu_param.grad)
                    param.requires_grad = False
                    if grad is None:
                        param.grad = None
                        continue
                    view = (
                        arena[byte_offset : byte_offset + nbytes]
                        .view(param.dtype)
                        .view(param.shape)
                    )
                    view.copy_(grad, non_blocking=True)
                    param.grad = view
            self._gs_events[wave_index % 2] = self._d2h_stream.record_event()

    def _streamed_cpu_steps(self, closure=None):
        """Consume gradient waves while the following wave transfers."""
        import os
        import time

        debug = os.environ.get("GRAD_STREAMING_DEBUG", "0") == "1"
        rank = os.environ.get("RANK", "?")

        def debug_log(message):
            if debug:
                print(f"[grad-stream][rank {rank}] {message}", flush=True)

        self._ensure_grad_streaming_buffers()
        if not self._gs_waves:
            return

        wave_count = len(self._gs_waves)
        started = time.monotonic()
        sync_seconds = 0.0
        step_seconds = 0.0
        max_step_seconds = 0.0
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        self._gs_issue_wave(0)
        self._gs_issue_wave(1)
        for wave_index, wave in enumerate(self._gs_waves):
            sync_started = time.monotonic()
            event = self._gs_events[wave_index % 2]
            if event is not None:
                event.synchronize()
            step_started = time.monotonic()
            for optimizer, _ in wave:
                optimizer.step(closure)
            step_finished = time.monotonic()
            self._gs_issue_wave(wave_index + 2)
            sync_seconds += step_started - sync_started
            step_seconds += step_finished - step_started
            max_step_seconds = max(max_step_seconds, step_finished - step_started)

        arena_text = (
            f" arena {len(self._gs_arenas[0]) / 2**30:.2f}GiBx2"
            if self._gs_arenas
            else ""
        )
        debug_log(
            f"{wave_count} waves in {time.monotonic() - started:.2f}s: "
            f"sync {sync_seconds:.2f}s step {step_seconds:.2f}s "
            f"(max {max_step_seconds:.2f}s){arena_text}"
        )

    def _register_param_copy_back_gpu_hook(self):
        def param_copy_back_gpu_hook_closure():
            def param_copy_back_gpu_hook(optimizer, args, kwargs):
                self._h2d_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._h2d_stream):
                    for param in _param_generator(optimizer):
                        gpu_param = self.cpu_copys_map_gpu_param[param]
                        if get_fp8_cpu_offload_proxy_info(gpu_param) is not None:
                            continue
                        gpu_param.data.copy_(param.data, non_blocking=True)
                self._h2d_stream.record_event().wait(torch.cuda.current_stream())

            return param_copy_back_gpu_hook

        def fp32_param_copy_back_gpu_hook_closure():
            def fp32_param_copy_back_gpu_hook(optimizer, args, kwargs):
                for group in self.param_groups:
                    for param in group["params"]:
                        if param in self.gpu_params_map_cpu_copy:
                            # Skip if the param is offloaded to GPU, it has been
                            # copied back in the previous hook.
                            continue

                        if param in self.param_to_fp32_param:
                            fp32_param = self.param_to_fp32_param[param]
                            param.data.copy_(fp32_param.data)

            return fp32_param_copy_back_gpu_hook

        for optimizer in self.sub_optimizers:
            if optimizer is not self.gpu_optimizer:
                optimizer.register_step_post_hook(param_copy_back_gpu_hook_closure())
            elif self.param_update_in_fp32:
                optimizer.register_step_post_hook(fp32_param_copy_back_gpu_hook_closure())

    def step(self, closure=None):
        """
        Override the step method to perform the following operations:
            1. Sync the HDO param_groups to sub-optimizers.
            2. Sync the grads from GPU to CPU.
            3. Step the sub-optimizers.
            4. Sync the sub-optimizers state to HDO.
        """
        # Sync param_groups to sub-optimizers before each step to make sure
        # the lr, wd, etc. are up-to-date.
        self._sync_hdo_param_groups_to_sub_optimizers()

        if self.grad_streaming:
            self._d2h_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._d2h_stream):
                self._set_gpu_fp32_grads()
            if self.gpu_optimizer:
                self.gpu_optimizer.step(closure)
            self._streamed_cpu_steps(closure)
            if hasattr(self, "_writeback_fp8_cpu_offload_masters"):
                self._writeback_fp8_cpu_offload_masters()
            self._sync_sub_optimizers_state_to_hdo()
            return

        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            self._set_sub_optimizer_grads()

        # Step the sub-optimizers.
        if self.gpu_optimizer:
            self.gpu_optimizer.step(closure)

        for cpu_optimizer in self.cpu_optimizers:
            d2h_event = self._cpu_optimizer_map_data_event.pop(cpu_optimizer, None)
            if d2h_event is not None:
                d2h_event.synchronize()
            cpu_optimizer.step(closure)

        self._writeback_fp8_cpu_offload_masters()

        # Sync state and param_groups to HDO after each step.
        # NOTE: It is possible for the optimizer to change the properties
        #   in param_groups.
        self._sync_sub_optimizers_state_to_hdo()

    def _init_sub_optimizers(self):
        (
            self.cpu_param_groups,
            self.gpu_param_groups,
            self.gpu_params_map_cpu_copy,
            self.cpu_copys_map_gpu_param,
            self.param_to_fp32_param,
        ) = self._get_sub_optimizer_param_groups(self.offload_fraction)
        self.param_to_inner_param = {}
        self.inner_param_to_orig_param = {}
        for group in self.param_groups:
            for param in group["params"]:
                if param in self.param_to_fp32_param:
                    inner_param = self.param_to_fp32_param[param]
                elif param in self.gpu_params_map_cpu_copy:
                    inner_param = self.gpu_params_map_cpu_copy[param]
                else:
                    inner_param = param
                self.param_to_inner_param[param] = inner_param
                self.inner_param_to_orig_param[inner_param] = param
        self.fp32_param_to_orig_param = {v: k for k, v in self.param_to_fp32_param.items()}

        self.cpu_optimizers = []
        if self.grad_streaming:
            self.cpu_optimizers = self.build_cpu_optimizer_list_bucketed(
                self.cpu_optimizer_cls,
                self.cpu_param_groups,
                self.grad_streaming_bucket_bytes,
            )
        elif self.overlap_cpu_optimizer_d2h_h2d:
            self.cpu_optimizers = self.build_cpu_optimizer_list(
                self.cpu_optimizer_cls, self.cpu_param_groups
            )
        elif len(self.cpu_param_groups) > 0:
            self.cpu_optimizers = [self.cpu_optimizer_cls(self.cpu_param_groups)]

        if len(self.gpu_param_groups) > 0:
            self.gpu_optimizer = self.gpu_optimizer_cls(self.gpu_param_groups)
        else:
            self.gpu_optimizer = None

        self.cpu_copy_map_grad: Dict[torch.Tensor, torch.Tensor] = defaultdict(torch.Tensor)
        self._d2h_stream = torch.cuda.current_stream()
        self._h2d_stream = torch.cuda.current_stream()
        if self.overlap_cpu_optimizer_d2h_h2d:
            self._d2h_stream = torch.cuda.Stream()
            self._h2d_stream = torch.cuda.Stream()
        self._cpu_optimizer_map_data_event = dict()
        self._gs_waves = None
        self._gs_arenas = []
        self._gs_events = [None, None]

        self._register_param_copy_back_gpu_hook()

    @staticmethod
    def build_cpu_optimizer_list_bucketed(cpu_optimizer_cls, cpu_param_groups, bucket_bytes):
        """Build CPU optimizers at transfer-wave granularity."""
        alignment = 16
        cpu_optimizers = []
        for group in cpu_param_groups:
            group_defaults = group.copy()
            params = group_defaults.pop("params")
            if isinstance(params, torch.Tensor):
                params = [params]
            current_params = []
            current_bytes = 0
            for param in params:
                nbytes = (
                    param.numel() * param.element_size() + alignment - 1
                ) // alignment * alignment
                if current_params and current_bytes + nbytes > bucket_bytes:
                    sub_group = group_defaults.copy()
                    sub_group["params"] = current_params
                    cpu_optimizers.append(cpu_optimizer_cls([sub_group]))
                    current_params = []
                    current_bytes = 0
                current_params.append(param)
                current_bytes += nbytes
            if current_params:
                sub_group = group_defaults.copy()
                sub_group["params"] = current_params
                cpu_optimizers.append(cpu_optimizer_cls([sub_group]))
        return cpu_optimizers

    @staticmethod
    def build_cpu_optimizer_list(cpu_optimizer_cls, cpu_param_groups):
        """Build several cpu optimizers to enable overlap. Currently we naively
        assign each parameter to an individual optimizer.

        Args:
            cpu_optimizer_cls (Type[torch.optim.Optimizer]): A torch optimizer class
            cpu_param_groups (List[Dict[str, Any]]): The CPU parameter groups
        """
        cpu_optimizers = []

        if len(cpu_param_groups) == 0:
            return cpu_optimizers

        for group in cpu_param_groups:
            group_defaults = group.copy()
            params = group_defaults.pop("params")
            if isinstance(params, torch.Tensor):
                params = [params]
            for param in params:
                _cpu_param_group = group_defaults.copy()
                _cpu_param_group["params"] = [param]
                cpu_optimizers.append(cpu_optimizer_cls([_cpu_param_group]))
        return cpu_optimizers

    def _rebuild_arena_master_view(self, orig_param, shaped_like):
        """Return the existing arena master view for a rebuilt parameter."""
        if not self.contiguous_state:
            return None
        arenas = self.state_arenas
        arena_slices = getattr(self, "_arena_slices", None)
        if arenas is None or not arena_slices:
            return None
        entry = arena_slices.get(orig_param)
        if entry is None:
            return None
        gbuf_idx, start, numel = entry
        if shaped_like.numel() != numel:
            return None
        return arenas[gbuf_idx]["param"][start : start + numel].view(shaped_like.shape)

    def _get_sub_optimizer_param_groups(self, offload_fraction: float):
        params = []
        for group in self.param_groups:
            params.extend(group["params"])
        params_total_numel = sum([get_fp8_cpu_offload_proxy_numel(param) for param in params])
        gpu_params_total_numel = sum(
            [get_fp8_cpu_offload_proxy_numel(param) for param in params if param.is_cuda]
        )
        cpu_params_total_numel = params_total_numel - gpu_params_total_numel
        offload_threshold = gpu_params_total_numel * offload_fraction
        offload_params_numel = 0
        cpu_param_groups = []
        gpu_param_groups = []
        gpu_params_map_cpu_copy = {}
        cpu_copys_map_gpu_param = {}
        param_to_fp32_param = {}
        for group in self.param_groups:
            gpu_group = group.copy()
            cpu_group = group.copy()
            gpu_group["params"] = []
            cpu_group["params"] = []
            for param in group["params"]:
                orig_param = param
                cpu_copy = False
                if get_fp8_cpu_offload_proxy_info(param) is not None:
                    param = self._build_fp8_cpu_offload_master_param_shard(param)
                    offload_params_numel += param.numel()
                    cpu_copy = True
                elif offload_params_numel < offload_threshold and param.is_cuda:
                    arena_view = self._rebuild_arena_master_view(orig_param, param)
                    if arena_view is not None:
                        arena_view.copy_(param.detach())
                        param = arena_view
                    else:
                        param = param.detach().clone().cpu()
                        if not self.contiguous_state:
                            param = param.pin_memory()
                    offload_params_numel += param.numel()
                    cpu_copy = True
                if self.param_update_in_fp32:
                    if param.dtype != torch.float32:
                        param = param.detach().clone().float()
                        if cpu_copy and self.pin_cpu_params and not self.contiguous_state:
                            param = param.pin_memory()
                    param_to_fp32_param[orig_param] = param

                if cpu_copy:
                    gpu_params_map_cpu_copy[orig_param] = param
                    cpu_copys_map_gpu_param[param] = orig_param

                if param.is_cuda:
                    gpu_group["params"].append(param)
                else:
                    cpu_group["params"].append(param)
            if len(gpu_group["params"]) != 0:
                gpu_param_groups.append(gpu_group)
            if len(cpu_group["params"]) != 0:
                cpu_param_groups.append(cpu_group)

        return (
            cpu_param_groups,
            gpu_param_groups,
            gpu_params_map_cpu_copy,
            cpu_copys_map_gpu_param,
            param_to_fp32_param,
        )

    def _sync_sub_optimizers_state_to_hdo(self):
        """
        Update HDO state attribute to sub-optimizers.
        """

        # optimizer.state:
        # {
        #    torch.nn.Parameter: {
        #        str: Any,
        #    },
        #    ...
        # }
        new_state = defaultdict(dict)
        for optimizer in self.sub_optimizers:
            for param in optimizer.state:
                orig_param = self.inner_param_to_orig_param[param]
                new_state[orig_param] = optimizer.state[param]
                if self.param_update_in_fp32:
                    new_state[orig_param]["master_param"] = param
        self.state = new_state

    def _sync_hdo_state_to_sub_optimizers(self):
        for optimizer in self.sub_optimizers:
            new_state = defaultdict(dict)
            for group in optimizer.param_groups:
                for param in group["params"]:
                    orig_param = self.inner_param_to_orig_param[param]
                    new_state[param] = self.state[orig_param]
            optimizer.state = new_state
        self._update_fp32_params_by_new_state()
        self._move_new_state_to_right_device()

    def _sync_hdo_param_groups_to_sub_optimizers(self):
        """Sync HDO new param_groups attribute (e.g. lr, wd, etc.) to sub-optimizers."""
        param_in_param_group_index = {}
        for i, group in enumerate(self.param_groups):
            for p_id, param in enumerate(group["params"]):
                inner_param = self.param_to_inner_param[param]
                param_in_param_group_index[inner_param] = (i, p_id)

        for optimizer in self.sub_optimizers:
            new_param_groups = []
            for group in optimizer.param_groups:
                new_group = group.copy()
                # After sync-up the sub-optimizer last update, we need to sync-up the
                # HDO new param_groups attributes to the sub-optimizer.
                assert len(group["params"]) > 0, "param_groups should not be empty"
                group_id, _ = param_in_param_group_index[group["params"][0]]
                update_group_attrs = self.param_groups[group_id].copy()
                del update_group_attrs["params"]
                new_group.update(update_group_attrs)

                new_param_groups.append(new_group)
            optimizer.param_groups = new_param_groups

    def _move_new_state_to_right_device(self):
        for optimizer in self.sub_optimizers:
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    orig_param = self.inner_param_to_orig_param.get(param, param)
                    if isinstance(optimizer, self.defaults["cpu_optimizer_cls"]):
                        v = self._move_cpu_optimizer_state_tensor(param, k, v)
                    else:
                        v = v.to("cuda")
                    self.state[orig_param][k] = state[k] = v

    @staticmethod
    def _move_cpu_optimizer_state_tensor(param, key, value):
        """Move CPU optimizer state to CPU, matching Adam moments to the master param dtype."""
        if key in _CPU_ADAM_STATE_KEYS and value.is_floating_point():
            # CPU Adam kernels require moment buffers to match the CPU master param dtype.
            return value.to(device="cpu", dtype=param.dtype)
        return value.to("cpu")

    def _update_fp32_params_by_new_state(self):
        if not self.param_update_in_fp32:
            return
        for param, v in self.state.items():
            fp32_param = self.param_to_fp32_param.get(param)
            if fp32_param is not None:
                fp32_param.data.copy_(v["master_param"])

    def update_fp32_param_by_new_param(self):
        """
        Update the fp32 parameters by the new parameters.
        """
        for param, fp32_param in self.param_to_fp32_param.items():
            if get_fp8_cpu_offload_proxy_info(param) is not None:
                model_param_shard = self._get_high_prec_param_shard_for_fp8_proxy(param)
                fp32_param.data.copy_(
                    model_param_shard.to(device=fp32_param.device, dtype=fp32_param.dtype)
                )
            else:
                fp32_param.data.copy_(param)

    @torch.no_grad()
    def materialize_state_arenas(self, layout):
        """Rebase CPU optimizer state onto contiguous per-buffer arenas."""
        if not self.contiguous_state:
            return False

        def fail(reason):
            self._arena_slices = None
            import warnings

            warnings.warn(f"HDO contiguous state arenas disabled: {reason}")
            return False

        if self.gpu_optimizer is not None:
            return fail("not fully offloaded (gpu sub-optimizer present)")
        for optimizer in self.cpu_optimizers:
            if not (
                isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW))
                or type(optimizer).__name__ == "DeepSpeedCPUAdam"
            ):
                return fail(f"unsupported cpu optimizer {type(optimizer).__name__}")

        entries = layout["entries"]
        slice_map = {}
        for orig_param, gbuf_idx, start, numel in entries:
            inner = self.param_to_inner_param.get(orig_param)
            if inner is None or inner.is_cuda or inner.dtype != torch.float32:
                return fail("param without a CPU fp32 master")
            if inner.numel() != numel:
                return fail(f"shard numel mismatch ({inner.numel()} vs layout {numel})")
            slice_map[orig_param] = (gbuf_idx, start, numel)

        inner_covered = {self.param_to_inner_param[param] for param in slice_map}
        for optimizer in self.cpu_optimizers:
            for param in _param_generator(optimizer):
                if param not in inner_covered:
                    return fail("cpu sub-optimizer param not covered by layout")

        numels = list(layout["gbuf_numels_unpadded"])
        arenas = self.state_arenas
        if arenas is None or [arena["param"].numel() for arena in arenas] != numels:
            arenas = [
                {
                    key: torch.zeros(
                        (numel,), dtype=torch.float32, pin_memory=self.pin_cpu_params
                    )
                    for key in ("param", "exp_avg", "exp_avg_sq")
                }
                for numel in numels
            ]
            self.state_arenas = arenas

        for orig_param, gbuf_idx, start, numel in entries:
            inner = self.param_to_inner_param[orig_param]
            view = arenas[gbuf_idx]["param"][start : start + numel].view(inner.shape)
            if inner.data_ptr() != view.data_ptr():
                view.copy_(inner.data)
                inner.data = view

        self._arena_slices = slice_map
        self._reseed_arena_state_views()
        self._sync_sub_optimizers_state_to_hdo()
        return True

    @torch.no_grad()
    def _reseed_arena_state_views(self):
        """Point each CPU sub-optimizer state entry at its arena view."""
        for optimizer in self.cpu_optimizers:
            is_deepspeed = type(optimizer).__name__ == "DeepSpeedCPUAdam"
            for param in _param_generator(optimizer):
                orig_param = self.inner_param_to_orig_param[param]
                gbuf_idx, start, numel = self._arena_slices[orig_param]
                state = optimizer.state[param]
                step = state.get("step")
                if step is None:
                    step = 0 if is_deepspeed else torch.zeros((), dtype=torch.float32)
                elif is_deepspeed and isinstance(step, torch.Tensor):
                    step = int(step.item())
                elif not is_deepspeed and not isinstance(step, torch.Tensor):
                    step = torch.tensor(float(step), dtype=torch.float32)
                for key in ("exp_avg", "exp_avg_sq"):
                    view = self.state_arenas[gbuf_idx][key][
                        start : start + numel
                    ].view(param.shape)
                    old = state.get(key)
                    if isinstance(old, torch.Tensor):
                        if old.data_ptr() == view.data_ptr():
                            continue
                        if old.shape == view.shape:
                            view.copy_(old.to(view.dtype))
                    state[key] = view
                state["step"] = step

    def state_arenas_intact(self):
        """Whether all CPU masters and moment buffers remain arena views."""
        arenas = self.state_arenas
        slice_map = self._arena_slices
        if arenas is None or slice_map is None or self.gpu_optimizer is not None:
            return False
        element_size = 4
        try:
            checked = 0
            for orig_param, (gbuf_idx, start, numel) in slice_map.items():
                inner = self.param_to_inner_param.get(orig_param)
                if inner is None:
                    return False
                base = arenas[gbuf_idx]
                if inner.data_ptr() != base["param"].data_ptr() + start * element_size:
                    return False
                state = self.state.get(orig_param)
                if not state:
                    return False
                for key in ("exp_avg", "exp_avg_sq"):
                    tensor = state.get(key)
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or tensor.data_ptr() != base[key].data_ptr() + start * element_size
                    ):
                        return False
                checked += 1
            total = sum(
                1 for optimizer in self.cpu_optimizers for _ in _param_generator(optimizer)
            )
            return checked == total
        except (KeyError, IndexError, AttributeError):
            return False

    def _register_load_state_dict_hooks(self):
        def pre_load_state_dict_hook(self, state_dict):
            """
            Pre-load state dictionary hook to prevent loss of precision in
            mixed-precision training.

            When loading a state dictionary with `torch.load_state_dict`,
            optimizer states are reset and cast from `float32` to `bfloat16`/`float16`,
            potentially losing precision. This hook replaces parameters with
            their `float32` copies to mitigate this issue.

            Args:
                state_dict (dict): The state dictionary to be loaded.

            Returns:
                dict: The modified state dictionary with `float32` parameters.
            """
            if not self.param_update_in_fp32:
                return state_dict

            new_state = {}
            for param, v in self.state.items():
                param = self.param_to_fp32_param.get(param, param)
                new_state[param] = v
            self.state = new_state

            for group in self.param_groups:
                for i, param in enumerate(group["params"]):
                    group["params"][i] = self.param_to_fp32_param.get(param, param)

            return state_dict

        self.register_load_state_dict_pre_hook(pre_load_state_dict_hook)

        def post_load_state_dict_hook(self):
            # 1. Replace the temporarily replaced fp32 parameters back. Please
            # refer to the documentation in `pre_load_state_dict_hook`.
            if self.param_update_in_fp32:
                new_state = {}
                for param, v in self.state.items():
                    orig_param = self.fp32_param_to_orig_param.get(param, param)
                    new_state[orig_param] = v
                self.state = new_state

                for group in self.param_groups:
                    for i, param in enumerate(group["params"]):
                        group["params"][i] = self.fp32_param_to_orig_param.get(param, param)

            # 2. After loading state_dict, the parameters may change, and we need to
            # reinitialize the sub-optimizers to regenerate the new parameters and
            # cpu copy pairs.
            self._init_sub_optimizers()
            self._sync_hdo_param_groups_to_sub_optimizers()
            self._sync_hdo_state_to_sub_optimizers()

        self.register_load_state_dict_post_hook(post_load_state_dict_hook)

    def zero_grad(self, set_to_none: bool = True):
        """
        Zero or zero to none the gradients of all the parameters in the model.
        """
        super(HybridDeviceOptimizer, self).zero_grad(set_to_none)
        for group in self.param_groups:
            for param in group["params"]:
                if hasattr(param, "decoupled_grad"):
                    if set_to_none:
                        param.decoupled_grad = None
                    else:
                        param.decoupled_grad.zero_()

    def dummy_step(self):
        """
        The dummy step can be used to initialize the potential optimizer.state,
        which can solve the problem of checkpoint loading for an inplace operation
        such as loading a torch distributed checkpoint, for example.

        Steps sub-optimizers directly so synthetic gradients only occupy one
        optimizer's parameters at a time.
        """
        self._sync_hdo_param_groups_to_sub_optimizers()
        for optimizer in self.sub_optimizers:
            for param in _param_generator(optimizer):
                param.grad = torch.randn_like(param)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.zero_grad()

    @property
    def sub_optimizers(self):
        """
        Return the list of sub-optimizers.
        """
        if self.gpu_optimizer is not None:
            return self.cpu_optimizers + [self.gpu_optimizer]
        return self.cpu_optimizers
