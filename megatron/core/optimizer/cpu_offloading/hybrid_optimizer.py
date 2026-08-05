# Copyright (c) 2025, NVIDIA CORPORATION and Alibaba PAI. All rights reserved.
from collections import defaultdict
from typing import Dict
from ..muon import Muon

import torch

from megatron.core.fp8_utils import (
    dequantize_fp8_tensor,
    get_fp8_cpu_offload_proxy_info,
    get_fp8_cpu_offload_proxy_numel,
)

_CPU_ADAM_STATE_KEYS = {"exp_avg", "exp_avg_sq", "adamw_exp_avg", "adamw_exp_avg_sq"}

# Device-side staging budget (bytes of FP32 master data) for one wave of the
# FP8 CPU-offload master write-back.
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
        **kwargs
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
        # Bucketed gradient streaming: replace the persistent full-size pinned
        # CPU grad mirror (4 B/param host RAM) with two bounded pinned staging
        # arenas consumed wave-by-wave. It needs the per-param cpu_optimizers
        # granularity and the dedicated copy streams, so it implies
        # overlap_cpu_optimizer_d2h_h2d.
        self.grad_streaming = grad_streaming
        self.grad_streaming_bucket_bytes = grad_streaming_bucket_bytes
        self.overlap_cpu_optimizer_d2h_h2d = overlap_cpu_optimizer_d2h_h2d or grad_streaming
        self.param_update_in_fp32 = param_update_in_fp32
        # Arenas are materialized by DistributedOptimizer (it owns the dp_zero
        # layout); while not intact, every consumer falls back to the regular
        # paths. Pre-migration masters stay unpinned so freeing them returns
        # memory to the OS instead of the pinned-memory cache.
        self.contiguous_state = contiguous_state
        self.state_arenas = None
        self._arena_slices = None
        self.sub_optimizer_kwargs = kwargs
        # Rank-invariant FP8 master write-back plan; installed by the owner of
        # the grad buffers (DistributedOptimizer). See
        # set_fp8_cpu_offload_writeback_plan().
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
        if self.pin_cpu_params and not self.contiguous_state:
            master_param = master_param.pin_memory()
        # Release the CPU bf16 high-precision init copy NOW — the master shard
        # has already been built from it, so the init copy is dead weight.
        # Without this, every FP8 weight keeps a CPU bf16 duplicate for the
        # whole run, inflating host RAM by ~param_size per rank (e.g. ~240GB
        # per node for full-size Kimi K2.6 8 ranks), which causes host OOM.
        model_param = info.blockwise_fp8_model_param
        if hasattr(model_param, "clear_high_precision_init_val"):
            model_param.clear_high_precision_init_val()
        return master_param

    def set_fp8_cpu_offload_writeback_plan(
        self, model_params, data_parallel_group, stage_bytes=_FP8_WRITEBACK_STAGE_BYTES
    ):
        """Register the rank-invariant plan for the FP8 master write-back.

        Casting CPU FP32 master shards back into blockwise-FP8 model params
        reduces amaxes over the data-parallel group, i.e. it is a COLLECTIVE
        whose message shape is derived from the *list of model params passed in*
        (see TE's cast_master_weights_to_fp8: "Each rank has a shard of the
        master weights (possibly empty) and a full copy of the model weights").
        Every rank must therefore pass the same params in the same order and
        hand in None for the shards it does not own. A rank only ever sees the
        params whose grad-buffer slice it owns, so the full ordered list has to
        come from the caller (DistributedOptimizer, which owns the buffers).

        The list is chunked into waves of at most `stage_bytes` of FP32 master
        data so that staging host masters back to the device stays bounded --
        offload exists to save device memory, so materializing every master at
        once would defeat it. Wave boundaries are computed from the model params
        alone, hence identical on every rank.
        """
        model_params = list(model_params)
        waves = []
        cur_wave = []
        cur_bytes = 0
        for model_param in model_params:
            # FP32 upper bound: the shard this rank owns is at most the param.
            nbytes = model_param.numel() * 4
            if cur_wave and cur_bytes + nbytes > stage_bytes:
                waves.append(cur_wave)
                cur_wave = []
                cur_bytes = 0
            cur_wave.append(model_param)
            cur_bytes += nbytes
        if cur_wave:
            waves.append(cur_wave)

        self._fp8_writeback_waves = waves
        self._fp8_writeback_group = data_parallel_group

    def _collect_fp8_offload_master_shards(self):
        """Map blockwise-FP8 model param -> (CPU FP32 master shard, start offset).

        Rebuilt per step on purpose: load_state_dict re-runs
        _init_sub_optimizers(), which hands out new master tensors.
        """
        shards = {}
        for cpu_param, gpu_param in self.cpu_copys_map_gpu_param.items():
            info = get_fp8_cpu_offload_proxy_info(gpu_param)
            if info is not None:
                shards[info.blockwise_fp8_model_param] = (cpu_param, info.start_offset)
        return shards

    def _writeback_fp8_cpu_offload_masters(self):
        """Quantize all CPU FP32 master shards back into their FP8 model params.

        Batched per wave, so the number of amax all-reduces and the size of each
        one depend only on the rank-invariant plan. Issuing this per param (or
        per sub-optimizer, as a step post-hook would) makes both depend on which
        grad-buffer slice the rank happens to own, and NCCL then deadlocks with
        no error message.
        """
        from megatron.core.fp8_utils import quantize_param_shard

        shards = self._collect_fp8_offload_master_shards()
        if self._fp8_writeback_waves is None:
            assert not shards, (
                "FP8 CPU-offload master shards exist but no write-back plan was "
                "registered; set_fp8_cpu_offload_writeback_plan() must be called "
                "after building the optimizer or the FP8 weights never get updated."
            )
            return

        for wave in self._fp8_writeback_waves:
            main_params = []
            start_offsets = []
            staged = []
            for model_param in wave:
                entry = shards.get(model_param)
                if entry is None:
                    # Not this rank's shard: join the collective contributing no
                    # amax. TE skips the copy for a None master.
                    main_params.append(None)
                    start_offsets.append(None)
                    continue
                master, start_offset = entry
                if not master.is_cuda:
                    master = master.to(model_param.device, non_blocking=True)
                    staged.append(master)
                main_params.append(master)
                start_offsets.append(start_offset)

            quantize_param_shard(wave, main_params, start_offsets, self._fp8_writeback_group)
            del staged

    def _set_gpu_fp32_grads(self):
        """fp32 grads for the non-offloaded (GPU optimizer) params."""
        if not self.param_update_in_fp32:
            return
        for param in self.param_to_fp32_param:
            if param in self.gpu_params_map_cpu_copy:
                # Skip if the param is offloaded to CPU, it should be handled
                # in the following part.
                continue
            fp32_param = self.param_to_fp32_param[param]
            grad = getattr(param, "decoupled_grad", param.grad)
            if grad is not None:
                fp32_param.grad = grad.to(fp32_param.dtype)
                fp32_param.requires_grad = True
            else:
                fp32_param.requires_grad = False

    def _set_sub_optimizer_grads(self):
        self._set_gpu_fp32_grads()

        # Sync the grads from GPU to CPU.
        for optimizer in self.cpu_optimizers:
            for param in _param_generator(optimizer):
                gpu_param = self.cpu_copys_map_gpu_param[param]
                grad = getattr(gpu_param, "decoupled_grad", gpu_param.grad)
                if grad is None:
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

    # ------------------------------------------------------------------
    # Bucketed gradient streaming
    #
    # Gradients are consume-once data: their CPU lifetime is only
    # "D2H copy done -> consumed by this param's Adam update". Instead of one
    # persistent pinned mirror per param (4 B/param host RAM, e.g. ~503 GB per
    # node on Kimi K2.7 at offload 1.0), stage them through TWO pinned arenas
    # of at most max(bucket_bytes, largest param) each: wave k stages into
    # arena k%2 while the CPU consumes wave k-1 (producer-consumer pipeline,
    # gated by one CUDA event per wave). Arena k%2 is only overwritten by wave
    # k+2, strictly after wave k's optimizer steps finished on this thread.
    # The arithmetic (values, kernel, per-param update order) is unchanged, so
    # results are bit-exact vs the mirror path.
    # ------------------------------------------------------------------

    def _build_grad_streaming_plan(self):
        """Partition cpu_optimizers (in order) into waves of <= bucket_bytes.

        Returns (waves, arena_bytes) where each wave is a list of
        (optimizer, [(param, byte_offset, nbytes), ...]) and byte offsets are
        16-byte aligned within the wave's arena.
        """
        ALIGN = 16
        waves = []
        cur_wave = []
        cur_bytes = 0
        arena_bytes = 0
        for optimizer in self.cpu_optimizers:
            opt_params = []
            opt_bytes = 0
            for param in _param_generator(optimizer):
                nbytes = param.numel() * param.element_size()
                aligned = (nbytes + ALIGN - 1) // ALIGN * ALIGN
                opt_params.append((param, opt_bytes, nbytes))
                opt_bytes += aligned
            # An optimizer's params never split across waves (its step consumes
            # all of them at once).
            if cur_wave and cur_bytes + opt_bytes > self.grad_streaming_bucket_bytes:
                waves.append(cur_wave)
                arena_bytes = max(arena_bytes, cur_bytes)
                cur_wave = []
                cur_bytes = 0
            opt_params = [(p, cur_bytes + off, n) for (p, off, n) in opt_params]
            cur_wave.append((optimizer, opt_params))
            cur_bytes += opt_bytes
        if cur_wave:
            waves.append(cur_wave)
            arena_bytes = max(arena_bytes, cur_bytes)
        return waves, arena_bytes

    def _ensure_grad_streaming_buffers(self):
        if getattr(self, "_gs_waves", None) is not None:
            return
        self._gs_waves, arena_bytes = self._build_grad_streaming_plan()
        self._gs_arenas = [
            torch.empty(arena_bytes, dtype=torch.uint8, device="cpu",
                        pin_memory=self.pin_cpu_grads)
            for _ in range(2)
        ] if arena_bytes > 0 else []
        self._gs_events = [None, None]

    def _gs_issue_wave(self, w):
        """Producer: enqueue async D2H copies of wave w's grads into arena w%2."""
        if w >= len(self._gs_waves):
            return
        arena = self._gs_arenas[w % 2]
        with torch.cuda.stream(self._d2h_stream):
            for optimizer, opt_params in self._gs_waves[w]:
                for param, byte_off, nbytes in opt_params:
                    gpu_param = self.cpu_copys_map_gpu_param[param]
                    grad = getattr(gpu_param, "decoupled_grad", gpu_param.grad)
                    param.requires_grad = False
                    if grad is None:
                        param.grad = None
                        continue
                    view = (
                        arena[byte_off : byte_off + nbytes]
                        .view(param.dtype)
                        .view(param.shape)
                    )
                    view.copy_(grad, non_blocking=True)
                    param.grad = view
            self._gs_events[w % 2] = self._d2h_stream.record_event()

    def _streamed_cpu_steps(self, closure=None):
        """Consumer loop: wait wave event, step its optimizers, refill arena."""
        import os
        import time

        debug = os.environ.get("GRAD_STREAMING_DEBUG", "0") == "1"
        rank = os.environ.get("RANK", "?")

        def dbg(msg):
            if debug:
                print(f"[grad-stream][rank {rank}] {msg}", flush=True)

        self._ensure_grad_streaming_buffers()
        if not self._gs_waves:
            return
        n = len(self._gs_waves)
        t_begin = time.monotonic()
        sync_s = step_s = step_mx = 0.0
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        self._gs_issue_wave(0)
        self._gs_issue_wave(1)
        for w, wave in enumerate(self._gs_waves):
            t0 = time.monotonic()
            event = self._gs_events[w % 2]
            if event is not None:
                event.synchronize()
            t1 = time.monotonic()
            for optimizer, _ in wave:
                if isinstance(optimizer, Muon):
                    optimizer.step(self.cpu_copys_map_gpu_param)
                else:
                    optimizer.step(closure)
            t2 = time.monotonic()
            # Wave w fully consumed on this thread -> arena w%2 is free for
            # wave w+2. (Master copy-back reads masters, not grads.)
            self._gs_issue_wave(w + 2)
            sync_s += t1 - t0
            step_s += t2 - t1
            if t2 - t1 > step_mx:
                step_mx = t2 - t1
        # one line per optimizer chain per iteration
        arena = (
            f" arena {len(self._gs_arenas[0]) / 2**30:.2f}GiBx2" if self._gs_arenas else ""
        )
        dbg(
            f"{n} waves in {time.monotonic() - t_begin:.2f}s: sync {sync_s:.2f}s "
            f"step {step_s:.2f}s (max {step_mx:.2f}s){arena}"
        )

    def _register_param_copy_back_gpu_hook(self):
        def param_copy_back_gpu_hook_closure():
            def param_copy_back_gpu_hook(optimizer, args, kwargs):
                self._h2d_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._h2d_stream):
                    for param in _param_generator(optimizer):
                        gpu_param = self.cpu_copys_map_gpu_param[param]
                        if get_fp8_cpu_offload_proxy_info(gpu_param) is not None:
                            # FP8 masters are written back once per step() by
                            # _writeback_fp8_cpu_offload_masters(): the cast is a
                            # collective, so it must not be issued per
                            # sub-optimizer (the param set is rank-dependent).
                            continue
                        gpu_param.data.copy_(param.data, non_blocking=True)
                self._d2h_stream.record_event().wait(torch.cuda.current_stream())

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
            # Bucketed gradient streaming: no persistent grad mirrors; grads
            # are staged wave-by-wave through two bounded pinned arenas.
            self._d2h_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._d2h_stream):
                self._set_gpu_fp32_grads()
            if self.gpu_optimizer:
                self.gpu_optimizer.step(closure)
            self._streamed_cpu_steps(closure)
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
            if isinstance(cpu_optimizer, Muon):
                cpu_optimizer.step(self.cpu_copys_map_gpu_param)
            else:
                cpu_optimizer.step(closure)

        # All CPU masters are final now: one rank-consistent FP8 write-back.
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
            # Wave-granularity optimizers: one CPU optimizer per <=bucket_bytes
            # group of params. Per-PARAM granularity (below) constructs O(1000)
            # optimizer instances per rank; with DeepSpeed CPUAdam every
            # instance re-enters the extension loader (file-locked across the
            # 8 local ranks), which serializes init into tens of minutes.
            self.cpu_optimizers = self.build_cpu_optimizer_list_bucketed(
                self.cpu_optimizer_cls, self.cpu_param_groups,
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
        # Grad-streaming plan is param-identity based; rebuild lazily after any
        # re-init (e.g. load_state_dict replaces the CPU master params).
        self._gs_waves = None
        self._gs_arenas = []
        self._gs_events = [None, None]

        self._register_param_copy_back_gpu_hook()

    @staticmethod
    def build_cpu_optimizer_list_bucketed(cpu_optimizer_cls, cpu_param_groups, bucket_bytes):
        """One CPU optimizer per <=bucket_bytes run of params (in order, never
        crossing a param-group boundary — groups carry distinct hyperparams).
        Used by grad streaming: each optimizer becomes one pipeline wave."""
        ALIGN = 16
        cpu_optimizers = []
        for group in cpu_param_groups:
            group_defaults = group.copy()
            params = group_defaults.pop("params")
            if isinstance(params, torch.Tensor):
                params = [params]
            cur, cur_bytes = [], 0
            for param in params:
                nbytes = (param.numel() * param.element_size() + ALIGN - 1) // ALIGN * ALIGN
                if cur and cur_bytes + nbytes > bucket_bytes:
                    sub = group_defaults.copy()
                    sub["params"] = cur
                    cpu_optimizers.append(cpu_optimizer_cls([sub]))
                    cur, cur_bytes = [], 0
                cur.append(param)
                cur_bytes += nbytes
            if cur:
                sub = group_defaults.copy()
                sub["params"] = cur
                cpu_optimizers.append(cpu_optimizer_cls([sub]))
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
        """Arena view for a rebuilt CPU master (e.g. on load_state_dict).

        Binding masters straight to their arena slices avoids the rebuild
        transient where every temp master coexists until migration
        (+4 B/param/rank host RAM). Returns None on the first build (no arena
        layout yet) or on any mismatch — callers then allocate normally and
        materialize_state_arenas() migrates as before."""
        if not self.contiguous_state:
            return None
        arenas = self.state_arenas
        slices = getattr(self, "_arena_slices", None)
        if arenas is None or not slices:
            return None
        entry = slices.get(orig_param)
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
        gpu_params_total_numel = sum([get_fp8_cpu_offload_proxy_numel(param) for param in params if param.is_cuda])
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
                    cpu_master_param = self._build_fp8_cpu_offload_master_param_shard(param)
                    view = self._rebuild_arena_master_view(orig_param, cpu_master_param)
                    if view is not None:
                        # temp master dies this iteration -> transient is one
                        # param, not the whole model
                        view.copy_(cpu_master_param)
                        cpu_master_param = view
                    param = cpu_master_param
                    offload_params_numel += param.numel()
                    cpu_copy = True
                elif offload_params_numel < offload_threshold and param.is_cuda:
                    view = self._rebuild_arena_master_view(orig_param, param)
                    if view is not None:
                        view.copy_(param.detach())  # D2H straight into the pinned arena
                        param = view
                    else:
                        param = param.detach().clone().cpu()
                        if not self.contiguous_state:
                            param = param.pin_memory()
                    offload_params_numel += param.numel()
                    cpu_copy = True
                if self.param_update_in_fp32:
                    # In FP8 case, the passed in param_groups is fp32 shard main param, so just do a self-reference
                    if param.dtype != torch.float32:
                        param = param.detach().clone().float()
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

    @staticmethod
    def _move_cpu_optimizer_state_tensor(param, key, value):
        if key in _CPU_ADAM_STATE_KEYS and value.is_floating_point():
            # CPU Adam kernels expect moment buffers to match the CPU master param dtype.
            return value.to(device="cpu", dtype=param.dtype)
        return value.to("cpu")

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

    def _update_fp32_params_by_new_state(self):
        if not self.param_update_in_fp32:
            return
        for param, v in self.state.items():
            fp32_param = self.param_to_fp32_param[param]
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
        """Rebase the CPU optimizer state onto contiguous per-buffer arenas.

        `layout` is provided by DistributedOptimizer (the owner of the grad
        buffer geometry) and describes the dp_zero world (unpadded) layout:

            layout = {
                "gbuf_numels_unpadded": [numel per grad buffer],
                # bucket order, so the transient during master migration is
                # bounded to one param at a time:
                "entries": [(orig_param, gbuf_idx, world_start, numel), ...],
            }

        For every entry the existing per-param CPU fp32 master is copied into
        its arena slice and rebound in place (`inner.data = view`), so object
        identity in param_groups / state / all internal maps is preserved.
        exp_avg / exp_avg_sq are pre-seeded as (zero) arena views on the CPU
        sub-optimizers, which skips their lazy `zeros_like` initialization on
        first step (both DeepSpeedCPUAdam and torch Adam/AdamW gate it on
        `len(state) == 0` — `step` must therefore be pre-seeded too).

        Must be called again after anything that reruns _init_sub_optimizers()
        (e.g. load_state_dict): the rebuild replaces masters and sub-optimizer
        state, leaving the arenas unbound. Arena storages themselves are
        reused across re-materializations. Returns True when the arenas are
        bound; False (with everything left on the regular non-arena path) when
        a precondition fails.
        """
        if not self.contiguous_state:
            return False

        def _fail(reason):
            self._arena_slices = None
            import warnings

            warnings.warn(f"HDO contiguous state arenas disabled: {reason}")
            return False

        # Preconditions: fully offloaded, Adam-family CPU sub-optimizers
        # (their state is exactly {step, exp_avg, exp_avg_sq}).
        if self.gpu_optimizer is not None:
            return _fail("not fully offloaded (gpu sub-optimizer present)")
        for opt in self.cpu_optimizers:
            if not (
                isinstance(opt, (torch.optim.Adam, torch.optim.AdamW))
                or type(opt).__name__ == "DeepSpeedCPUAdam"
            ):
                return _fail(f"unsupported cpu optimizer {type(opt).__name__}")

        # Validate the full layout before touching anything, so a failure
        # never leaves a partial binding behind.
        entries = layout["entries"]
        slice_map = {}
        for orig_param, gbuf_idx, start, numel in entries:
            inner = self.param_to_inner_param.get(orig_param)
            if inner is None or inner.is_cuda or inner.dtype != torch.float32:
                return _fail("param without a CPU fp32 master")
            if inner.numel() != numel:
                return _fail(
                    f"shard numel mismatch ({inner.numel()} vs layout {numel})"
                )
            slice_map[orig_param] = (gbuf_idx, start, numel)
        inner_covered = {self.param_to_inner_param[p] for p in slice_map}
        for opt in self.cpu_optimizers:
            for p in _param_generator(opt):
                if p not in inner_covered:
                    return _fail("cpu sub-optimizer param not covered by layout")

        # Allocate (or reuse across re-materializations) the arenas. Zeros
        # matter: gap bytes (intra-bucket alignment) must match the zeros of
        # the regular dp_zero save path for byte-identical checkpoints.
        numels = list(layout["gbuf_numels_unpadded"])
        arenas = self.state_arenas
        if arenas is None or [a["param"].numel() for a in arenas] != numels:
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

        # Migrate masters. Old per-param storages (kept unpinned in arena
        # mode) are freed back to the OS one by one as they are rebound.
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
        """(Re-)point every CPU sub-optimizer state entry at its arena views.

        Existing values are preserved: moment tensors already holding data
        (e.g. deep-copied by a preceding torch load_state_dict) are copied
        into the arena views before rebinding; missing entries start from the
        arena zeros. `step` keeps its current value when present, otherwise it
        is initialized to zero with the type the optimizer class expects
        (plain int for DeepSpeedCPUAdam, 0-dim tensor for torch Adam/AdamW).
        """
        for opt in self.cpu_optimizers:
            is_ds = type(opt).__name__ == "DeepSpeedCPUAdam"
            for param in _param_generator(opt):
                orig = self.inner_param_to_orig_param[param]
                gbuf_idx, start, numel = self._arena_slices[orig]
                state = opt.state[param]
                step = state.get("step")
                if step is None:
                    step = 0 if is_ds else torch.zeros((), dtype=torch.float32)
                elif is_ds and isinstance(step, torch.Tensor):
                    # DistributedOptimizer.load_state_dict writes `step` as a
                    # fp32 tensor; DeepSpeedCPUAdam's C++ adam_update expects a
                    # plain int — normalize.
                    step = int(step.item())
                elif not is_ds and not isinstance(step, torch.Tensor):
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
        """Whether every CPU master / moment tensor is still an arena view.

        Cheap data_ptr check used to gate the zero-copy save/load fast paths.
        Returns False whenever the arenas were never materialized, or any
        state-replacing path (e.g. a rebuild without re-materialization) has
        silently detached tensors from the arenas — consumers then fall back
        to the regular gather/scatter code.
        """
        arenas = self.state_arenas
        slices = self._arena_slices
        if arenas is None or slices is None or self.gpu_optimizer is not None:
            return False
        esize = 4  # fp32
        try:
            checked = 0
            for orig, (gbuf_idx, start, numel) in slices.items():
                inner = self.param_to_inner_param.get(orig)
                if inner is None:
                    return False
                base = arenas[gbuf_idx]
                if inner.data_ptr() != base["param"].data_ptr() + start * esize:
                    return False
                state = self.state.get(orig)
                if not state:
                    return False
                for key in ("exp_avg", "exp_avg_sq"):
                    t = state.get(key)
                    if (
                        not isinstance(t, torch.Tensor)
                        or t.data_ptr() != base[key].data_ptr() + start * esize
                    ):
                        return False
                checked += 1
            total = sum(1 for opt in self.cpu_optimizers for _ in _param_generator(opt))
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

        Steps the sub-optimizers directly: step()'s grad transfer reads
        GPU-side grads, which don't exist yet at checkpoint-load time.
        One sub-optimizer at a time, so the synthetic grads only ever
        occupy one sub-optimizer's worth of host RAM (all-at-once means
        +4 B/param — hundreds of GB per node at large scale).
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
