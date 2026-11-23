# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026, Baidu, Inc. All rights reserved.

"""
Generic (pure PyTorch) forward / backward for fused linear + cross-entropy.

This implementation replicates the same chunked-vocab, online-softmax algorithm
used by the architecture-specific (e.g. Blackwell) CUTLASS kernels, but with
standard PyTorch ops so that it runs on any CUDA-capable GPU.

Memory behavior is identical to the fused kernel path: the full
(num_tokens, vocab_size) logits tensor is never materialised.

Performance notes:
  - Matmul is done in the native dtype (bf16/fp16) for tensor-core throughput;
    only the online-softmax element-wise ops use float32.
  - Pre-allocated buffers are reused across splits to avoid per-iteration allocation.
"""

import logging
import os
import typing
from dataclasses import dataclass, field
from functools import lru_cache

import torch
import torch.distributed as dist

from .. import utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_VOCAB_PER_SPLIT = 512 * 6  # 3072, same default as Blackwell path


@dataclass
class GenericConfig:
    """Runtime configuration for the Generic (pure-PyTorch) fused LCE path.

    Both fields are read once at first use (via :func:`_get_config`) and cached
    for the lifetime of the process.  Override them with environment variables
    before launching training; changing them at runtime has no effect.

    Attributes:
        fwd_vocab_per_split: Vocabulary chunk size used during the **forward**
            pass.  Each chunk produces a ``(num_tokens, fwd_vocab_per_split)``
            logits buffer that is immediately consumed and discarded.
            Resolution order (first non-empty wins):

            1. ``LCE_GENERIC_FWD_VOCAB_SPLIT_SIZE``
            2. ``LCE_FWD_VOCAB_SPLIT_SIZE``
            3. Built-in default (``3072``).

        bwd_vocab_per_split: Vocabulary chunk size used during the **backward**
            pass.  Governs the size of the reused ``d_logits`` scratch buffer.
            Resolution order:

            1. ``LCE_GENERIC_BWD_VOCAB_SPLIT_SIZE``
            2. ``LCE_BWD_VOCAB_SPLIT_SIZE``
            3. Built-in default (``3072``).

    Tuning guidance:
        Larger chunks → fewer kernel launches, better GPU utilisation, but more
        peak memory per split buffer.  Smaller chunks → lower peak memory at the
        cost of more launches and potentially lower throughput.  The default
        ``3072`` balances memory and performance on typical A100/A800 workloads.
    """

    fwd_vocab_per_split: int = field(
        default_factory=lambda: int(
            os.environ.get(
                "LCE_GENERIC_FWD_VOCAB_SPLIT_SIZE",
                os.environ.get("LCE_FWD_VOCAB_SPLIT_SIZE", _DEFAULT_VOCAB_PER_SPLIT),
            )
        )
    )
    bwd_vocab_per_split: int = field(
        default_factory=lambda: int(
            os.environ.get(
                "LCE_GENERIC_BWD_VOCAB_SPLIT_SIZE",
                os.environ.get("LCE_BWD_VOCAB_SPLIT_SIZE", _DEFAULT_VOCAB_PER_SPLIT),
            )
        )
    )


@lru_cache(maxsize=1)
def _get_config() -> GenericConfig:
    return GenericConfig()


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@torch.no_grad()
def forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    tp_group: typing.Optional[dist.ProcessGroup] = None,
    reduction: typing.Literal["none", "sum", "mean"] = "mean",
    ignore_index: int = -100,
    sequence_parallel: bool = False,
) -> typing.Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor
]:
    """
    Chunked forward pass for fused linear + cross-entropy (pure PyTorch).

    Computes ``loss = cross_entropy(hidden @ weight.T, labels)`` without ever
    materialising the full ``(num_tokens, vocab_size)`` logits tensor.  Instead,
    the vocabulary dimension is split into chunks of size ``fwd_vocab_per_split``
    and processed sequentially, accumulating online-softmax statistics
    (``maximum``, ``accumulate``) across chunks.

    Algorithm per chunk
    -------------------
    1. ``logits_chunk = hidden @ weight[v_start:v_end].T``  — bf16 matmul (tensor cores)
    2. Cast ``logits_chunk`` to float32 for numerically stable arithmetic.
    3. Before any in-place modification, extract ``logit_at_label`` for tokens
       whose true label falls in this vocab chunk.
    4. Online-softmax update::

           new_max  = max(maximum, chunk_max)
           accumulate = accumulate * exp(maximum - new_max)
                        + sum(exp(logits_chunk - new_max))
           maximum  = new_max

    Final loss (per token)::

        loss[t] = maximum[t] + log(accumulate[t]) - logit_at_label[t]
               = -log softmax(logit_at_label[t])

    Tensor-Parallel support
    -----------------------
    * **TP mode** (``tp_group`` is not None, ``sequence_parallel=False``): each
      rank holds a ``(num_tokens, vocab_size / tp)`` weight shard.  After all
      chunks are processed, three all-reduces synchronise the online-softmax
      state across ranks: ``maximum`` (MAX), ``accumulate`` (SUM, after
      correction), ``logit_at_label`` (SUM — only one rank contributes per
      token).
    * **Sequence-Parallel mode** (``sequence_parallel=True``): each rank holds a
      ``(num_tokens / tp, dim)`` hidden shard.  An all-gather reconstructs
      ``global_hidden`` before the chunked matmul loop.

    Args:
        hidden: Activations, shape ``(num_tokens, dim)`` or
            ``(seq, batch, dim)``; bf16/fp16; contiguous.
        weight: LM-head weight, shape ``(vocab_size, dim)``; same dtype as
            ``hidden``; contiguous.
        labels: Token ids, shape ``(num_tokens,)`` or ``(seq, batch)``; int64.
        tp_group: Tensor-parallel process group, or ``None`` for single-GPU.
        reduction: Loss reduction mode — ``"none"``, ``"sum"``, or ``"mean"``.
        ignore_index: Label value to mask out (default ``-100``).
        sequence_parallel: If ``True``, ``hidden`` is a local shard and an
            all-gather is performed before the matmul.

    Returns:
        Tuple of seven elements saved for backward:

        * ``logprobs``   — scalar / per-token loss (dtype float32).
        * ``maximum``    — per-token running max, shape ``(num_tokens,)``.
        * ``accumulate`` — per-token sum-of-exp,  shape ``(num_tokens,)``.
        * ``num_valid_tokens`` — number of tokens with ``label != ignore_index``.
        * ``tp_rank``    — rank within ``tp_group`` (0 if no TP).
        * ``tp_world_size`` — size of ``tp_group`` (1 if no TP).
        * ``global_hidden`` — the (possibly all-gathered) hidden tensor, kept
          for the backward recomputation.
    """
    # ---- TP bookkeeping ----
    tp_rank = 0 if tp_group is None else dist.get_rank(tp_group)
    tp_world_size = 1 if tp_group is None else dist.get_world_size(tp_group)
    in_tp_mode = (tp_group is not None) and (tp_world_size > 1)

    # ---- input validation (mirrors blackwell/entry.py) ----
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() in (2, 3)
    assert weight.dim() == 2
    assert (hidden.dim() == 2 and labels.dim() == 1) or (
        hidden.dim() == 3 and labels.dim() == 2
    )
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()

    hidden_view = hidden.view(-1, hidden.shape[-1])
    labels_view = labels.view(-1)

    assert (
        sequence_parallel and hidden_view.shape[0] * tp_world_size == labels_view.shape[0]
    ) or (not sequence_parallel and hidden_view.shape[0] == labels_view.shape[0])
    assert hidden_view.shape[1] == weight.shape[1]

    # ---- sequence-parallel all-gather ----
    global_hidden = hidden
    if in_tp_mode and sequence_parallel:
        partial_hidden_shape = hidden.shape
        global_hidden_shape = (
            partial_hidden_shape[0] * tp_world_size,
            *partial_hidden_shape[1:],
        )
        global_hidden = torch.empty(
            global_hidden_shape, dtype=hidden.dtype, device=hidden.device
        )
        dist.all_gather_into_tensor(global_hidden, hidden, group=tp_group)
        hidden_view = global_hidden.view(-1, global_hidden.shape[-1])

    num_tokens = hidden_view.shape[0]
    vocab_size = weight.shape[0]
    device = hidden.device

    REDUCTION = utils.str_to_reduction_enum(reduction)

    # ---- valid-token mask & count ----
    valid_mask = labels_view != ignore_index
    num_valid_tokens = valid_mask.sum().to(torch.int64)

    # ---- online-softmax state (float32) ----
    maximum = torch.full((num_tokens,), -float("inf"), device=device, dtype=torch.float32)
    accumulate = torch.zeros((num_tokens,), device=device, dtype=torch.float32)
    logit_at_label = torch.zeros((num_tokens,), device=device, dtype=torch.float32)

    # ---- vocab-global offset for this TP rank ----
    vocab_offset = tp_rank * vocab_size

    # ---- pre-allocate reusable buffers ----
    vocab_per_split = _get_config().fwd_vocab_per_split
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    # bf16 buffer for matmul output (fast tensor-core path)
    matmul_buf = torch.empty(
        (num_tokens, vocab_per_split), device=device, dtype=hidden.dtype
    )
    # float32 buffer for online-softmax element-wise ops
    logits_buf = torch.empty(
        (num_tokens, vocab_per_split), device=device, dtype=torch.float32
    )
    row_idx = torch.arange(num_tokens, device=device)

    for split_idx in range(num_splits):
        v_start = split_idx * vocab_per_split
        v_end = min(v_start + vocab_per_split, vocab_size)
        chunk_size = v_end - v_start

        # ---- matmul in native dtype (bf16/fp16) — uses tensor cores ----
        matmul_chunk = matmul_buf[:, :chunk_size]
        torch.matmul(hidden_view, weight[v_start:v_end, :].t(), out=matmul_chunk)

        # ---- cast to float32 for online-softmax arithmetic ----
        logits_chunk = logits_buf[:, :chunk_size]
        logits_chunk.copy_(matmul_chunk)

        # ---- extract label logit BEFORE in-place exp overwrites logits ----
        global_v_start = vocab_offset + v_start
        label_in_chunk = (
            (labels_view >= global_v_start)
            & (labels_view < (vocab_offset + v_end))
            & valid_mask
        )
        if label_in_chunk.any():
            local_idx = (labels_view - global_v_start).clamp(0, chunk_size - 1)
            label_rows = row_idx[label_in_chunk]
            logit_at_label[label_rows] = logits_chunk[label_rows, local_idx[label_in_chunk]]

        # ---- online softmax update (in-place) ----
        chunk_max = logits_chunk.max(dim=1).values
        new_maximum = torch.maximum(maximum, chunk_max)

        logits_chunk.sub_(new_maximum.unsqueeze(1)).exp_()   # exp(logit - new_max)
        chunk_sum = logits_chunk.sum(dim=1)

        accumulate.mul_(torch.exp(maximum - new_maximum)).add_(chunk_sum)
        maximum = new_maximum

    # ---- TP reduction ----
    if in_tp_mode:
        local_max = maximum.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)

        accumulate.mul_(torch.exp(local_max - maximum))
        dist.all_reduce(accumulate, op=dist.ReduceOp.SUM, group=tp_group)

        dist.all_reduce(logit_at_label, op=dist.ReduceOp.SUM, group=tp_group)

    # ---- final loss: -log_softmax(logit_at_label) ----
    per_token_loss = maximum + torch.log(accumulate) - logit_at_label
    per_token_loss = torch.where(valid_mask, per_token_loss, per_token_loss.new_zeros(()))

    if REDUCTION == utils.EntropyReductionEnum.kNone:
        logprobs = per_token_loss
    elif REDUCTION == utils.EntropyReductionEnum.kSum:
        logprobs = per_token_loss.sum()
    else:  # kMean
        logprobs = per_token_loss.sum() / num_valid_tokens.float().clamp(min=1)

    return (
        logprobs,
        maximum,
        accumulate,
        num_valid_tokens,
        tp_rank,
        tp_world_size,
        global_hidden,
    )


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@torch.no_grad()
def backward(
    dlogprobs: torch.Tensor,
    global_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    accu: torch.Tensor,
    num_valid_tokens: torch.Tensor,
    reduction: typing.Literal["none", "sum", "mean"] = "mean",
    ignore_index: int = -100,
    tp_group: typing.Optional[dist.ProcessGroup] = None,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    sequence_parallel: bool = False,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked backward pass for fused linear + cross-entropy (pure PyTorch).

    Recomputes logits split-by-split (no logits tensor is saved from forward)
    and accumulates ``d_hidden`` and ``d_weight`` in-place.  The ``maximum``
    and ``accumulate`` statistics saved by the forward pass are used to
    reconstruct softmax probabilities without a second full forward pass.

    Gradient derivation
    -------------------
    For a single token ``t`` and vocab position ``v``::

        softmax_prob[t, v] = exp(logit[t,v] - maximum[t]) / accumulate[t]

        d_logits[t, v] = softmax_prob[t, v] * (dlogprobs[t] / accumulate[t])   # ∂L/∂logit (softmax branch)
                         - (v == label[t]) * dlogprobs[t]                       # ∂L/∂logit (label branch)

    Which simplifies to::

        d_logits[t, v] = grad_scale[t] * exp(logit[t,v] - maximum[t])
                         - (v == label[t]) * dlogprobs_per_token[t]

    where ``grad_scale[t] = dlogprobs[t] / accumulate[t]``.

    Algorithm per chunk
    -------------------
    1. Recompute ``logits_chunk = hidden @ weight[v_start:v_end].T`` (bf16 matmul).
    2. Cast to float32 and compute ``d_logits_chunk`` in-place using the formula
       above.
    3. Cast ``d_logits_chunk`` back to hidden dtype and accumulate::

           d_hidden += d_logits_chunk @ weight_chunk        (addmm, beta=0 on first split)
           d_weight[chunk] = d_logits_chunk.T @ hidden      (matmul, direct write)

    Pre-allocated buffers (``_d_logits``, ``matmul_buf``, ``logits_buf``) are
    reused across all splits to avoid repeated allocation.

    Tensor-Parallel support
    -----------------------
    * **TP mode**: ``d_hidden`` is all-reduced across ranks after the split loop
      because each rank only sees a vocab shard and accumulates partial gradients.
    * **Sequence-Parallel mode**: after the all-reduce, ``d_hidden`` is sliced
      to the local rank's token shard and reshaped to the original hidden shape.

    Args:
        dlogprobs: Upstream gradient of the loss, scalar (``"sum"``/``"mean"``)
            or shape ``(num_tokens,)`` (``"none"``); float32.
        global_hidden: Full (all-gathered) hidden tensor saved by forward,
            shape ``(num_tokens, dim)``; bf16/fp16; contiguous.
        weight: LM-head weight, shape ``(vocab_size, dim)``; same dtype as
            ``global_hidden``; contiguous.
        labels: Token ids, shape ``(num_tokens,)``; int64.
        maximum: Per-token running max from forward, shape ``(num_tokens,)``;
            float32.
        accu: Per-token sum-of-exp from forward, shape ``(num_tokens,)``; float32.
        num_valid_tokens: Number of non-ignored tokens (used for ``"mean"``
            reduction normalisation).
        reduction: Must match the value used in forward.
        ignore_index: Label value to mask out (default ``-100``).
        tp_group: Tensor-parallel process group, or ``None``.
        tp_rank: This rank's index within ``tp_group``.
        tp_world_size: Size of ``tp_group``.
        sequence_parallel: Whether sequence parallelism is active.

    Returns:
        ``(d_hidden, d_weight)`` — gradients w.r.t. the (local) hidden
        activations and the weight matrix; same shapes and dtypes as the
        corresponding forward inputs.
    """
    in_tp_mode = (tp_group is not None) and (tp_world_size > 1)

    hidden_view = global_hidden.view(-1, global_hidden.shape[-1])
    labels_view = labels.view(-1)

    num_tokens, dim = hidden_view.shape
    vocab_size = weight.shape[0]
    device = global_hidden.device

    REDUCTION = utils.str_to_reduction_enum(reduction)

    # ---- prepare per-token upstream gradient ----
    valid_mask = labels_view != ignore_index
    if REDUCTION == utils.EntropyReductionEnum.kMean:
        scale = dlogprobs.float() / num_valid_tokens.float().clamp(min=1)
        dlogprobs_per_token = scale.expand(num_tokens)
    elif REDUCTION == utils.EntropyReductionEnum.kSum:
        dlogprobs_per_token = dlogprobs.float().expand(num_tokens)
    else:  # kNone
        dlogprobs_per_token = dlogprobs.float().view(-1)
    dlogprobs_per_token = dlogprobs_per_token * valid_mask.float()

    # precompute: dlogprobs / accu  (used in every split)
    grad_scale = dlogprobs_per_token / accu.clamp(min=1e-30)  # (num_tokens,)

    d_hidden = torch.empty_like(global_hidden)
    d_weight = torch.empty_like(weight)

    vocab_offset = tp_rank * vocab_size
    vocab_per_split = _get_config().bwd_vocab_per_split
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    # reusable buffers
    _d_logits = torch.empty(
        (num_tokens, vocab_per_split), device=device, dtype=global_hidden.dtype
    )
    # bf16 buffer for matmul output
    matmul_buf = torch.empty(
        (num_tokens, vocab_per_split), device=device, dtype=global_hidden.dtype
    )
    # float32 buffer for gradient arithmetic
    logits_buf = torch.empty(
        (num_tokens, vocab_per_split), device=device, dtype=torch.float32
    )
    row_idx = torch.arange(num_tokens, device=device)

    d_hidden_flat = d_hidden.view(-1, dim)

    for split_idx in range(num_splits):
        v_start = split_idx * vocab_per_split
        v_end = min(v_start + vocab_per_split, vocab_size)
        chunk_size = v_end - v_start
        weight_chunk = weight[v_start:v_end, :]

        # ---- recompute logits in native dtype (bf16, fast) ----
        matmul_chunk = matmul_buf[:, :chunk_size]
        torch.matmul(hidden_view, weight_chunk.t(), out=matmul_chunk)

        # ---- cast to float32 for gradient computation ----
        logits_chunk = logits_buf[:, :chunk_size]
        logits_chunk.copy_(matmul_chunk)

        # d_logits = grad_scale * exp(logit - max)   (in-place)
        logits_chunk.sub_(maximum.unsqueeze(1)).exp_()            # exp(logit - max)
        logits_chunk.mul_(grad_scale.unsqueeze(1))                # * (dlogprobs / accu)

        # subtract at label positions
        global_v_start = vocab_offset + v_start
        label_in_chunk = (
            (labels_view >= global_v_start)
            & (labels_view < (vocab_offset + v_end))
            & valid_mask
        )
        if label_in_chunk.any():
            local_idx = (labels_view - global_v_start).clamp(0, chunk_size - 1)
            logits_chunk[row_idx[label_in_chunk], local_idx[label_in_chunk]] -= (
                dlogprobs_per_token[label_in_chunk]
            )

        # cast to hidden dtype into reusable buffer
        _d_logits[:, :chunk_size] = logits_chunk.to(global_hidden.dtype)
        valid_d_logits = _d_logits[:, :chunk_size]

        # d_hidden += d_logits @ weight_chunk  (bf16 matmul)
        torch.addmm(
            input=d_hidden_flat,
            mat1=valid_d_logits,
            mat2=weight_chunk,
            beta=float(split_idx != 0),
            alpha=1.0,
            out=d_hidden_flat,
        )

        # d_weight[chunk] = d_logits.T @ hidden  (bf16 matmul)
        torch.matmul(
            valid_d_logits.t(),
            hidden_view,
            out=d_weight[v_start:v_end, :],
        )

    # ---- TP reduction ----
    if in_tp_mode:
        dist.all_reduce(d_hidden, op=dist.ReduceOp.SUM, group=tp_group)
        if sequence_parallel:
            partial_hidden_shape = (
                global_hidden.shape[0] // tp_world_size,
                *global_hidden.shape[1:],
            )
            partial_num_tokens = num_tokens // tp_world_size
            d_hidden = d_hidden.view(-1, d_hidden.shape[-1])[
                tp_rank * partial_num_tokens : (tp_rank + 1) * partial_num_tokens, :
            ]
            d_hidden = d_hidden.view(partial_hidden_shape).clone()

    return d_hidden, d_weight
