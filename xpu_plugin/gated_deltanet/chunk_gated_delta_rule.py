"""Chunk-based Gated Delta Rule attention for Baidu Kunlun XPU."""

import contextlib
import functools
import inspect
import warnings
from collections.abc import Callable

import cocopod
import kunlun_ops
import torch
import xspeedgate_ops

# ============================================================================
# Device utilities for XPU
# ============================================================================

def _get_device_type() -> str:
    """Detect the current XPU device backend name."""
    if hasattr(torch, '_C') and hasattr(torch._C, '_get_privateuse1_backend_name'):
        return torch._C._get_privateuse1_backend_name()
    return 'xpu'


_DEVICE_TYPE = _get_device_type()


def custom_device_ctx(index: int):
    """Return a device context manager for the current XPU backend."""
    mod = getattr(torch, _DEVICE_TYPE, None)
    if mod is not None and hasattr(mod, 'device'):
        return mod.device(index)
    return contextlib.nullcontext()


# ============================================================================
# Autocast utilities for XPU
# ============================================================================

def autocast_custom_fwd(fn):
    """No-op decorator for forward autocast on XPU (autocast not supported)."""
    return fn
def autocast_custom_bwd(fn):
    """No-op decorator for backward autocast on XPU (autocast not supported)."""
    return fn


# ============================================================================
# Input guard decorator
# ============================================================================

def input_guard(
    fn: Callable[..., torch.Tensor] | None=None,
    *,
    no_guard_contiguous: bool | list[str]=False,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]] | Callable[..., torch.Tensor]:
    """Decorator to ensure input tensors are contiguous and set device context."""

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            skip_params = set()
            if isinstance(no_guard_contiguous, list):
                skip_params = set(no_guard_contiguous)

            processed_args = []
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                else:
                    param_name = f"__arg_{i}"

                if isinstance(arg, torch.Tensor):
                    if no_guard_contiguous is True or param_name in skip_params:
                        processed_args.append(arg)
                    else:
                        processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)

            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    if no_guard_contiguous is True or k in skip_params:
                        processed_kwargs[k] = v
                    else:
                        processed_kwargs[k] = v.contiguous()
                else:
                    processed_kwargs[k] = v

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            if tensor is not None:
                ctx = custom_device_ctx(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)

    return decorator


# ============================================================================
# L2 normalization (pure PyTorch, for XPU compatibility)
# ============================================================================

def l2norm(x: torch.FloatTensor, dim: int=-1, eps: float=1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def l2norm_fwd(
    x: torch.Tensor,
    eps: float=1e-6,
    output_dtype: torch.dtype | None=None,
):
    """Forward pass of L2 normalization along the last dimension."""
    out = l2norm(x, dim=-1, eps=eps)
    inv_norm = torch.rsqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) + eps)
    rstd = inv_norm
    return out.to(x.dtype), rstd

def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
):
    # xspeedgate_ops.l2norm_bwd with bf16 inputs [N, D]
    orig_shape = y.shape
    y_2d = y.reshape(-1, y.shape[-1]).bfloat16()
    rstd_2d = rstd.reshape(-1, rstd.shape[-1]).bfloat16()
    dy_2d = dy.reshape(-1, dy.shape[-1]).bfloat16()
    dx_2d = torch.ops.xspeedgate_ops.l2norm_bwd(y_2d, rstd_2d, dy_2d, eps)
    return dx_2d.to(y.dtype).reshape(orig_shape)


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool=False,
    scale: float=None,
    cu_seqlens: torch.Tensor | None=None,
    head_first: bool=False,
    output_dtype: torch.dtype | None=None,
    chunk_indices: torch.LongTensor | None=None,
    **kwargs,
):
    """Compute chunk-local cumulative sum of the forget gate in log space."""
    return _xpu_call_fp32(torch.ops.xspeedgate_ops.chunk_local_cumsum,
        g=g,
        chunk_size=chunk_size,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        head_first=head_first,
    )


def _cdiv(x, y):
    """Ceiling division: returns ``(x + y - 1) // y``."""
    return (x + y - 1) // y


def _xpu_call_fp16(fn, *args, **kwargs):
    """Cast floating point tensor inputs to fp16 with cloned memory for XPU."""
    new_args = []
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            new_args.append(a.half() if a.dtype != torch.float16 else a.clone())
        else:
            new_args.append(a)
    new_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            new_kwargs[k] = v.half() if v.dtype != torch.float16 else v.clone()
        else:
            new_kwargs[k] = v
    return fn(*new_args, **new_kwargs)


def _xpu_call_fp32(fn, *args, **kwargs):
    """Cast non-fp32 floating point tensor inputs to fp32. Output dtype is not converted."""
    new_args = []
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != torch.float32:
            new_args.append(a.float())
        else:
            new_args.append(a)
    new_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch.float32:
            new_kwargs[k] = v.float()
        else:
            new_kwargs[k] = v
    return fn(*new_args, **new_kwargs)


def _prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int):
    """Compute cumulative chunk offsets from sequence lengths for variable-length inputs."""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]

    chunk_counts = (lens + chunk_size - 1) // chunk_size

    return torch.cat([cu_seqlens.new_tensor([0]), chunk_counts]).cumsum(-1).to(torch.int32)


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    """Build a chunk-index lookup table for variable-length sequences."""
    lens = torch.diff(cu_seqlens)
    num_chunks = _cdiv(lens, chunk_size)
    indices = torch.cat([torch.arange(n, device=cu_seqlens.device) for n in num_chunks.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None=None,
    beta: torch.Tensor | None=None,
    cu_seqlens: torch.LongTensor | None=None,
    chunk_size: int=64,
    output_dtype: torch.dtype=torch.float32,
) -> torch.Tensor:
    """Compute the scaled dot-product K^T K matrix within each chunk."""
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    k = k.contiguous()
    A = _xpu_call_fp16(torch.ops.xspeedgate_ops.chunk_scaled_dot_kkt_fwd,
        k=k,
        beta=beta,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return A


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None=None,
    chunk_indices: torch.LongTensor | None=None,
    output_dtype: torch.dtype=torch.float,
):
    """In-place solve the lower-triangular system for the WY representation."""
    if A.dtype == torch.bfloat16:
        A_fp16 = A.half()
        torch.ops.xspeedgate_ops.solve_tril_fwd(
            A=A_fp16,
            cu_seqlens=cu_seqlens,
        )
        A.copy_(A_fp16)
    else:
        torch.ops.xspeedgate_ops.solve_tril_fwd(
            A=A,
            cu_seqlens=cu_seqlens,
        )
    return A


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None=None,
    g_gamma: torch.Tensor | None=None,
    scale: float | None=None,
    cu_seqlens: torch.LongTensor | None=None,
    chunk_size: int=64,
) -> torch.Tensor:
    """Compute the output tensor in the forward pass using the recurrent state."""
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = _xpu_call_fp16(torch.ops.xspeedgate_ops.chunk_fwd_o,
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return o


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None=None,
    gk: torch.Tensor | None=None,
    initial_state: torch.Tensor | None=None,
    output_final_state: bool=False,
    chunk_size: int=64,
    save_new_value: bool=True,
    cu_seqlens: torch.LongTensor | None=None,
    chunk_indices: torch.LongTensor | None=None,
    use_exp2: bool=False,
):
    """Compute the recurrent hidden state for the gated delta rule forward pass."""
    B, H, K, V = k.shape[0], k.shape[2], k.shape[3], u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    if cu_seqlens is None:
        N, chunk_offsets = B, None
    else:
        N, chunk_offsets = len(cu_seqlens) - 1, _prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h, v_new, final_state = _xpu_call_fp16(torch.ops.xspeedgate_ops.chunk_gated_delta_rule_fwd_h,
        k=k,
        u=u,
        w=w,
        g=g,
        h0=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        chunk_size=chunk_size,
        output_final_state=output_final_state,
        save_new_value=save_new_value,
    )
    return h, v_new, final_state


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None=None,
    cu_seqlens: torch.LongTensor | None=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the WY-representation weights and values from solved A."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None

    w, u = _xpu_call_fp16(torch.ops.xspeedgate_ops.recompute_w_u_fwd,
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
    )
    return w, u


def chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None=None,
    g_gamma: torch.Tensor | None=None,
    A: torch.Tensor | None=None,
    scale: float=None,
    cu_seqlens: torch.LongTensor | None=None,
    chunk_size: int=64,
    chunk_indices: torch.LongTensor | None=None,
) -> torch.Tensor:
    """Compute the local (intra-chunk) gradient w.r.t. values."""
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    dv = _xpu_call_fp32(torch.ops.xspeedgate_ops.chunk_bwd_dv_local,
        q=q,
        k=k,
        g=g,
        do_=do,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return dv

def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None=None,
    gk: torch.Tensor | None=None,
    h0: torch.Tensor | None=None,
    dht: torch.Tensor | None=None,
    scale: float | None=None,
    cu_seqlens: torch.LongTensor | None=None,
    chunk_size: int=64,
    chunk_indices: torch.LongTensor | None=None,
    use_exp2: bool=False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gradients w.r.t. hidden state, initial state, and values."""
    return _xpu_call_fp32(kunlun_ops.chunk_gated_delta_rule_bwd_dhu,
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens.cpu(),
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )


def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None=None,
    g: torch.Tensor | None=None,
    g_gamma: torch.Tensor | None=None,
    dv: torch.Tensor | None=None,
    scale: float | None=None,
    cu_seqlens: torch.LongTensor | None=None,
    chunk_size: int=64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute gradients w.r.t. queries, keys, WY weights, and forget gate."""
    return _xpu_call_fp32(kunlun_ops.chunk_bwd_dqkwg,
        q=q,
        k=k,
        v=v,
        do=do,
        h=h,
        dh=dh,
        w=w,
        g=g,
        g_gamma=g_gamma,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens.cpu(),
        chunk_size=chunk_size,
    )


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor=None,
    cu_seqlens: torch.LongTensor | None=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward pass for the WY representation preparation step."""
    BT = 64
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None

    dk, dv, db, dg = _xpu_call_fp32(torch.ops.xspeedgate_ops.prepare_wy_repr_bwd,
        k=k,
        v=v,
        beta=beta,
        A=A,
        dw=dw,
        du=du,
        chunk_size=BT,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return dk, dv, db, dg


# ============================================================================
# chunk_gated_delta_rule_fwd / chunk_gated_delta_rule_bwd
# ============================================================================

def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None=None,
    cp_context=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Forward pass of the chunked gated delta rule."""

    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)

    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )

    A = solve_tril(A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)

    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )

    if cp_context is not None:
        raise NotImplementedError("chunk_gated_delta_rule_fwd does not support CP yet")

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    return g, o, A, final_state, initial_state

def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None=None,
    cp_context=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass of the chunked gated delta rule."""
    if initial_state is None:
        dht = None
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens,
    )

    if cp_context is not None:
        raise NotImplementedError("chunk_gated_delta_rule_bwd does not support CP yet")

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=initial_state,
        output_final_state=False, cu_seqlens=cu_seqlens,
    )

    dv = chunk_bwd_dv_local(
        q=q, k=k, do=do, g=g, scale=scale, cu_seqlens=cu_seqlens,
    )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q, k=k, w=w, do=do, dv=dv, g=g, h0=initial_state,
        dht=dht, scale=scale, cu_seqlens=cu_seqlens,
    )

    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q, k=k, v=v_new, do=do, h=h, dh=dh, w=w, g=g,
        dv=dv, scale=scale, cu_seqlens=cu_seqlens,
    )

    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv, cu_seqlens=cu_seqlens,
    )

    dk.add_(dk2)
    dg.add_(dg2)

    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)

    return dq, dk, dv, db, dg, dh0

# ============================================================================
# Custom autograd function
# ============================================================================

class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Custom autograd function for the chunked gated delta rule with XPU kernels."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None=None,
        use_qk_l2norm_in_kernel: bool=False,
        cp_context=None,
    ):
        """Forward pass: compute gated delta rule attention output."""
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        g, o, A, final_state, initial_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cp_context = cp_context
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        """Backward pass: compute gradients for the gated delta rule."""
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            cp_context=ctx.cp_context,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None


# ============================================================================
# Top-level API
# ============================================================================

@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None=None,
    initial_state: torch.Tensor=None,
    output_final_state: bool=False,
    use_qk_l2norm_in_kernel: bool=False,
    cu_seqlens: torch.LongTensor | None=None,
    cp_context=None,
    **kwargs,
):
    r"""Chunk-based Gated Delta Rule attention for XPU."""
    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )

    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        cu_seqlens = cp_context.cu_seqlens

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is None:
        B, T = q.shape[0], q.shape[1]
        cu_seqlens = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=q.device)
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        cp_context,
    )
    return o, final_state
