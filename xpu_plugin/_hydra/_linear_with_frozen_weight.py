"""
Reinplementation of `LinearWithFrozenWeight` via hydra.
"""

from typing import Optional

import torch
import torch.distributed as dist
from hydrax.xaccelerator.linear import linear_bwd_dgrad, linear_fwd, linear_with_all_reduce
from hydrax.xaccelerator.linear._linear import view_matrix
from hydrax.xaccelerator.stateful import StatefulConfig
from torch_xmlir.config import config


@StatefulConfig(
    retain_state_for_parameter=not config.disable_cast_cache
).make_autograd_function_stateful
class LinearWithFrozenWeight(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        allreduce_dgrad: bool,
        tp_group: Optional[dist.ProcessGroup],
        output_dtype: Optional[torch.dtype] = None,
    ):
        if output_dtype is not None and output_dtype != input.dtype:
            raise NotImplementedError(
                "XPU Hydra frozen-weight linear does not support an output dtype different "
                "from the input dtype."
            )
        result = linear_fwd(input, weight, bias)

        ctx.save_for_backward(weight)
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        ctx.input_shape = input.shape
        ctx.input_dtype = input.dtype

        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[reportIncompatibleMethodOverride]
        grad_input = grad_output.new_empty(ctx.input_shape, dtype=ctx.input_dtype)
        (weight,) = ctx.saved_tensors

        if ctx.allreduce_dgrad:
            linear_with_all_reduce(
                view_matrix(grad_output),
                weight,
                view_matrix(grad_input),
                group=ctx.tp_group,
                output=view_matrix(grad_input),
            )
        else:
            linear_bwd_dgrad(grad_output, weight, grad_input)

        return grad_input, None, None, None, None, None
