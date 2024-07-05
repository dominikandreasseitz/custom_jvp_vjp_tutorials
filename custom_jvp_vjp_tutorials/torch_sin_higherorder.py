from __future__ import annotations

from typing import Any

import torch

# This tutorial is based on
# https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html


def sin_fwd(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


def sin_bwd(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # dfdx = cos(x)
    return grad_out * torch.cos(x)


def sin_bwd_bwd(
    grad_out: torch.Tensor, sav_grad_out: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    # dfdxx = -sin(x)
    return grad_out * sav_grad_out * torch.tensor([-1.0]) * torch.sin(x)


class Sin(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return sin_fwd(x)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple:
        (x,) = ctx.saved_tensors
        # We return the output of `SinBackward` which is the first derivative
        return SinBackward.apply(grad_out, x)


class SinBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(grad_out, x)
        return sin_bwd(grad_out, x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sav_grad_out, x = ctx.saved_tensors
        # We return a tuple of tensors containing gradients for each input to `SinBackward.forward`
        # Hence, dfdx for `grad_out` and dfdxx for `x`
        return sin_bwd(grad_out, x), sin_bwd_bwd(grad_out, sav_grad_out, x)


if __name__ == "__main__":
    x = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    res = Sin.apply(x)
    dfdx = torch.autograd.grad(res, x, torch.ones_like(res), create_graph=True)[0]
    dfdxx = torch.autograd.grad(dfdx, x, torch.ones_like(dfdx))[0]

    res_ad = torch.sin(x)
    dfdx_ad = torch.autograd.grad(
        res_ad, x, torch.ones_like(res_ad), create_graph=True
    )[0]
    dfdxx_ad = torch.autograd.grad(dfdx_ad, x, torch.ones_like(dfdx_ad))[0]
    assert torch.allclose(dfdx, dfdx_ad)
    assert torch.allclose(dfdxx, dfdxx_ad)
    assert torch.autograd.gradcheck(Sin.apply, x)
    assert torch.autograd.gradgradcheck(Sin.apply, x)
