# custom_jvp_vjp_tutorials

## What are vector-jacobian products and jacobian-vector products?
`jvp` refers to forward-mode automatic differentiation, meaning carrying both a primals and tangents
forward while computing the "end result" of your computation.
`vjp` refers to reverse-mode automatic differentiation and computes derivatives using the chainrule
applied to the "end" of your computation stack all the way up to the root.
## When to use which?
JVP: for function with a small number of inputs (parameters) and a large number of outputs
VJP: for functions with a large number of inputs (parameters) and a small number of ouputs.
## Torch
### torch.autograd.Function
`torch` allows users to implement their own custom backward functions (vjps for now, jvps in the making as of July 2024). The interface looks like:

```python exec="on" source="material-block" html="1"
from typing import Any
import torch


class CustomBackward(torch.autograd.Function):

    def forward(ctx: Any):
        pass

    def backward(ctx: Any):
        pass

```

We require two functions: `forward` and `backward.
`torch` uses a context object `ctx` to store interemediate results of the forward pass which
can then accessed in the backward pass to compute derivatives.

### Implementing a custom torch backward pass using `torch.autograd.Function`
Let's look at a fleshed-out pseudo example:

```python exec="on" source="material-block" html="1"
from typing import Any, Tuple
import torch


def custom_logic(intermediate_result: torch.Tensor, some_tensors: torch.Tensor) -> torch.Tensor:
    # This function does some custom gradient logic
    return intermediate_result * some_tensors

class CustomTorchBackward(torch.autograd.Function):
    def forward(ctx: Any, staticarg0: Any, staticarg1: Any, requires_grad_arg: torch.Tensor) -> torch.Tensor:
        # 1. compute and store intermediate results needed for backward pass in `ctx`
        # NOTE Assume `requires_grad_arg` is a parameter Tensor which `requires_grad`
        intermediate_result = staticarg0(requires_grad_arg)
        ctx.save_for_backward(requires_grad_arg)
        ctx.intermediate_result = intermediate_result
        # 2. compute full forward pass
        result = staticarg1(intermediate_result)
        # 3. Return result
        return result

    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Any, Any, torch.Tensor]:
        # 1. retrieve intermediate result needed for vjp computation from `ctx` object
        # 2. compute gradients using `custom_logic` and saved_tensors from forward
        grad = grad_out * custom_logic(ctx.intermediate_result, ctx.saved_tensors)
        # 3. Return a tuple containing as many `None`s as there are static args which we
        # do not intend to compute grads for (so `staticarg0, staticarg1` in our case)
        # And lastly the unpacked `grad` objects containing gradients for each of our
        # tensors which require grad in our `requires_grad_arg` object
        return (None, None, *grad)
```

### Torch Double backward Functions

[The torch custom function double backward tutorial](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)

```python exec="on" source="material-block" html="1"
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

```
## JAX
[Custom JAX derivative rules](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)

### Custom vjps and jvps
