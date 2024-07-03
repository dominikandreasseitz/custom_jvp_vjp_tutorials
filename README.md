# custom_jvp_vjp_tutorials

## What are vector-jacobian products and jacobian-vector products?

## When to use which?




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

## JAX
[Custom JAX derivative rules](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)

### Custom vjps and jvps
