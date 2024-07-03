# custom_jvp_vjp_tutorials

## What are vector-jacobian products and jacobian-vector products?

## When to use which?

## torch.autograd.Function

[The torch custom function double backward tutorial](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)

### The context object `ctx`

```python
import torch

class CustomBackward(torch.autograd.Function):
    def forward(ctx: Any, staticarg0: Any, staticarg1: Any, requires_grad_arg: torch.Tensor):
        # 1. compute and store intermediate results needed for backward pass in `ctx`
        # assume arg1 contains something which we want to compute gradients for
        intermediate_result = staticarg0(requires_grad_arg)
        ctx.save_for_backward(requires_grad_arg)
        ctx.intermediate_result = intermediate_result
        # 2. compute full forward pass
        result = staticarg1(intermediate_result)
        # 3. Return result
        return result

    def backward(ctx: Any, grad_out: torch.Tensor):
        # 1. retrieve intermediate result needed for vjp computation from `ctx` object
        # 2. compute gradients using `custom_logic` and saved_tensors from forward
        grad = grad_out * custom_logic(ctx.intermediate_result, ctx.saved_tensors)
        # 3. Return a tuple containing as many `None`s as there are static args which we
        # do not intend to compute grads for (so `staticarg0, staticarg1` in our case)
        # And lastly the unpacked `grad` objects containing gradients for each of our
        # tensors which require grad in our `requires_grad_arg` object
        return (None, None, *grad)
```
## jax.custom_vjp
[Custom JAX derivative rules](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
