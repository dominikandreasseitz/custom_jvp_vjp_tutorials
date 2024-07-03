from __future__ import annotations

from typing import Any

import torch


class CustomBackward(torch.autograd.Function):

    def forward(ctx: Any):
        pass

    def backward(ctx: Any):
        pass
