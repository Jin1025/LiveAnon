"""Stand-ins for the two icefall dependencies the vendored Zipformer encoder needs.

`scaling.py` imports k2 ONLY for the Swoosh activations, and `zipformer.py`/`scaling.py`
import `torch_autocast` from the icefall package. Installing k2 + icefall to run an
encoder forward is not worth it, so both are reimplemented here.

The Swoosh formulas are copied verbatim from the torch.jit.is_scripting() branch of
scaling.py, which is the reference definition k2's CUDA kernels implement — so this is
numerically equivalent, not an approximation.
"""
from __future__ import annotations

import contextlib

import torch


def _logaddexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(a, b)


def swoosh_l_forward(x: torch.Tensor) -> torch.Tensor:
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return _logaddexp(zero, x - 4.0) - 0.08 * x - 0.035


def swoosh_r_forward(x: torch.Tensor) -> torch.Tensor:
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return _logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687


# autograd handles the backward pass for these closed forms, so the plain
# forward doubles as the differentiable version.
swoosh_l = swoosh_l_forward
swoosh_r = swoosh_r_forward


def swoosh_l_forward_and_deriv(x: torch.Tensor):
    y = swoosh_l_forward(x)
    return y, torch.sigmoid(x - 4.0) - 0.08


def swoosh_r_forward_and_deriv(x: torch.Tensor):
    y = swoosh_r_forward(x)
    return y, torch.sigmoid(x - 1.0) - 0.08


@contextlib.contextmanager
def torch_autocast(device_type="cuda", **kwargs):
    with torch.amp.autocast(device_type=device_type, **kwargs):
        yield
