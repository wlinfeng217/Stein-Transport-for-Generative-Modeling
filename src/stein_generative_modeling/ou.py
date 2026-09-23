"""Forward OU marginals and numerical reverse-time integration."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def ou_coefficients(time: Tensor | float, *, like: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``a_t`` and ``sigma_t`` on the device and dtype of ``like``."""
    t = torch.as_tensor(time, device=like.device, dtype=like.dtype)
    if bool(torch.any(t < 0)):
        raise ValueError("OU time must be non-negative")
    a = torch.exp(-t)
    sigma = torch.sqrt(-torch.expm1(-2.0 * t))
    return a, sigma


def forward_marginal(training: Tensor, noise: Tensor, time: Tensor | float) -> Tensor:
    """Apply the exact forward OU transition using supplied Gaussian noise."""
    if training.shape != noise.shape:
        raise ValueError("training and noise must have identical shapes")
    if training.ndim != 2 or not training.is_floating_point():
        raise ValueError("training and noise must be floating tensors of shape (N, d)")
    if training.device != noise.device or training.dtype != noise.dtype:
        raise ValueError("training and noise must have the same device and dtype")
    a, sigma = ou_coefficients(time, like=training)
    return a * training + sigma * noise


def reverse_euler(
    initial: Tensor,
    drifts: list[Callable[[Tensor], Tensor]],
    step_size: float,
    *,
    brownian: Tensor | None = None,
) -> Tensor:
    """Integrate a full reverse drift with Euler or Euler--Maruyama.

    ``brownian`` contains standard-normal vectors, one slice per drift. It is
    omitted for the deterministic probability-flow ODE.
    """
    if initial.ndim != 2 or not initial.is_floating_point():
        raise ValueError("initial must be a floating tensor of shape (M, d)")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if brownian is not None:
        expected = (len(drifts), *initial.shape)
        if tuple(brownian.shape) != expected:
            raise ValueError(f"brownian must have shape {expected}")
        if brownian.device != initial.device or brownian.dtype != initial.dtype:
            raise ValueError("brownian must match initial device and dtype")
    state = initial.clone()
    noise_scale = initial.new_tensor(2.0 * step_size).sqrt()
    for index, drift in enumerate(drifts):
        value = drift(state)
        if value.shape != state.shape:
            raise ValueError("drift returned a tensor with the wrong shape")
        state = state + step_size * value
        if brownian is not None:
            state = state + noise_scale * brownian[index]
    return state
