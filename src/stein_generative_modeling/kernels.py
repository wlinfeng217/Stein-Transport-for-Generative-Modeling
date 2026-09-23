"""Multiscale inverse-multiquadric kernels and analytic derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

IMQ_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _matrix(value: Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d), got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def select_imq_bandwidths(reference: Tensor, scaling_factor: float = 1.0) -> Tensor:
    """Select the five documented distance-quantile bandwidths.

    Zero off-diagonal distances are excluded. The result inherits the reference
    tensor's device and dtype.
    """
    _matrix(reference, "reference")
    if reference.shape[0] < 2:
        raise ValueError("bandwidth selection requires at least two samples")
    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive")
    distances = torch.pdist(reference)
    distances = distances[distances > 0]
    if distances.numel() == 0:
        raise ValueError("bandwidth selection requires a positive pairwise distance")
    levels = reference.new_tensor(IMQ_QUANTILES)
    return torch.quantile(distances, levels) * scaling_factor


@dataclass(frozen=True)
class KernelEvaluation:
    """Pairwise kernel values and derivatives for two ensembles."""

    value: Tensor
    grad1: Tensor
    grad2: Tensor
    mixed_diag: Tensor


@dataclass(frozen=True)
class KernelFirstOrder:
    """Pairwise kernel values and both first derivatives."""

    value: Tensor
    grad1: Tensor
    grad2: Tensor


@dataclass(frozen=True)
class MultiscaleIMQ:
    """Equally weighted IMQ kernel with fixed exponent -1/2."""

    bandwidths: Tensor

    def __post_init__(self) -> None:
        if self.bandwidths.ndim != 1 or self.bandwidths.numel() == 0:
            raise ValueError("bandwidths must be a non-empty one-dimensional tensor")
        if not self.bandwidths.is_floating_point():
            raise TypeError("bandwidths must be floating point")
        if not bool(torch.all(self.bandwidths > 0)):
            raise ValueError("all bandwidths must be positive")

    @classmethod
    def from_reference(
        cls, reference: Tensor, scaling_factor: float = 1.0
    ) -> MultiscaleIMQ:
        return cls(select_imq_bandwidths(reference, scaling_factor))

    def _pairwise(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _matrix(x, "x")
        _matrix(y, "y")
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have the same state dimension")
        if x.device != y.device or x.dtype != y.dtype:
            raise ValueError("x and y must have the same device and dtype")
        if x.device != self.bandwidths.device or x.dtype != self.bandwidths.dtype:
            raise ValueError("kernel bandwidths must match x device and dtype")

        delta = x[:, None, :] - y[None, :, :]  # (m, n, d)
        radius2 = delta.square().sum(dim=-1)  # (m, n)
        ell2 = self.bandwidths.square()[:, None, None]  # (b, 1, 1)
        u = 1.0 + radius2[None, :, :] / ell2  # (b, m, n)
        return delta, radius2, ell2, u

    def first_order(self, x: Tensor, y: Tensor) -> KernelFirstOrder:
        """Evaluate values and first derivatives without second-order work."""
        delta, _radius2, ell2, u = self._pairwise(x, y)
        u_m32 = u.pow(-1.5)
        value = u.pow(-0.5).mean(dim=0)
        grad1 = (-delta[None, :, :, :] / ell2[:, :, :, None] * u_m32[..., None]).mean(
            dim=0
        )
        return KernelFirstOrder(value, grad1, -grad1)

    def evaluate(self, x: Tensor, y: Tensor) -> KernelEvaluation:
        """Evaluate all derivatives needed while fitting the estimators.

        The leading two dimensions index every pair from ``x`` and ``y``;
        derivative tensors additionally have a final coordinate dimension.
        """
        delta, _radius2, ell2, u = self._pairwise(x, y)
        u_m32 = u.pow(-1.5)
        u_m52 = u.pow(-2.5)
        value = u.pow(-0.5).mean(dim=0)
        grad1 = (-delta[None, :, :, :] / ell2[:, :, :, None] * u_m32[..., None]).mean(
            dim=0
        )
        mixed_diag = (
            u_m32[..., None] / ell2[:, :, :, None]
            - 3.0
            * delta.square()[None, :, :, :]
            / ell2.square()[:, :, :, None]
            * u_m52[..., None]
        ).mean(dim=0)
        return KernelEvaluation(value, grad1, -grad1, mixed_diag)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        _delta, _radius2, _ell2, u = self._pairwise(x, y)
        return u.pow(-0.5).mean(dim=0)
