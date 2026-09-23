"""Shared sample-dependent quantities used by every drift estimator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .kernels import MultiscaleIMQ
from .ou import forward_marginal, ou_coefficients


@dataclass(frozen=True)
class EstimatorQuantities:
    """Reusable ``N``-sample tensors at one positive OU noising time.

    Directional arrays use shape ``(d, N, N)``; sample-vector quantities use
    shape ``(N, d)``. Every tensor shares the training data's dtype and device.
    """

    z: Tensor
    conditional_score: Tensor
    psi_det_directional: Tensor
    psi_sto_directional: Tensor
    laplacian_directional: Tensor
    kernel: MultiscaleIMQ
    kernel_gram: Tensor
    xi_directional: Tensor
    xi: Tensor
    cross_gram_directional: Tensor

    @property
    def psi_det(self) -> Tensor:
        return self.psi_det_directional.sum(dim=1)

    @property
    def psi_sto(self) -> Tensor:
        return self.psi_sto_directional.sum(dim=1)


def build_quantities(
    training: Tensor,
    noise: Tensor,
    time: Tensor | float,
    *,
    scaling_factor: float = 1.0,
    noised_training: Tensor | None = None,
) -> EstimatorQuantities:
    """Construct the shared quantities at one strictly positive noising level."""
    if training.shape[0] < 2:
        raise ValueError("estimators require at least two training samples")
    t = torch.as_tensor(time, device=training.device, dtype=training.dtype)
    if t.ndim != 0 or not bool(t > 0):
        raise ValueError("estimator time must be a positive scalar")
    computed_z = forward_marginal(training, noise, t)
    if noised_training is None:
        z = computed_z
    else:
        if noised_training.shape != training.shape:
            raise ValueError("noised_training must have the training shape")
        if (
            noised_training.device != training.device
            or noised_training.dtype != training.dtype
        ):
            raise ValueError("noised_training must match training device and dtype")
        if not torch.equal(noised_training, computed_z):
            raise ValueError(
                "noised_training is inconsistent with training, noise, and time"
            )
        z = noised_training
    a, sigma = ou_coefficients(t, like=training)
    score = -noise / sigma
    lap_directional = (noise.square() - 1.0) / sigma.square()
    psi_det = (
        -a.square() / sigma.square()
        - a * training * noise / sigma
        + a.square() * noise.square() / sigma.square()
    )
    psi_sto = psi_det + lap_directional

    kernel = MultiscaleIMQ.from_reference(z, scaling_factor)
    pair = kernel.evaluate(z, z)
    k = pair.value
    # Coordinate tensor convention: (r, i, j).
    score_i = score.T[:, :, None]
    score_j = score.T[:, None, :]
    xi_directional = (
        k[None, :, :] * score_i * score_j
        + score_i * pair.grad2.permute(2, 0, 1)
        + score_j * pair.grad1.permute(2, 0, 1)
        + pair.mixed_diag.permute(2, 0, 1)
    )
    cross = score_i * k[None, :, :] + pair.grad1.permute(2, 0, 1)
    return EstimatorQuantities(
        z=z,
        conditional_score=score,
        psi_det_directional=psi_det,
        psi_sto_directional=psi_sto,
        laplacian_directional=lap_directional,
        kernel=kernel,
        kernel_gram=k,
        xi_directional=xi_directional,
        xi=xi_directional.sum(dim=0),
        cross_gram_directional=cross,
    )
