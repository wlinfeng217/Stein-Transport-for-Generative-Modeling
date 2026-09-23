"""Documented score-matching and Stein-transport estimators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .quantities import EstimatorQuantities


def _regularized_solve(
    matrix: Tensor, response: Tensor, regularization: float
) -> Tensor:
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return torch.linalg.solve(
        matrix / matrix.shape[-1] + regularization * identity, response
    )


@dataclass(frozen=True)
class PlainScore:
    """Kernel expansion of the plain conditional score estimator."""

    q: EstimatorQuantities
    coefficients: Tensor  # (N, d)

    def score(self, x: Tensor) -> Tensor:
        """Evaluate the score at an ``(M, d)`` query tensor."""
        return self.q.kernel(x, self.q.z) @ self.coefficients / self.q.z.shape[0]


@dataclass(frozen=True)
class SteinInformedScore:
    """Coordinate-wise Stein-informed conditional score estimator."""

    q: EstimatorQuantities
    alpha: Tensor  # (d, N)
    beta: Tensor  # (d, N)

    def score(self, x: Tensor) -> Tensor:
        """Evaluate the score at an ``(M, d)`` query tensor."""
        pair = self.q.kernel.first_order(x, self.q.z)
        representers = (
            pair.value[..., None] * self.q.conditional_score[None, :, :] + pair.grad2
        )
        first = torch.stack(
            [representers[:, :, r] @ self.alpha[r] for r in range(x.shape[1])],
            dim=1,
        )
        second = pair.value @ self.beta.T
        return (first + second) / self.q.z.shape[0]


@dataclass(frozen=True)
class SteinTransport:
    """Scalar or coordinate-wise Stein approximation of the reverse drift."""

    q: EstimatorQuantities
    coefficients: Tensor  # (N,) for scalar or (d, N) for vectorized

    def drift(self, x: Tensor) -> Tensor:
        """Evaluate the fitted reverse drift at an ``(M, d)`` query tensor."""
        pair = self.q.kernel.first_order(x, self.q.z)
        representers = (
            pair.value[..., None] * self.q.conditional_score[None, :, :] + pair.grad2
        )
        if self.coefficients.ndim == 1:
            return (
                torch.einsum("mnd,n->md", representers, self.coefficients)
                / self.q.z.shape[0]
            )
        return (
            torch.stack(
                [
                    representers[:, :, r] @ self.coefficients[r]
                    for r in range(x.shape[1])
                ],
                dim=1,
            )
            / self.q.z.shape[0]
        )


def fit_plain_score(q: EstimatorQuantities, regularization: float) -> PlainScore:
    """Fit plain score matching by a regularized kernel linear solve."""
    coefficients = _regularized_solve(
        q.kernel_gram, q.conditional_score, regularization
    )
    return PlainScore(q, coefficients)


def fit_stein_informed_score(
    q: EstimatorQuantities, regularization: float
) -> SteinInformedScore:
    """Fit the documented coordinate-wise Stein-informed score systems."""
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    n, dimension = q.z.shape
    alpha = q.z.new_empty((dimension, n))
    beta = q.z.new_empty((dimension, n))
    identity = torch.eye(2 * n, device=q.z.device, dtype=q.z.dtype)
    for r in range(dimension):
        block = torch.cat(
            (
                torch.cat((q.xi_directional[r], q.cross_gram_directional[r]), dim=1),
                torch.cat((q.cross_gram_directional[r].T, q.kernel_gram), dim=1),
            ),
            dim=0,
        )
        response = torch.cat((q.laplacian_directional[:, r], q.conditional_score[:, r]))
        coefficients = torch.linalg.solve(
            block / n + regularization * identity, response
        )
        alpha[r], beta[r] = coefficients[:n], coefficients[n:]
    return SteinInformedScore(q, alpha, beta)


def fit_stein_transport(
    q: EstimatorQuantities,
    regularization: float,
    *,
    stochastic: bool,
    vectorized: bool,
) -> SteinTransport:
    """Fit scalar or vectorized deterministic/stochastic Stein transport."""
    if vectorized:
        response = q.psi_sto_directional if stochastic else q.psi_det_directional
        coefficients = torch.stack(
            [
                _regularized_solve(q.xi_directional[r], response[:, r], regularization)
                for r in range(q.z.shape[1])
            ]
        )
    else:
        response = q.psi_sto if stochastic else q.psi_det
        coefficients = _regularized_solve(q.xi, response, regularization)
    return SteinTransport(q, coefficients)
