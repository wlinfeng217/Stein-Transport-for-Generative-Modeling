"""Registry and shared reverse-OU runner for all eight published methods.

Every method receives tensor-identical initial particles, forward OU noise, and
(for stochastic methods) reverse Brownian increments. This is the central
implementation of the comparison protocol used by all toy targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, cast

from torch import Tensor
from tqdm.auto import tqdm

from .estimators import (
    fit_plain_score,
    fit_stein_informed_score,
    fit_stein_transport,
)
from .quantities import build_quantities

MethodName: TypeAlias = Literal[
    "sm_plain_det",
    "sm_stein_det",
    "st_det",
    "st_vec_det",
    "sm_plain_sto",
    "sm_stein_sto",
    "st_sto",
    "st_vec_sto",
]

DETERMINISTIC_METHODS: tuple[MethodName, ...] = (
    "sm_plain_det",
    "sm_stein_det",
    "st_det",
    "st_vec_det",
)
STOCHASTIC_METHODS: tuple[MethodName, ...] = (
    "sm_plain_sto",
    "sm_stein_sto",
    "st_sto",
    "st_vec_sto",
)
ALL_METHODS: tuple[MethodName, ...] = DETERMINISTIC_METHODS + STOCHASTIC_METHODS

METHOD_TITLES: Mapping[MethodName, str] = {
    "sm_plain_det": r"$\mathrm{SM}_{\mathrm{plain}}^{\mathrm{det}}$",
    "sm_stein_det": r"$\mathrm{SM}_{\mathrm{stein}}^{\mathrm{det}}$",
    "st_det": r"$\mathrm{ST}^{\mathrm{det}}$",
    "st_vec_det": r"$\mathrm{ST}_{\mathrm{vec}}^{\mathrm{det}}$",
    "sm_plain_sto": r"$\mathrm{SM}_{\mathrm{plain}}^{\mathrm{sto}}$",
    "sm_stein_sto": r"$\mathrm{SM}_{\mathrm{stein}}^{\mathrm{sto}}$",
    "st_sto": r"$\mathrm{ST}^{\mathrm{sto}}$",
    "st_vec_sto": r"$\mathrm{ST}_{\mathrm{vec}}^{\mathrm{sto}}$",
}


def _failure(error: Exception, step: int) -> dict[str, Any]:
    return {
        "status": "failed",
        "failure": {
            "step": step,
            "exception_type": type(error).__name__,
            "message": str(error),
        },
    }


def run_all_methods(
    training: Tensor,
    shared: Mapping[str, Tensor],
    config: Any,
    *,
    progress: bool = True,
) -> dict[MethodName, dict[str, Any]]:
    """Run all eight algorithms with exactly shared random inputs.

    ``training`` has shape ``(N, d)``. ``shared`` contains forward noise of
    shape ``(L, N, d)``, initial particles of shape ``(M, d)``, and reverse
    Brownian noise of shape ``(L, M, d)``. Score methods use the reverse ODE or
    SDE drift; Stein-transport methods directly estimate the corresponding
    drift. Failures are recorded per method rather than silently discarded.
    """
    states = {name: shared["initial_particles"].clone() for name in ALL_METHODS}
    failures: dict[MethodName, dict[str, Any]] = {}
    step_size = config.step_size
    noise_scale = training.new_tensor(2.0 * step_size).sqrt()

    steps = tqdm(range(config.reverse_steps), desc="reverse OU", disable=not progress)
    for step in steps:
        time = config.horizon - step * step_size
        try:
            quantities = build_quantities(
                training,
                shared["forward_noise"][step],
                time,
                scaling_factor=config.scaling_factor,
                noised_training=shared["noised_training"][step],
            )
        except Exception as error:
            for name in ALL_METHODS:
                if name not in failures:
                    failures[name] = _failure(error, step)
            break

        families = [
            (
                ("sm_plain_det", "sm_plain_sto"),
                lambda q=quantities: fit_plain_score(q, config.regularization),
            ),
            (
                ("sm_stein_det", "sm_stein_sto"),
                lambda q=quantities: fit_stein_informed_score(q, config.regularization),
            ),
            (
                ("st_det",),
                lambda q=quantities: fit_stein_transport(
                    q,
                    config.regularization,
                    stochastic=False,
                    vectorized=False,
                ),
            ),
            (
                ("st_vec_det",),
                lambda q=quantities: fit_stein_transport(
                    q,
                    config.regularization,
                    stochastic=False,
                    vectorized=True,
                ),
            ),
            (
                ("st_sto",),
                lambda q=quantities: fit_stein_transport(
                    q,
                    config.regularization,
                    stochastic=True,
                    vectorized=False,
                ),
            ),
            (
                ("st_vec_sto",),
                lambda q=quantities: fit_stein_transport(
                    q,
                    config.regularization,
                    stochastic=True,
                    vectorized=True,
                ),
            ),
        ]
        fitted: dict[MethodName, Any] = {}
        for names, fit in families:
            typed_names = cast(tuple[MethodName, ...], names)
            active = [name for name in typed_names if name not in failures]
            if not active:
                continue
            try:
                estimator = fit()
                fitted.update({name: estimator for name in active})
            except Exception as error:
                for name in active:
                    failures[name] = _failure(error, step)

        for name, estimator in fitted.items():
            try:
                state = states[name]
                if name.startswith("sm_"):
                    score_factor = 1.0 if name.endswith("_det") else 2.0
                    drift = state + score_factor * estimator.score(state)
                else:
                    drift = estimator.drift(state)
                state = state + step_size * drift
                if name in STOCHASTIC_METHODS:
                    state = state + noise_scale * shared["reverse_brownian"][step]
                states[name] = state
            except Exception as error:
                failures[name] = _failure(error, step)

    return {
        name: failures.get(name, {"status": "success", "samples": states[name]})
        for name in ALL_METHODS
    }
