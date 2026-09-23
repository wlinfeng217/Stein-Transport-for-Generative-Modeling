"""Reproducible eight-component Gaussian-mixture toy experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .experiment import (
    PROJECT_ROOT,
    _generator,
    generate_ou_shared_inputs,
    run_cli,
    run_toy_experiment,
)
from .experiment import (
    plot_saved_run as plot_toy_run,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "gaussian_mixture"
PLOT_LIMITS = (-2.8, 2.8)


@dataclass(frozen=True)
class GaussianMixtureConfig:
    dimension: int = 2
    components: int = 8
    ring_radius: float = 2.0
    component_std: float = 0.1
    training_size: int = 100
    reference_size: int = 10_000
    generated_size: int = 2_000
    horizon: float = 3.0
    reverse_steps: int = 300
    regularization: float = 1e-3
    scaling_factor: float = 1.0
    device: str = "cpu"
    dtype: str = "float64"
    seeds: dict[str, int] = field(
        default_factory=lambda: {
            "training": 3025,
            "reference": 3026,
            "initial_particles": 3027,
            "forward_noise": 3028,
            "reverse_brownian": 3029,
        }
    )

    @property
    def step_size(self) -> float:
        return self.horizon / self.reverse_steps

    def validate(self) -> None:
        if self.dimension != 2:
            raise ValueError("the ring Gaussian mixture has dimension 2")
        if self.components < 2:
            raise ValueError("components must be at least two")
        if self.ring_radius <= 0 or self.component_std <= 0:
            raise ValueError("ring_radius and component_std must be positive")
        if (
            self.training_size < self.components
            or self.reference_size < self.components
        ):
            raise ValueError("dataset sizes must be at least the component count")
        if self.generated_size < 1 or self.horizon <= 0 or self.reverse_steps < 1:
            raise ValueError(
                "generated_size, horizon, and reverse_steps must be positive"
            )
        if self.regularization <= 0 or self.scaling_factor <= 0:
            raise ValueError("regularization and scaling_factor must be positive")
        required = {
            "training",
            "reference",
            "initial_particles",
            "forward_noise",
            "reverse_brownian",
        }
        if set(self.seeds) != required:
            raise ValueError(f"seeds must have exactly these keys: {sorted(required)}")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")


def smoke_config() -> GaussianMixtureConfig:
    """Return a small configuration for tests and pipeline checks."""
    return replace(
        GaussianMixtureConfig(),
        training_size=32,
        reference_size=160,
        generated_size=32,
        horizon=0.3,
        reverse_steps=3,
    )


def component_centers(
    components: int,
    radius: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return equally spaced two-dimensional centers on a circle."""
    if components < 2 or radius <= 0:
        raise ValueError("components must be at least two and radius must be positive")
    angles = (
        math.tau * torch.arange(components, device=device, dtype=dtype) / components
    )
    return radius * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)


def sample_gaussian_mixture(
    size: int,
    components: int,
    radius: float,
    component_std: float,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Sample the uniform ring mixture and return samples and component labels."""
    if size < 1 or component_std <= 0:
        raise ValueError("size and component_std must be positive")
    centers = component_centers(components, radius, device=device, dtype=dtype)
    labels = torch.randint(components, (size,), generator=generator, device=device)
    noise = torch.randn((size, 2), generator=generator, device=device, dtype=dtype)
    return {"samples": centers[labels] + component_std * noise, "labels": labels}


def generate_shared_inputs(
    config: GaussianMixtureConfig,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Generate independent target datasets and all shared OU randomness once."""
    config.validate()
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)
    target_kwargs = {
        "components": config.components,
        "radius": config.ring_radius,
        "component_std": config.component_std,
        "device": device,
        "dtype": dtype,
    }
    training = sample_gaussian_mixture(
        config.training_size,
        generator=_generator(config.seeds["training"], device),
        **target_kwargs,
    )
    reference = sample_gaussian_mixture(
        config.reference_size,
        generator=_generator(config.seeds["reference"], device),
        **target_kwargs,
    )
    counts = {
        "training": torch.bincount(
            training["labels"], minlength=config.components
        ).tolist(),
        "reference": torch.bincount(
            reference["labels"], minlength=config.components
        ).tolist(),
    }
    for dataset_name, values in counts.items():
        if any(value == 0 for value in values):
            raise RuntimeError(
                f"the seeded {dataset_name} realization does not contain all components"
            )
    target = {
        "training": training,
        "reference": reference,
        "component_centers": component_centers(
            config.components, config.ring_radius, device=device, dtype=dtype
        ),
        "component_counts": counts,
    }
    return target, generate_ou_shared_inputs(training["samples"], config)


def plot_saved_run(run_dir: str | Path) -> None:
    """Regenerate Gaussian-mixture figures using only saved artifacts."""
    plot_toy_run(run_dir, xlim=PLOT_LIMITS, ylim=PLOT_LIMITS)


def run_experiment(
    config: GaussianMixtureConfig | None = None,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    progress: bool = True,
    make_figures: bool = True,
) -> Path:
    """Execute and persist one unique Gaussian-mixture experiment run."""
    config = config or GaussianMixtureConfig()
    root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    return run_toy_experiment(
        config,
        output_root=root,
        run_id=run_id,
        progress=progress,
        make_figures=make_figures,
        generate_inputs=generate_shared_inputs,
        metadata=lambda target: {"component_counts": target["component_counts"]},
        config_extras={
            "plot_limits": {"x1": list(PLOT_LIMITS), "x2": list(PLOT_LIMITS)}
        },
        plot_limits=PLOT_LIMITS,
        y_limits=PLOT_LIMITS,
    )


def main(argv: list[str] | None = None) -> None:
    run_cli(
        argv,
        description=__doc__ or "Gaussian-mixture experiment",
        default_output_root=DEFAULT_OUTPUT_ROOT,
        config_factory=GaussianMixtureConfig,
        smoke_factory=smoke_config,
        runner=run_experiment,
    )


if __name__ == "__main__":
    main()
