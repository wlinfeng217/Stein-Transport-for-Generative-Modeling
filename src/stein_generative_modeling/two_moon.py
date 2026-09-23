"""Reproducible two-dimensional two-moons experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from . import experiment as _experiment_support
from . import methods as _method_support
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

# Compatibility names retained for readers and existing client code that used
# the original two-moons module as the shared experiment namespace.
THESIS_PLOT_STYLE = _experiment_support.PLOT_STYLE
_environment = _experiment_support._environment
_json_write = _experiment_support._json_write
_scatter_panel = _experiment_support._scatter_panel
create_run_directory = _experiment_support.create_run_directory
ALL_METHODS = _method_support.ALL_METHODS
DETERMINISTIC_METHODS = _method_support.DETERMINISTIC_METHODS
STOCHASTIC_METHODS = _method_support.STOCHASTIC_METHODS
METHOD_TITLES = _method_support.METHOD_TITLES
run_all_methods = _method_support.run_all_methods

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "two_moon"


@dataclass(frozen=True)
class TwoMoonConfig:
    """Complete scientific and execution configuration for two moons."""

    dimension: int = 2
    training_size: int = 300
    reference_size: int = 10_000
    generated_size: int = 2_000
    target_noise: float = 0.05
    horizon: float = 3.0
    reverse_steps: int = 300
    regularization: float = 1e-3
    scaling_factor: float = 1.0
    device: str = "cpu"
    dtype: str = "float64"
    seeds: dict[str, int] = field(
        default_factory=lambda: {
            "training": 2025,
            "reference": 2026,
            "initial_particles": 2027,
            "forward_noise": 2028,
            "reverse_brownian": 2029,
        }
    )

    @property
    def step_size(self) -> float:
        return self.horizon / self.reverse_steps

    def validate(self) -> None:
        if self.dimension != 2:
            raise ValueError("the two-moons target has dimension 2")
        if self.training_size < 2 or self.reference_size < 2 or self.generated_size < 1:
            raise ValueError("dataset sizes are invalid")
        if self.target_noise < 0 or self.horizon <= 0 or self.reverse_steps < 1:
            raise ValueError("noise, horizon, or reverse_steps is invalid")
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


def smoke_config() -> TwoMoonConfig:
    """Return a tiny deterministic configuration for tests and CI."""
    return replace(
        TwoMoonConfig(),
        training_size=10,
        reference_size=100,
        generated_size=24,
        horizon=0.3,
        reverse_steps=3,
    )


def sample_two_moons(
    size: int,
    noise_scale: float,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Draw ``size`` samples and return their component labels and angles."""
    if size < 1 or noise_scale < 0:
        raise ValueError("size must be positive and noise_scale nonnegative")
    labels = torch.randint(0, 2, (size,), generator=generator, device=device)
    angles = math.pi * torch.rand(size, generator=generator, device=device, dtype=dtype)
    observation_noise = torch.randn(
        (size, 2), generator=generator, device=device, dtype=dtype
    )
    first = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    second = torch.stack((1.0 - torch.cos(angles), 0.5 - torch.sin(angles)), dim=1)
    means = torch.where(labels[:, None].bool(), second, first)
    return {
        "samples": means + noise_scale * observation_noise,
        "labels": labels,
        "angles": angles,
    }


def generate_shared_inputs(
    config: TwoMoonConfig,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Generate target data and every named random tensor exactly once."""
    config.validate()
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)
    training = sample_two_moons(
        config.training_size,
        config.target_noise,
        generator=_generator(config.seeds["training"], device),
        device=device,
        dtype=dtype,
    )
    reference = sample_two_moons(
        config.reference_size,
        config.target_noise,
        generator=_generator(config.seeds["reference"], device),
        device=device,
        dtype=dtype,
    )
    target = {
        "training": training,
        "reference": reference,
        "component_counts": {
            "training": torch.bincount(training["labels"], minlength=2).tolist(),
            "reference": torch.bincount(reference["labels"], minlength=2).tolist(),
        },
    }
    for dataset_name in ("training", "reference"):
        if 0 in target["component_counts"][dataset_name]:
            message = (
                f"the seeded {dataset_name} realization does not contain "
                "both components"
            )
            raise RuntimeError(message)
    return target, generate_ou_shared_inputs(training["samples"], config)


def plot_saved_run(run_dir: str | Path) -> None:
    """Regenerate the two-moons figures using only saved artifacts."""
    plot_toy_run(run_dir)


def run_experiment(
    config: TwoMoonConfig | None = None,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    progress: bool = True,
    make_figures: bool = True,
) -> Path:
    """Execute and persist one unique two-moons experiment run."""
    config = config or TwoMoonConfig()
    root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    return run_toy_experiment(
        config,
        output_root=root,
        run_id=run_id,
        progress=progress,
        make_figures=make_figures,
        generate_inputs=generate_shared_inputs,
        metadata=lambda target: {"component_counts": target["component_counts"]},
    )


def main(argv: list[str] | None = None) -> None:
    run_cli(
        argv,
        description=__doc__ or "Two-moons experiment",
        default_output_root=DEFAULT_OUTPUT_ROOT,
        config_factory=TwoMoonConfig,
        smoke_factory=smoke_config,
        runner=run_experiment,
    )


if __name__ == "__main__":
    main()
