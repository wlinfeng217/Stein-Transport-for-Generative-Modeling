"""Reproducible two-dimensional Swiss-roll toy experiment."""

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

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "swiss_roll"
PLOT_LIMITS = (-3.5, 3.5)


@dataclass(frozen=True)
class SwissRollConfig:
    dimension: int = 2
    parameter_lower: float = 1.5 * math.pi
    parameter_upper: float = 4.5 * math.pi
    pre_scale_noise_std: float = 1.0
    coordinate_scale: float = 5.0
    training_size: int = 300
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
            "training": 5025,
            "reference": 5026,
            "initial_particles": 5027,
            "forward_noise": 5028,
            "reverse_brownian": 5029,
        }
    )

    @property
    def step_size(self) -> float:
        return self.horizon / self.reverse_steps

    def validate(self) -> None:
        if self.dimension != 2:
            raise ValueError("the projected Swiss-roll target has dimension 2")
        if self.parameter_lower < 0 or self.parameter_upper <= self.parameter_lower:
            raise ValueError(
                "parameter_upper must exceed a nonnegative parameter_lower"
            )
        if self.pre_scale_noise_std < 0 or self.coordinate_scale <= 0:
            raise ValueError(
                "noise must be nonnegative and coordinate_scale must be positive"
            )
        if self.training_size < 2 or self.reference_size < 2 or self.generated_size < 1:
            raise ValueError("dataset sizes are invalid")
        if self.horizon <= 0 or self.reverse_steps < 1:
            raise ValueError("horizon and reverse_steps must be positive")
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


def smoke_config() -> SwissRollConfig:
    """Return a small configuration for tests and pipeline checks."""
    return replace(
        SwissRollConfig(),
        training_size=24,
        reference_size=160,
        generated_size=32,
        horizon=0.3,
        reverse_steps=3,
    )


def sample_swiss_roll(
    size: int,
    parameter_lower: float,
    parameter_upper: float,
    pre_scale_noise_std: float,
    coordinate_scale: float,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Sample the scaled two-dimensional projection of a noisy Swiss roll."""
    if size < 1:
        raise ValueError("size must be positive")
    if parameter_lower < 0 or parameter_upper <= parameter_lower:
        raise ValueError("parameter_upper must exceed a nonnegative parameter_lower")
    if pre_scale_noise_std < 0 or coordinate_scale <= 0:
        raise ValueError(
            "noise must be nonnegative and coordinate_scale must be positive"
        )

    uniform = torch.rand(size, generator=generator, device=device, dtype=dtype)
    parameters = parameter_lower + (parameter_upper - parameter_lower) * uniform
    noiseless = torch.stack(
        (parameters * torch.cos(parameters), parameters * torch.sin(parameters)), dim=1
    )
    noise = torch.randn((size, 2), generator=generator, device=device, dtype=dtype)
    samples = (noiseless + pre_scale_noise_std * noise) / coordinate_scale
    return {
        "samples": samples,
        "parameters": parameters,
        "noise": noise,
        "noiseless_samples": noiseless / coordinate_scale,
    }


def generate_shared_inputs(
    config: SwissRollConfig,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Generate independent target datasets and all shared OU randomness once."""
    config.validate()
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)
    target_kwargs = {
        "parameter_lower": config.parameter_lower,
        "parameter_upper": config.parameter_upper,
        "pre_scale_noise_std": config.pre_scale_noise_std,
        "coordinate_scale": config.coordinate_scale,
        "device": device,
        "dtype": dtype,
    }
    training = sample_swiss_roll(
        config.training_size,
        generator=_generator(config.seeds["training"], device),
        **target_kwargs,
    )
    reference = sample_swiss_roll(
        config.reference_size,
        generator=_generator(config.seeds["reference"], device),
        **target_kwargs,
    )
    target = {"training": training, "reference": reference}
    return target, generate_ou_shared_inputs(training["samples"], config)


def plot_saved_run(run_dir: str | Path) -> None:
    """Regenerate Swiss-roll figures using only saved artifacts."""
    plot_toy_run(run_dir, xlim=PLOT_LIMITS, ylim=PLOT_LIMITS)


def run_experiment(
    config: SwissRollConfig | None = None,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    progress: bool = True,
    make_figures: bool = True,
) -> Path:
    """Execute and persist one unique Swiss-roll experiment run."""
    config = config or SwissRollConfig()
    root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    return run_toy_experiment(
        config,
        output_root=root,
        run_id=run_id,
        progress=progress,
        make_figures=make_figures,
        generate_inputs=generate_shared_inputs,
        metadata=lambda target: {
            "parameter_ranges": {
                name: [
                    float(dataset["parameters"].min()),
                    float(dataset["parameters"].max()),
                ]
                for name, dataset in (
                    ("training", target["training"]),
                    ("reference", target["reference"]),
                )
            }
        },
        config_extras={
            "effective_noise_std": config.pre_scale_noise_std / config.coordinate_scale,
            "plot_limits": {"x1": list(PLOT_LIMITS), "x2": list(PLOT_LIMITS)},
        },
        plot_limits=PLOT_LIMITS,
        y_limits=PLOT_LIMITS,
    )


def main(argv: list[str] | None = None) -> None:
    run_cli(
        argv,
        description=__doc__ or "Swiss-roll experiment",
        default_output_root=DEFAULT_OUTPUT_ROOT,
        config_factory=SwissRollConfig,
        smoke_factory=smoke_config,
        runner=run_experiment,
    )


if __name__ == "__main__":
    main()
