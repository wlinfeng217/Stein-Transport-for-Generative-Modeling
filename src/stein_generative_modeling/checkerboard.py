"""Reproducible two-dimensional checkerboard toy experiment."""

from __future__ import annotations

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

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "checkerboard"
PLOT_LIMITS = (-4.5, 4.5)


@dataclass(frozen=True)
class CheckerboardConfig:
    dimension: int = 2
    cells_per_axis: int = 4
    lower_bound: float = -4.0
    upper_bound: float = 4.0
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
            "training": 4025,
            "reference": 4026,
            "initial_particles": 4027,
            "forward_noise": 4028,
            "reverse_brownian": 4029,
        }
    )

    @property
    def step_size(self) -> float:
        return self.horizon / self.reverse_steps

    @property
    def occupied_cells(self) -> int:
        return self.cells_per_axis**2 // 2

    def validate(self) -> None:
        if self.dimension != 2:
            raise ValueError("the checkerboard target has dimension 2")
        if self.cells_per_axis < 2 or self.cells_per_axis % 2:
            raise ValueError("cells_per_axis must be an even integer of at least two")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must exceed lower_bound")
        if self.training_size < self.occupied_cells:
            raise ValueError("training_size must be at least the occupied-cell count")
        if self.reference_size < self.occupied_cells or self.generated_size < 1:
            raise ValueError("reference_size or generated_size is invalid")
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


def smoke_config() -> CheckerboardConfig:
    """Return a small configuration for tests and pipeline checks."""
    return replace(
        CheckerboardConfig(),
        training_size=32,
        reference_size=160,
        generated_size=32,
        horizon=0.3,
        reverse_steps=3,
    )


def sample_checkerboard(
    size: int,
    cells_per_axis: int,
    lower_bound: float,
    upper_bound: float,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Sample uniformly from even-parity cells of a square checkerboard."""
    if size < 1:
        raise ValueError("size must be positive")
    if cells_per_axis < 2 or cells_per_axis % 2:
        raise ValueError("cells_per_axis must be an even integer of at least two")
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound must exceed lower_bound")

    columns = torch.randint(cells_per_axis, (size,), generator=generator, device=device)
    row_pairs = torch.randint(
        cells_per_axis // 2, (size,), generator=generator, device=device
    )
    rows = 2 * row_pairs + torch.remainder(columns, 2)
    cell_indices = torch.stack((columns, rows), dim=1)
    offsets = torch.rand((size, 2), generator=generator, device=device, dtype=dtype)
    cell_width = (upper_bound - lower_bound) / cells_per_axis
    samples = lower_bound + cell_width * (cell_indices.to(dtype=dtype) + offsets)
    cell_ids = columns * cells_per_axis + rows
    return {
        "samples": samples,
        "cell_indices": cell_indices,
        "cell_ids": cell_ids,
        "offsets": offsets,
    }


def generate_shared_inputs(
    config: CheckerboardConfig,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Generate independent target datasets and all shared OU randomness once."""
    config.validate()
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)
    target_kwargs = {
        "cells_per_axis": config.cells_per_axis,
        "lower_bound": config.lower_bound,
        "upper_bound": config.upper_bound,
        "device": device,
        "dtype": dtype,
    }
    training = sample_checkerboard(
        config.training_size,
        generator=_generator(config.seeds["training"], device),
        **target_kwargs,
    )
    reference = sample_checkerboard(
        config.reference_size,
        generator=_generator(config.seeds["reference"], device),
        **target_kwargs,
    )
    occupied_ids = [
        column * config.cells_per_axis + row
        for column in range(config.cells_per_axis)
        for row in range(config.cells_per_axis)
        if (column + row) % 2 == 0
    ]
    counts = {
        name: [
            int(torch.count_nonzero(dataset["cell_ids"] == cell_id))
            for cell_id in occupied_ids
        ]
        for name, dataset in (("training", training), ("reference", reference))
    }
    for dataset_name, values in counts.items():
        if any(value == 0 for value in values):
            message = (
                f"the seeded {dataset_name} realization does not contain all "
                "occupied cells"
            )
            raise RuntimeError(message)
    target = {
        "training": training,
        "reference": reference,
        "occupied_cell_ids": occupied_ids,
        "cell_counts": counts,
    }
    return target, generate_ou_shared_inputs(training["samples"], config)


def plot_saved_run(run_dir: str | Path) -> None:
    """Regenerate checkerboard figures using only saved artifacts."""
    plot_toy_run(run_dir, xlim=PLOT_LIMITS, ylim=PLOT_LIMITS)


def run_experiment(
    config: CheckerboardConfig | None = None,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    progress: bool = True,
    make_figures: bool = True,
) -> Path:
    """Execute and persist one unique checkerboard experiment run."""
    config = config or CheckerboardConfig()
    root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    return run_toy_experiment(
        config,
        output_root=root,
        run_id=run_id,
        progress=progress,
        make_figures=make_figures,
        generate_inputs=generate_shared_inputs,
        metadata=lambda target: {
            "occupied_cell_ids": target["occupied_cell_ids"],
            "cell_counts": target["cell_counts"],
        },
        config_extras={
            "occupied_cells": config.occupied_cells,
            "plot_limits": {"x1": list(PLOT_LIMITS), "x2": list(PLOT_LIMITS)},
        },
        plot_limits=PLOT_LIMITS,
        y_limits=PLOT_LIMITS,
    )


def main(argv: list[str] | None = None) -> None:
    run_cli(
        argv,
        description=__doc__ or "Checkerboard experiment",
        default_output_root=DEFAULT_OUTPUT_ROOT,
        config_factory=CheckerboardConfig,
        smoke_factory=smoke_config,
        runner=run_experiment,
    )


if __name__ == "__main__":
    main()
