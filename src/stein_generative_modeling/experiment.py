"""Shared reproducibility, persistence, CLI, and plotting utilities."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from .methods import (
    ALL_METHODS,
    DETERMINISTIC_METHODS,
    METHOD_TITLES,
    STOCHASTIC_METHODS,
    run_all_methods,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 13.0,
    "axes.titlesize": 14.0,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 13.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


class ToyConfig(Protocol):
    """Structural interface shared by the four target configurations."""

    training_size: int
    generated_size: int
    reverse_steps: int
    horizon: float
    device: str
    dtype: str
    seeds: dict[str, int]

    @property
    def step_size(self) -> float: ...

    def validate(self) -> None: ...


def _generator(seed: int, device: torch.device) -> torch.Generator:
    """Create an explicitly seeded generator on ``device``."""
    return torch.Generator(device=device).manual_seed(seed)


def generate_ou_shared_inputs(training: Tensor, config: ToyConfig) -> dict[str, Tensor]:
    """Generate tensor-identical randomness for a fair eight-method comparison."""
    device = training.device
    dtype = training.dtype
    dimension = training.shape[1]
    forward_noise = torch.randn(
        (config.reverse_steps, config.training_size, dimension),
        generator=_generator(config.seeds["forward_noise"], device),
        device=device,
        dtype=dtype,
    )
    forward_times = (
        torch.arange(config.reverse_steps, device=device, dtype=dtype)
        .neg()
        .mul(config.step_size)
        .add(config.horizon)
    )
    a = torch.exp(-forward_times)[:, None, None]
    sigma = torch.sqrt(-torch.expm1(-2.0 * forward_times))[:, None, None]
    noised_training = a * training[None, :, :] + sigma * forward_noise
    return {
        "initial_particles": torch.randn(
            (config.generated_size, dimension),
            generator=_generator(config.seeds["initial_particles"], device),
            device=device,
            dtype=dtype,
        ),
        "forward_times": forward_times,
        "forward_noise": forward_noise,
        "noised_training": noised_training,
        "reverse_brownian": torch.randn(
            (config.reverse_steps, config.generated_size, dimension),
            generator=_generator(config.seeds["reverse_brownian"], device),
            device=device,
            dtype=dtype,
        ),
    }


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _environment(config: ToyConfig) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": sys.version,
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": config.device,
        "dtype": config.dtype,
        "code_revision": revision,
    }


def create_run_directory(
    output_root: Path, config: ToyConfig, run_id: str | None
) -> Path:
    """Create a unique result directory and refuse accidental overwrites."""
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = f"seed{config.seeds['training']}_{stamp}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_toy_experiment(
    config: ToyConfig,
    *,
    output_root: Path,
    run_id: str | None,
    progress: bool,
    make_figures: bool,
    generate_inputs: Callable[[Any], tuple[dict[str, Any], dict[str, Tensor]]],
    metadata: Callable[[dict[str, Any]], dict[str, Any]],
    config_extras: Mapping[str, Any] | None = None,
    plot_limits: tuple[float, float] = (-1.5, 2.5),
    y_limits: tuple[float, float] = (-1.0, 1.5),
) -> Path:
    """Run, save, and optionally plot one immutable toy experiment."""
    config.validate()
    run_dir = create_run_directory(output_root, config, run_id)
    resolved = asdict(config) | {
        "step_size": config.step_size,
        "methods": list(ALL_METHODS),
        "method_labels": dict(METHOD_TITLES),
    }
    if config_extras:
        resolved.update(config_extras)
    _json_write(run_dir / "config.json", resolved)
    _json_write(run_dir / "environment.json", _environment(config))

    target, shared = generate_inputs(config)
    torch.save(target, run_dir / "target_data.pt")
    torch.save(shared, run_dir / "shared_randomness.pt")
    outputs = run_all_methods(
        target["training"]["samples"], shared, config, progress=progress
    )
    torch.save(outputs, run_dir / "samples.pt")
    run_metadata = metadata(target) | {
        "failures": {
            name: item["failure"]
            for name, item in outputs.items()
            if item["status"] == "failed"
        }
    }
    _json_write(run_dir / "run_metadata.json", run_metadata)
    if make_figures:
        plot_saved_run(run_dir, xlim=plot_limits, ylim=y_limits)
    return run_dir


def _scatter_panel(
    ax: Any,
    reference: Tensor,
    result: Mapping[str, Any],
    name: str,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.scatter(
        reference[:, 0],
        reference[:, 1],
        s=4,
        c="0.82",
        alpha=0.5,
        label="reference",
        rasterized=True,
    )
    if result["status"] == "success":
        samples = result["samples"]
        finite = torch.isfinite(samples).all(dim=1)
        visible = samples[finite]
        ax.scatter(
            visible[:, 0],
            visible[:, 1],
            s=7,
            c=color,
            alpha=0.45,
            label="generated",
            rasterized=True,
        )
        outside = finite & (
            (samples[:, 0] < xlim[0])
            | (samples[:, 0] > xlim[1])
            | (samples[:, 1] < ylim[0])
            | (samples[:, 1] > ylim[1])
        )
        ax.text(
            0.02,
            0.98,
            f"outside: {int(outside.sum())}",
            transform=ax.transAxes,
            va="top",
            fontsize=11.5,
        )
    else:
        failure = result["failure"]
        ax.text(
            0.5,
            0.5,
            f"FAILED\n{failure['exception_type']}: {failure['message']}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            wrap=True,
        )
    ax.set(title=METHOD_TITLES[name], xlim=xlim, ylim=ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def plot_saved_run(
    run_dir: str | Path,
    *,
    xlim: tuple[float, float] = (-1.5, 2.5),
    ylim: tuple[float, float] = (-1.0, 1.5),
) -> None:
    """Regenerate PNG and PDF figures using saved artifacts only."""
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    target = torch.load(
        run_dir / "target_data.pt", map_location="cpu", weights_only=False
    )
    outputs = torch.load(run_dir / "samples.pt", map_location="cpu", weights_only=False)
    reference = target["reference"]["samples"]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4.2), layout="constrained")
        ax.scatter(
            reference[:, 0],
            reference[:, 1],
            s=4,
            c="0.78",
            alpha=0.55,
            label="reference",
            rasterized=True,
        )
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False)
        for suffix in ("png", "pdf"):
            fig.savefig(run_dir / f"reference.{suffix}", dpi=300)
        plt.close(fig)

        palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
        fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.5), layout="constrained")
        rows: Sequence[Sequence[str]] = (
            DETERMINISTIC_METHODS,
            STOCHASTIC_METHODS,
        )
        for row, methods in enumerate(rows):
            for ax, name, color in zip(axes[row], methods, palette, strict=True):
                _scatter_panel(ax, reference, outputs[name], name, color, xlim, ylim)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncols=len(labels),
            frameon=False,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(run_dir / f"comparison.{suffix}", dpi=300)
        plt.close(fig)


def run_cli(
    argv: list[str] | None,
    *,
    description: str,
    default_output_root: Path,
    config_factory: Callable[[], ToyConfig],
    smoke_factory: Callable[[], ToyConfig],
    runner: Callable[..., Path],
) -> None:
    """Parse the common command-line interface for a toy target."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-root",
        default=None,
        help=f"result parent directory (default: {default_output_root})",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--smoke", action="store_true", help="run a tiny integration configuration"
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    config = smoke_factory() if args.smoke else config_factory()
    path = runner(
        config,
        output_root=args.output_root,
        run_id=args.run_id,
        progress=not args.no_progress,
        make_figures=not args.no_figures,
    )
    print(path)
