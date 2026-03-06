from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


@dataclass
class BlockedCVConfig:
    """
    Scaffold config for blocked_CV pathway.

    This is intentionally lightweight for Phase 3A; fit/predict logic is added
    in later phases.
    """

    block_mode: str = "equal_voxels"  # "equal_voxels" or "fixed_mm"
    n_folds: int = 5
    block_length_mm: float | None = None
    min_block_mm: float = 5.0
    position_mode: str = "begin"
    target_stat: str = "median"
    mean_mode: str = "ordinary"
    predictive_variance_mode: str = "observed_mc"
    kernel_specs: Iterable[Tuple[str, float | None, str]] | None = None
    write_per_kernel_point_csvs: bool = False
    write_per_kernel_metrics_csvs: bool = False
    write_per_kernel_summary_csvs: bool = False


def init_blocked_cv_dirs(output_dir: Path, subdir_name: str = "blocked_CV") -> tuple[Path, Path, Path]:
    """
    Initialize blocked_CV directory scaffold.
    Returns (root_dir, figures_dir, csv_dir).
    """
    root = output_dir.joinpath(subdir_name)
    figs = root.joinpath("figures")
    csv = root.joinpath("csv")
    root.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    csv.mkdir(parents=True, exist_ok=True)
    return root, figs, csv


def run_blocked_cv_scaffold(
    all_voxel_wise_dose_df: pd.DataFrame,
    semivariogram_df: pd.DataFrame,
    *,
    output_dir: Path,
    figs_dir: Path,
    csv_dir: Path,
    config: BlockedCVConfig,
) -> dict:
    """
    Phase 3A scaffold entrypoint for blocked_CV.

    This does not run CV yet. It validates that dataframes are present, confirms
    options, and records a status dict for logging/debugging.
    """
    n_bx = int(
        all_voxel_wise_dose_df[["Patient ID", "Bx index"]].drop_duplicates().shape[0]
    ) if {"Patient ID", "Bx index"}.issubset(all_voxel_wise_dose_df.columns) else 0

    status = {
        "phase": "3A_scaffold",
        "status": "ready",
        "blocked_cv_root": str(output_dir),
        "blocked_cv_figs_dir": str(figs_dir),
        "blocked_cv_csv_dir": str(csv_dir),
        "n_biopsies_seen": n_bx,
        "semivariogram_rows_seen": int(len(semivariogram_df)),
        "block_mode": config.block_mode,
        "n_folds": int(config.n_folds),
        "position_mode": config.position_mode,
        "target_stat": config.target_stat,
        "mean_mode": config.mean_mode,
        "predictive_variance_mode": config.predictive_variance_mode,
    }

    return status

