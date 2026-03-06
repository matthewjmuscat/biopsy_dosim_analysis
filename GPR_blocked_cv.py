from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
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


def _voxel_position_table(
    g: pd.DataFrame,
    *,
    position_mode: str = "begin",
) -> pd.DataFrame:
    """Build per-voxel axial position table for one biopsy."""
    if {"Voxel begin (Z)", "Voxel end (Z)"}.issubset(g.columns):
        pos = g.groupby("Voxel index").agg(
            x_mm_begin=("Voxel begin (Z)", "first"),
            x_mm_end=("Voxel end (Z)", "first"),
        )
        if position_mode == "center":
            pos["x_mm"] = 0.5 * (pos["x_mm_begin"] + pos["x_mm_end"])
        else:
            pos["x_mm"] = pos["x_mm_begin"]
    else:
        z_col = "Z (Bx frame)"
        agg_fn = "mean" if position_mode == "center" else "min"
        pos = g.groupby("Voxel index").agg(x_mm=(z_col, agg_fn))
    out = pos.reset_index().sort_values("x_mm", kind="stable").reset_index(drop=True)
    return out


def _assign_folds_equal_voxels(n_vox: int, n_folds: int) -> np.ndarray:
    """Assign contiguous equal-size voxel blocks to fold ids [1..K]."""
    k_use = max(1, min(int(n_folds), int(n_vox)))
    fold_ids = np.empty(n_vox, dtype=int)
    chunks = np.array_split(np.arange(n_vox), k_use)
    for fid, idxs in enumerate(chunks, start=1):
        fold_ids[idxs] = fid
    return fold_ids


def _assign_folds_fixed_mm(
    x_mm: np.ndarray,
    *,
    n_folds: int,
    block_length_mm: float | None,
    min_block_mm: float,
) -> np.ndarray:
    """
    Assign folds by contiguous physical-length bins.
    If block_length_mm is None, derive it from span/n_folds.
    """
    x = np.asarray(x_mm, dtype=float)
    z_min = float(np.nanmin(x))
    z_max = float(np.nanmax(x))
    span = max(0.0, z_max - z_min)

    if block_length_mm is None:
        length = span / max(1, int(n_folds))
    else:
        length = float(block_length_mm)
    length = max(float(min_block_mm), length)
    if length <= 0:
        length = 1.0

    # Bin by physical position; fold ids are 1-based.
    raw = np.floor((x - z_min) / length).astype(int)
    if raw.size:
        raw = raw - raw.min()
    fold_ids = raw + 1
    return fold_ids.astype(int)


def build_blocked_cv_fold_map(
    all_voxel_wise_dose_df: pd.DataFrame,
    *,
    config: BlockedCVConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build blocked_CV fold assignment tables (no model fitting).

    Returns
    -------
    fold_map_df:
        Long table with one row per (biopsy, fold_id, voxel), including is_test.
    fold_summary_df:
        One row per (biopsy, fold_id) with train/test counts and z-range.
    """
    fold_map_rows = []
    fold_summary_rows = []

    grp_cols = ["Patient ID", "Bx index"]
    for (patient_id, bx_index), g in all_voxel_wise_dose_df.groupby(grp_cols):
        vox = _voxel_position_table(g, position_mode=config.position_mode)
        n_vox = int(len(vox))
        if n_vox == 0:
            continue

        if config.block_mode == "equal_voxels":
            fold_of_voxel = _assign_folds_equal_voxels(n_vox, config.n_folds)
        elif config.block_mode == "fixed_mm":
            fold_of_voxel = _assign_folds_fixed_mm(
                vox["x_mm"].to_numpy(float),
                n_folds=config.n_folds,
                block_length_mm=config.block_length_mm,
                min_block_mm=config.min_block_mm,
            )
        else:
            raise ValueError(
                f"Unsupported blocked_CV block_mode '{config.block_mode}'. "
                "Use 'equal_voxels' or 'fixed_mm'."
            )

        vox = vox.copy()
        vox["test_fold_id"] = fold_of_voxel
        fold_ids = np.sort(np.unique(fold_of_voxel))

        for fold_id in fold_ids:
            test_mask = vox["test_fold_id"] == fold_id
            n_test = int(test_mask.sum())
            n_train = int(n_vox - n_test)
            z_test = vox.loc[test_mask, "x_mm"].to_numpy(float)
            z_min = float(np.nanmin(z_test)) if z_test.size else np.nan
            z_max = float(np.nanmax(z_test)) if z_test.size else np.nan
            idx_test = np.flatnonzero(test_mask.to_numpy())
            contiguous = bool(idx_test.size and np.all(np.diff(idx_test) == 1))

            fold_summary_rows.append(
                {
                    "Patient ID": patient_id,
                    "Bx index": bx_index,
                    "fold_id": int(fold_id),
                    "block_mode": config.block_mode,
                    "position_mode": config.position_mode,
                    "n_voxels": n_vox,
                    "n_train": n_train,
                    "n_test": n_test,
                    "test_z_min_mm": z_min,
                    "test_z_max_mm": z_max,
                    "contiguous_test_block": contiguous,
                }
            )

            for _, r in vox.iterrows():
                is_test = bool(int(r["test_fold_id"]) == int(fold_id))
                fold_map_rows.append(
                    {
                        "Patient ID": patient_id,
                        "Bx index": bx_index,
                        "fold_id": int(fold_id),
                        "Voxel index": int(r["Voxel index"]),
                        "x_mm": float(r["x_mm"]),
                        "is_test": is_test,
                        "n_train": n_train,
                        "n_test": n_test,
                        "test_z_min_mm": z_min,
                        "test_z_max_mm": z_max,
                        "block_mode": config.block_mode,
                        "position_mode": config.position_mode,
                    }
                )

    fold_map_df = pd.DataFrame(fold_map_rows)
    fold_summary_df = pd.DataFrame(fold_summary_rows)
    return fold_map_df, fold_summary_df


def run_blocked_cv_phase3b(
    all_voxel_wise_dose_df: pd.DataFrame,
    semivariogram_df: pd.DataFrame,
    *,
    output_dir: Path,
    figs_dir: Path,
    csv_dir: Path,
    config: BlockedCVConfig,
) -> dict:
    """
    Phase 3B entrypoint: construct blocked fold maps and save CSV artifacts.
    """
    fold_map_df, fold_summary_df = build_blocked_cv_fold_map(
        all_voxel_wise_dose_df,
        config=config,
    )
    fold_map_path = csv_dir.joinpath("blocked_cv_fold_map.csv")
    fold_summary_path = csv_dir.joinpath("blocked_cv_fold_summary.csv")
    fold_map_df.to_csv(fold_map_path, index=False)
    fold_summary_df.to_csv(fold_summary_path, index=False)

    status = {
        "phase": "3B_fold_map",
        "status": "ready",
        "blocked_cv_root": str(output_dir),
        "blocked_cv_figs_dir": str(figs_dir),
        "blocked_cv_csv_dir": str(csv_dir),
        "n_biopsies_seen": int(
            all_voxel_wise_dose_df[["Patient ID", "Bx index"]].drop_duplicates().shape[0]
        ) if {"Patient ID", "Bx index"}.issubset(all_voxel_wise_dose_df.columns) else 0,
        "semivariogram_rows_seen": int(len(semivariogram_df)),
        "block_mode": config.block_mode,
        "n_folds": int(config.n_folds),
        "position_mode": config.position_mode,
        "target_stat": config.target_stat,
        "mean_mode": config.mean_mode,
        "predictive_variance_mode": config.predictive_variance_mode,
        "fold_map_csv": str(fold_map_path),
        "fold_summary_csv": str(fold_summary_path),
        "n_fold_map_rows": int(len(fold_map_df)),
        "n_fold_summary_rows": int(len(fold_summary_df)),
    }
    return status
