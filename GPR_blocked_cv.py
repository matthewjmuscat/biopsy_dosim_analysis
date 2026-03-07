from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import GPR_analysis_pipeline_functions as gpr_pf
import GPR_semivariogram as gpr_sv


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
    min_derived_block_mm: float = 5.0
    merge_tiny_tail_folds: bool = False
    min_test_voxels: int = 1
    min_test_block_mm: float = 0.0
    min_effective_folds_after_merge: int = 2
    rebalance_two_fold_splits: bool = False
    position_mode: str = "begin"
    target_stat: str = "median"
    mean_mode: str = "ordinary"
    primary_predictive_variance_mode: str = "observed_mc"
    compare_variance_modes: bool = False
    variance_modes_to_compare: Iterable[str] | None = None
    kernel_specs: Iterable[Tuple[str, float | None, str]] | None = None
    semivariogram_voxel_size_mm: float = 1.0
    semivariogram_lag_bin_width_mm: float | None = None
    write_per_kernel_predictions_csvs: bool = False
    write_per_kernel_fit_status_csvs: bool = False
    write_per_kernel_variance_compare_csvs: bool = False
    write_per_kernel_variance_summary_csvs: bool = False


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
    _blocked_cv_csv_subdirs(csv)
    return root, figs, csv


def _blocked_cv_csv_subdirs(csv_root: Path) -> dict[str, Path]:
    """
    Ensure blocked_CV CSV subfolders exist and return their paths.

    Layout:
    - folds: fold assignments and fold-map summaries
    - predictions: held-out point prediction tables
    - metrics: aggregated performance summaries
    - diagnostics: fit status and troubleshooting telemetry
    """
    csv_root.mkdir(parents=True, exist_ok=True)
    subdirs = {
        "root": csv_root,
        "folds": csv_root.joinpath("folds"),
        "predictions": csv_root.joinpath("predictions"),
        "metrics": csv_root.joinpath("metrics"),
        "diagnostics": csv_root.joinpath("diagnostics"),
    }
    for key, p in subdirs.items():
        if key == "root":
            continue
        p.mkdir(parents=True, exist_ok=True)
    return subdirs


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
    csv_subdirs = _blocked_cv_csv_subdirs(csv_dir)

    status = {
        "phase": "3A_scaffold",
        "status": "ready",
        "blocked_cv_root": str(output_dir),
        "blocked_cv_figs_dir": str(figs_dir),
        "blocked_cv_csv_dir": str(csv_dir),
        "blocked_cv_csv_folds_dir": str(csv_subdirs["folds"]),
        "blocked_cv_csv_predictions_dir": str(csv_subdirs["predictions"]),
        "blocked_cv_csv_metrics_dir": str(csv_subdirs["metrics"]),
        "blocked_cv_csv_diagnostics_dir": str(csv_subdirs["diagnostics"]),
        "n_biopsies_seen": n_bx,
        "semivariogram_rows_seen": int(len(semivariogram_df)),
        "block_mode": config.block_mode,
        "n_folds": int(config.n_folds),
        "min_derived_block_mm": float(config.min_derived_block_mm),
        "merge_tiny_tail_folds": bool(config.merge_tiny_tail_folds),
        "rebalance_two_fold_splits": bool(config.rebalance_two_fold_splits),
        "min_test_voxels": int(config.min_test_voxels),
        "min_test_block_mm": float(config.min_test_block_mm),
        "min_effective_folds_after_merge": int(config.min_effective_folds_after_merge),
        "position_mode": config.position_mode,
        "target_stat": config.target_stat,
        "mean_mode": config.mean_mode,
        "primary_predictive_variance_mode": config.primary_predictive_variance_mode,
        "compare_variance_modes": bool(config.compare_variance_modes),
        "variance_modes_to_compare": list(config.variance_modes_to_compare) if config.variance_modes_to_compare is not None else None,
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
    min_derived_block_mm: float,
) -> np.ndarray:
    """
    Assign folds by contiguous physical-length bins.
    If block_length_mm is None, derive it from span/n_folds and apply
    min_derived_block_mm as a floor.
    """
    x = np.asarray(x_mm, dtype=float)
    z_min = float(np.nanmin(x))
    z_max = float(np.nanmax(x))
    span = max(0.0, z_max - z_min)

    if block_length_mm is None:
        length = span / max(1, int(n_folds))
        length = max(float(min_derived_block_mm), length)
    else:
        length = float(block_length_mm)
    if length <= 0:
        length = 1.0

    # Bin by physical position; fold ids are 1-based.
    raw = np.floor((x - z_min) / length).astype(int)
    if raw.size:
        raw = raw - raw.min()
    fold_ids = raw + 1
    return fold_ids.astype(int)


def _rebalance_two_fold_splits_by_voxel_count(
    fold_ids: np.ndarray,
    x_mm: np.ndarray,
    *,
    min_test_voxels: int,
    min_test_block_mm: float,
) -> tuple[np.ndarray, bool]:
    """
    Rebalance fixed-mm two-fold cases into near-equal contiguous voxel-count halves.

    This avoids pathological tiny second folds when fixed bins leave short remainders.
    The override only triggers when exactly two folds exist and one fold violates
    count/span thresholds.
    """
    ids = np.asarray(fold_ids, dtype=int).copy()
    x = np.asarray(x_mm, dtype=float)
    uniq = np.sort(np.unique(ids))
    if uniq.size != 2:
        return ids, False

    min_vox = max(int(min_test_voxels), 1)
    min_span = max(float(min_test_block_mm), 0.0)

    def _fold_metrics(mask: np.ndarray) -> tuple[int, float]:
        n = int(mask.sum())
        if n > 1 and np.any(np.isfinite(x[mask])):
            span = float(np.nanmax(x[mask]) - np.nanmin(x[mask]))
        else:
            span = 0.0
        return n, span

    m1 = ids == int(uniq[0])
    m2 = ids == int(uniq[1])
    n1, s1 = _fold_metrics(m1)
    n2, s2 = _fold_metrics(m2)
    need_rebalance = (n1 < min_vox) or (n2 < min_vox) or (s1 < min_span) or (s2 < min_span)
    if not need_rebalance:
        return ids, False

    # Split contiguously by ordered axial position to n/n or n/(n+1).
    order = np.argsort(x, kind="stable")
    n = int(len(order))
    if n < 2:
        return ids, False
    n_left = n // 2
    if n_left < 1 or (n - n_left) < 1:
        return ids, False

    left_id, right_id = int(uniq[0]), int(uniq[1])
    new_ids = np.empty(n, dtype=int)
    new_ids[order[:n_left]] = left_id
    new_ids[order[n_left:]] = right_id
    return new_ids, True


def _merge_tiny_tail_fold_ids(
    fold_ids: np.ndarray,
    x_mm: np.ndarray,
    *,
    min_test_voxels: int,
    min_test_block_mm: float,
    min_effective_folds_after_merge: int,
) -> tuple[np.ndarray, bool]:
    """
    Merge tiny remainder tail folds into the previous fold (fixed_mm mode).

    This prevents pathological 1-voxel/near-zero-span held-out folds that are
    overly easy and can bias blocked_CV metrics optimistic.
    """
    ids = np.asarray(fold_ids, dtype=int).copy()
    x = np.asarray(x_mm, dtype=float)
    merged = False

    min_vox = max(int(min_test_voxels), 1)
    min_span = max(float(min_test_block_mm), 0.0)
    min_eff = max(int(min_effective_folds_after_merge), 1)

    while True:
        uniq = np.sort(np.unique(ids))
        if uniq.size <= min_eff:
            break
        tail = int(uniq[-1])
        prev = int(uniq[-2])
        m_tail = ids == tail
        n_tail = int(m_tail.sum())
        if n_tail <= 0:
            break
        x_tail = x[m_tail]
        if n_tail > 1 and np.any(np.isfinite(x_tail)):
            span_tail = float(np.nanmax(x_tail) - np.nanmin(x_tail))
        else:
            span_tail = 0.0
        tiny_tail = (n_tail < min_vox) or (span_tail < min_span)
        if not tiny_tail:
            break
        ids[m_tail] = prev
        merged = True

    # Renumber to contiguous 1..K labels to keep outputs tidy.
    uniq = np.sort(np.unique(ids))
    remap = {int(old): i + 1 for i, old in enumerate(uniq)}
    ids = np.array([remap[int(v)] for v in ids], dtype=int)
    return ids, merged


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
        bx_id_value = g["Bx ID"].dropna().iloc[0] if ("Bx ID" in g.columns and g["Bx ID"].notna().any()) else None
        vox = _voxel_position_table(g, position_mode=config.position_mode)
        n_vox = int(len(vox))
        if n_vox == 0:
            continue
        merged_tail_fold = False
        rebalance_two_fold_applied = False
        split_strategy = config.block_mode

        if config.block_mode == "equal_voxels":
            fold_of_voxel = _assign_folds_equal_voxels(n_vox, config.n_folds)
        elif config.block_mode == "fixed_mm":
            fold_of_voxel = _assign_folds_fixed_mm(
                vox["x_mm"].to_numpy(float),
                n_folds=config.n_folds,
                block_length_mm=config.block_length_mm,
                min_derived_block_mm=config.min_derived_block_mm,
            )
            if config.merge_tiny_tail_folds:
                fold_of_voxel, merged_tail_fold = _merge_tiny_tail_fold_ids(
                    fold_of_voxel,
                    vox["x_mm"].to_numpy(float),
                    min_test_voxels=config.min_test_voxels,
                    min_test_block_mm=config.min_test_block_mm,
                    min_effective_folds_after_merge=config.min_effective_folds_after_merge,
                )
            if config.rebalance_two_fold_splits:
                fold_of_voxel, rebalance_two_fold_applied = _rebalance_two_fold_splits_by_voxel_count(
                    fold_of_voxel,
                    vox["x_mm"].to_numpy(float),
                    min_test_voxels=config.min_test_voxels,
                    min_test_block_mm=config.min_test_block_mm,
                )
            if rebalance_two_fold_applied:
                split_strategy = "fixed_mm_rebalanced_two_fold"
            elif merged_tail_fold:
                split_strategy = "fixed_mm_tail_merge"
            else:
                split_strategy = "fixed_mm"
        else:
            raise ValueError(
                f"Unsupported blocked_CV block_mode '{config.block_mode}'. "
                "Use 'equal_voxels' or 'fixed_mm'."
            )

        vox = vox.copy()
        vox["test_fold_id"] = fold_of_voxel
        fold_ids = np.sort(np.unique(fold_of_voxel))
        effective_n_folds = int(len(fold_ids))

        for fold_id in fold_ids:
            test_mask = vox["test_fold_id"] == fold_id
            n_test = int(test_mask.sum())
            n_train = int(n_vox - n_test)
            n_train_plus_test = int(n_train + n_test)
            train_test_total_match = bool(n_train_plus_test == n_vox)
            z_test = vox.loc[test_mask, "x_mm"].to_numpy(float)
            z_min = float(np.nanmin(z_test)) if z_test.size else np.nan
            z_max = float(np.nanmax(z_test)) if z_test.size else np.nan
            test_span_mm = float(z_max - z_min) if np.isfinite(z_min) and np.isfinite(z_max) else np.nan
            tail_merge_rule_active = bool(config.block_mode == "fixed_mm" and config.merge_tiny_tail_folds)
            test_meets_min_voxels_threshold = bool(n_test >= int(config.min_test_voxels))
            test_meets_min_span_threshold = bool(test_span_mm >= float(config.min_test_block_mm)) if np.isfinite(test_span_mm) else False
            idx_test = np.flatnonzero(test_mask.to_numpy())
            contiguous = bool(idx_test.size and np.all(np.diff(idx_test) == 1))

            fold_summary_rows.append(
                {
                    "Patient ID": patient_id,
                    "Bx ID": bx_id_value,
                    "Bx index": bx_index,
                    "fold_id": int(fold_id),
                    "block_mode": config.block_mode,
                    "position_mode": config.position_mode,
                    "n_voxels": n_vox,
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_train_plus_test": n_train_plus_test,
                    "train_test_total_match": train_test_total_match,
                    "effective_n_folds": effective_n_folds,
                    "merged_tail_fold": bool(merged_tail_fold),
                    "rebalanced_two_fold_split": bool(rebalance_two_fold_applied),
                    "split_strategy": split_strategy,
                    "test_z_min_mm": z_min,
                    "test_z_max_mm": z_max,
                    "test_span_mm": test_span_mm,
                    "tail_merge_rule_active": tail_merge_rule_active,
                    "test_meets_min_voxels_threshold": test_meets_min_voxels_threshold,
                    "test_meets_min_span_threshold": test_meets_min_span_threshold,
                    "contiguous_test_block": contiguous,
                }
            )

            for _, r in vox.iterrows():
                is_test = bool(int(r["test_fold_id"]) == int(fold_id))
                fold_map_rows.append(
                    {
                        "Patient ID": patient_id,
                        "Bx ID": bx_id_value,
                        "Bx index": bx_index,
                        "fold_id": int(fold_id),
                        "Voxel index": int(r["Voxel index"]),
                        "x_mm": float(r["x_mm"]),
                        "is_test": is_test,
                        "n_voxels": n_vox,
                        "n_train": n_train,
                        "n_test": n_test,
                        "n_train_plus_test": n_train_plus_test,
                        "train_test_total_match": train_test_total_match,
                        "effective_n_folds": effective_n_folds,
                        "merged_tail_fold": bool(merged_tail_fold),
                        "rebalanced_two_fold_split": bool(rebalance_two_fold_applied),
                        "split_strategy": split_strategy,
                        "test_z_min_mm": z_min,
                        "test_z_max_mm": z_max,
                        "test_span_mm": test_span_mm,
                        "tail_merge_rule_active": tail_merge_rule_active,
                        "test_meets_min_voxels_threshold": test_meets_min_voxels_threshold,
                        "test_meets_min_span_threshold": test_meets_min_span_threshold,
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
    csv_subdirs = _blocked_cv_csv_subdirs(csv_dir)
    fold_map_path = csv_subdirs["folds"].joinpath("blocked_cv_fold_map.csv")
    fold_summary_path = csv_subdirs["folds"].joinpath("blocked_cv_fold_summary.csv")
    fold_map_df.to_csv(fold_map_path, index=False)
    fold_summary_df.to_csv(fold_summary_path, index=False)
    if not fold_summary_df.empty and {"Patient ID", "Bx index", "merged_tail_fold"}.issubset(fold_summary_df.columns):
        merged_bx_count = int(
            fold_summary_df.loc[fold_summary_df["merged_tail_fold"], ["Patient ID", "Bx index"]]
            .drop_duplicates()
            .shape[0]
        )
    else:
        merged_bx_count = 0
    if not fold_summary_df.empty and {"Patient ID", "Bx index", "rebalanced_two_fold_split"}.issubset(fold_summary_df.columns):
        rebalanced_two_fold_bx_count = int(
            fold_summary_df.loc[fold_summary_df["rebalanced_two_fold_split"], ["Patient ID", "Bx index"]]
            .drop_duplicates()
            .shape[0]
        )
    else:
        rebalanced_two_fold_bx_count = 0

    status = {
        "phase": "3B_fold_map",
        "status": "ready",
        "blocked_cv_root": str(output_dir),
        "blocked_cv_figs_dir": str(figs_dir),
        "blocked_cv_csv_dir": str(csv_dir),
        "blocked_cv_csv_folds_dir": str(csv_subdirs["folds"]),
        "blocked_cv_csv_predictions_dir": str(csv_subdirs["predictions"]),
        "blocked_cv_csv_metrics_dir": str(csv_subdirs["metrics"]),
        "blocked_cv_csv_diagnostics_dir": str(csv_subdirs["diagnostics"]),
        "n_biopsies_seen": int(
            all_voxel_wise_dose_df[["Patient ID", "Bx index"]].drop_duplicates().shape[0]
        ) if {"Patient ID", "Bx index"}.issubset(all_voxel_wise_dose_df.columns) else 0,
        "semivariogram_rows_seen": int(len(semivariogram_df)),
        "block_mode": config.block_mode,
        "n_folds": int(config.n_folds),
        "min_derived_block_mm": float(config.min_derived_block_mm),
        "merge_tiny_tail_folds": bool(config.merge_tiny_tail_folds),
        "rebalance_two_fold_splits": bool(config.rebalance_two_fold_splits),
        "min_test_voxels": int(config.min_test_voxels),
        "min_test_block_mm": float(config.min_test_block_mm),
        "min_effective_folds_after_merge": int(config.min_effective_folds_after_merge),
        "n_biopsies_with_merged_tail": merged_bx_count,
        "n_biopsies_with_rebalanced_two_fold_split": rebalanced_two_fold_bx_count,
        "position_mode": config.position_mode,
        "target_stat": config.target_stat,
        "mean_mode": config.mean_mode,
        "primary_predictive_variance_mode": config.primary_predictive_variance_mode,
        "compare_variance_modes": bool(config.compare_variance_modes),
        "variance_modes_to_compare": list(config.variance_modes_to_compare) if config.variance_modes_to_compare is not None else None,
        "fold_map_csv": str(fold_map_path),
        "fold_summary_csv": str(fold_summary_path),
        "n_fold_map_rows": int(len(fold_map_df)),
        "n_fold_summary_rows": int(len(fold_summary_df)),
    }
    return status


def _default_kernel_specs() -> list[Tuple[str, float | None, str]]:
    return [
        ("matern", 1.5, "matern_nu_1_5"),
        ("matern", 2.5, "matern_nu_2_5"),
        ("rbf", None, "rbf"),
        ("exp", None, "exp"),
    ]


def _predictive_variance_for_mode(
    *,
    latent_sd: np.ndarray,
    obs_var: np.ndarray,
    nugget: float,
    mode: str,
) -> np.ndarray:
    """Resolve predictive variance used for held-out standardization/NLPD."""
    latent_var = np.maximum(np.asarray(latent_sd, dtype=float) ** 2, 0.0)
    obs_var = np.maximum(np.asarray(obs_var, dtype=float), 0.0)
    if mode == "latent":
        return latent_var
    if mode == "observed_mc":
        return latent_var + obs_var
    if mode == "observed_mc_plus_nugget":
        return latent_var + obs_var + max(float(nugget), 0.0)
    raise ValueError(
        f"Unsupported blocked_CV predictive_variance_mode '{mode}'. "
        "Use 'latent', 'observed_mc', or 'observed_mc_plus_nugget'."
    )


def _resolve_variance_modes(config: BlockedCVConfig) -> tuple[str, list[str]]:
    """
    Resolve primary and scored predictive-variance modes for blocked_CV outputs.
    """
    primary_mode = str(config.primary_predictive_variance_mode)
    if config.compare_variance_modes:
        base_modes = (
            list(config.variance_modes_to_compare)
            if config.variance_modes_to_compare is not None
            else ["latent", "observed_mc"]
        )
    else:
        base_modes = [primary_mode]

    scored_modes: list[str] = []
    for m in base_modes:
        m_str = str(m)
        if m_str not in scored_modes:
            scored_modes.append(m_str)
    if primary_mode not in scored_modes:
        scored_modes.insert(0, primary_mode)

    valid = {"latent", "observed_mc", "observed_mc_plus_nugget"}
    invalid = [m for m in scored_modes if m not in valid]
    if invalid:
        raise ValueError(
            f"Unsupported blocked_CV variance mode(s) {invalid}. "
            "Use only: 'latent', 'observed_mc', 'observed_mc_plus_nugget'."
        )
    return primary_mode, scored_modes


def _fit_hyperparams_from_train_sv(
    sv_train: pd.DataFrame,
    *,
    patient_id,
    bx_index,
    kernel_name: str,
    kernel_param: float | None,
) -> gpr_pf.GPHyperparams:
    """Fit kernel hyperparameters from a train-only semivariogram table."""
    if kernel_name == "matern":
        nu_use = float(kernel_param) if kernel_param is not None else 1.5
        return gpr_pf.fit_variogram_matern(sv_train, patient_id, bx_index, nu=nu_use)
    if kernel_name == "rbf":
        return gpr_pf.fit_variogram_rbf(sv_train, patient_id, bx_index)
    if kernel_name == "exp":
        return gpr_pf.fit_variogram_exponential(sv_train, patient_id, bx_index)
    raise ValueError(f"Unsupported kernel_name '{kernel_name}' in blocked_CV.")


def _compute_fold_metrics_from_predictions(
    pred_df: pd.DataFrame,
    *,
    variance_mode_col: str,
) -> pd.DataFrame:
    """
    Aggregate held-out point predictions into one row per fold group.

    The output preserves grouping identity keys and selected run metadata while
    computing residual/standardized-residual/NLPD metrics from point-level rows.
    """
    if pred_df.empty:
        return pd.DataFrame()
    if variance_mode_col not in pred_df.columns:
        raise ValueError(f"Missing required variance-mode column '{variance_mode_col}' in predictions table.")

    group_cols = [
        "Patient ID",
        "Bx ID",
        "Bx index",
        "fold_id",
        "kernel_label",
        "kernel_name",
        "kernel_param",
        variance_mode_col,
    ]
    group_cols = [c for c in group_cols if c in pred_df.columns]

    meta_cols = [
        "gp_mean_mode",
        "target_stat",
        "primary_predictive_variance_mode",
        "block_mode",
        "position_mode",
        "split_strategy",
        "rebalanced_two_fold_split",
        "cv_n_folds",
        "cv_block_length_mm",
        "cv_min_derived_block_mm",
        "cv_merge_tiny_tail_folds",
        "cv_rebalance_two_fold_splits",
        "cv_min_test_voxels",
        "cv_min_test_block_mm",
        "cv_min_effective_folds_after_merge",
        "n_train_voxels",
        "n_test_voxels",
        "n_total_voxels_bx",
        "train_test_total_match",
        "train_count_matches_fold_map",
        "test_count_matches_fold_map",
    ]
    meta_cols = [c for c in meta_cols if c in pred_df.columns]

    rows = []
    for keys, g in pred_df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        if variance_mode_col in row and variance_mode_col != "variance_mode":
            row["variance_mode"] = row[variance_mode_col]
        for col in meta_cols:
            row[col] = g[col].iloc[0]

        residual = pd.to_numeric(g.get("residual", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        rstd = pd.to_numeric(g.get("rstd", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        var_pred_used = pd.to_numeric(g.get("var_pred_used", pd.Series(dtype=float)), errors="coerce").to_numpy(float)

        resid_v = residual[np.isfinite(residual)]
        rstd_v = rstd[np.isfinite(rstd)]
        valid_nlpd = np.isfinite(residual) & np.isfinite(var_pred_used)
        eps = 1e-12
        if np.any(valid_nlpd):
            resid_nlpd = residual[valid_nlpd]
            var_nlpd = np.maximum(var_pred_used[valid_nlpd], eps)
            nlpd = 0.5 * np.log(2.0 * np.pi * var_nlpd) + 0.5 * ((resid_nlpd ** 2) / var_nlpd)
            nlpd_mean = float(np.mean(nlpd))
        else:
            nlpd_mean = np.nan

        n_test = int(len(g))
        n_residual_valid = int(resid_v.size)
        n_rstd_valid = int(rstd_v.size)
        n_nlpd_valid = int(np.sum(valid_nlpd))
        n_abs_le1 = int(np.sum(np.abs(rstd_v) <= 1.0)) if n_rstd_valid else 0
        n_abs_le2 = int(np.sum(np.abs(rstd_v) <= 2.0)) if n_rstd_valid else 0
        n_abs_ge3 = int(np.sum(np.abs(rstd_v) >= 3.0)) if n_rstd_valid else 0

        row.update(
            {
                "n_test_points": n_test,
                "n_residual_valid": n_residual_valid,
                "n_rstd_valid": n_rstd_valid,
                "n_nlpd_valid": n_nlpd_valid,
                "mean_residual": float(np.mean(resid_v)) if n_residual_valid else np.nan,
                "rmse": float(np.sqrt(np.mean(resid_v ** 2))) if n_residual_valid else np.nan,
                "mae": float(np.mean(np.abs(resid_v))) if n_residual_valid else np.nan,
                "mean_rstd": float(np.mean(rstd_v)) if n_rstd_valid else np.nan,
                "sd_rstd": float(np.std(rstd_v, ddof=0)) if n_rstd_valid else np.nan,
                "n_abs_le1": n_abs_le1,
                "n_abs_le2": n_abs_le2,
                "n_abs_ge3": n_abs_ge3,
                "pct_abs_le1": float(100.0 * n_abs_le1 / n_rstd_valid) if n_rstd_valid else np.nan,
                "pct_abs_le2": float(100.0 * n_abs_le2 / n_rstd_valid) if n_rstd_valid else np.nan,
                "pct_abs_ge3": float(100.0 * n_abs_ge3 / n_rstd_valid) if n_rstd_valid else np.nan,
                "nlpd_mean": nlpd_mean,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _compute_biopsy_metrics_from_predictions(
    pred_df: pd.DataFrame,
    *,
    variance_mode_col: str,
) -> pd.DataFrame:
    """
    Aggregate held-out point predictions into one row per biopsy group.

    Biopsy metrics are computed from concatenated held-out points across all folds
    for a given biopsy/kernel/variance-mode group.
    """
    if pred_df.empty:
        return pd.DataFrame()
    if variance_mode_col not in pred_df.columns:
        raise ValueError(f"Missing required variance-mode column '{variance_mode_col}' in predictions table.")

    group_cols = [
        "Patient ID",
        "Bx ID",
        "Bx index",
        "kernel_label",
        "kernel_name",
        "kernel_param",
        variance_mode_col,
    ]
    group_cols = [c for c in group_cols if c in pred_df.columns]

    meta_cols = [
        "gp_mean_mode",
        "target_stat",
        "primary_predictive_variance_mode",
        "block_mode",
        "position_mode",
        "cv_n_folds",
        "cv_block_length_mm",
        "cv_min_derived_block_mm",
        "cv_merge_tiny_tail_folds",
        "cv_rebalance_two_fold_splits",
        "cv_min_test_voxels",
        "cv_min_test_block_mm",
        "cv_min_effective_folds_after_merge",
    ]
    meta_cols = [c for c in meta_cols if c in pred_df.columns]

    rows = []
    for keys, g in pred_df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        if variance_mode_col in row and variance_mode_col != "variance_mode":
            row["variance_mode"] = row[variance_mode_col]
        for col in meta_cols:
            row[col] = g[col].iloc[0]

        residual = pd.to_numeric(g.get("residual", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        rstd = pd.to_numeric(g.get("rstd", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        var_pred_used = pd.to_numeric(g.get("var_pred_used", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        resid_v = residual[np.isfinite(residual)]
        rstd_v = rstd[np.isfinite(rstd)]

        valid_nlpd = np.isfinite(residual) & np.isfinite(var_pred_used)
        eps = 1e-12
        if np.any(valid_nlpd):
            resid_nlpd = residual[valid_nlpd]
            var_nlpd = np.maximum(var_pred_used[valid_nlpd], eps)
            nlpd = 0.5 * np.log(2.0 * np.pi * var_nlpd) + 0.5 * ((resid_nlpd ** 2) / var_nlpd)
            nlpd_mean = float(np.mean(nlpd))
        else:
            nlpd_mean = np.nan

        n_points = int(len(g))
        n_residual_valid = int(resid_v.size)
        n_rstd_valid = int(rstd_v.size)
        n_nlpd_valid = int(np.sum(valid_nlpd))
        n_abs_le1 = int(np.sum(np.abs(rstd_v) <= 1.0)) if n_rstd_valid else 0
        n_abs_le2 = int(np.sum(np.abs(rstd_v) <= 2.0)) if n_rstd_valid else 0
        n_abs_ge3 = int(np.sum(np.abs(rstd_v) >= 3.0)) if n_rstd_valid else 0
        n_folds_contributing = int(
            g[["Patient ID", "Bx index", "fold_id"]].drop_duplicates().shape[0]
        ) if {"Patient ID", "Bx index", "fold_id"}.issubset(g.columns) else np.nan

        row.update(
            {
                "n_points": n_points,
                "n_folds_contributing": n_folds_contributing,
                "n_residual_valid": n_residual_valid,
                "n_rstd_valid": n_rstd_valid,
                "n_nlpd_valid": n_nlpd_valid,
                "mean_residual": float(np.mean(resid_v)) if n_residual_valid else np.nan,
                "rmse": float(np.sqrt(np.mean(resid_v ** 2))) if n_residual_valid else np.nan,
                "mae": float(np.mean(np.abs(resid_v))) if n_residual_valid else np.nan,
                "mean_rstd": float(np.mean(rstd_v)) if n_rstd_valid else np.nan,
                "sd_rstd": float(np.std(rstd_v, ddof=0)) if n_rstd_valid else np.nan,
                "n_abs_le1": n_abs_le1,
                "n_abs_le2": n_abs_le2,
                "n_abs_ge3": n_abs_ge3,
                "pct_abs_le1": float(100.0 * n_abs_le1 / n_rstd_valid) if n_rstd_valid else np.nan,
                "pct_abs_le2": float(100.0 * n_abs_le2 / n_rstd_valid) if n_rstd_valid else np.nan,
                "pct_abs_ge3": float(100.0 * n_abs_ge3 / n_rstd_valid) if n_rstd_valid else np.nan,
                "nlpd_mean": nlpd_mean,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _compute_cohort_summary_from_biopsy_metrics(biopsy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize biopsy-level metrics across biopsies per kernel/variance-mode group.
    """
    if biopsy_df.empty:
        return pd.DataFrame()

    group_cols = [
        "kernel_label",
        "kernel_name",
        "kernel_param",
        "variance_mode",
        "gp_mean_mode",
        "target_stat",
        "primary_predictive_variance_mode",
        "block_mode",
        "position_mode",
        "cv_n_folds",
        "cv_block_length_mm",
        "cv_min_derived_block_mm",
        "cv_merge_tiny_tail_folds",
        "cv_rebalance_two_fold_splits",
        "cv_min_test_voxels",
        "cv_min_test_block_mm",
        "cv_min_effective_folds_after_merge",
    ]
    group_cols = [c for c in group_cols if c in biopsy_df.columns]

    metric_cols = [
        "n_points",
        "n_folds_contributing",
        "n_residual_valid",
        "n_rstd_valid",
        "n_nlpd_valid",
        "mean_residual",
        "rmse",
        "mae",
        "mean_rstd",
        "sd_rstd",
        "n_abs_le1",
        "n_abs_le2",
        "n_abs_ge3",
        "pct_abs_le1",
        "pct_abs_le2",
        "pct_abs_ge3",
        "nlpd_mean",
    ]
    metric_cols = [c for c in metric_cols if c in biopsy_df.columns]

    rows = []
    for keys, g in biopsy_df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_biopsies"] = int(len(g))

        for col in metric_cols:
            arr = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_q05"] = np.nan
                row[f"{col}_q25"] = np.nan
                row[f"{col}_q75"] = np.nan
                row[f"{col}_q95"] = np.nan
                row[f"{col}_iqr"] = np.nan
                continue
            q05 = float(np.quantile(arr, 0.05))
            q25 = float(np.quantile(arr, 0.25))
            q75 = float(np.quantile(arr, 0.75))
            q95 = float(np.quantile(arr, 0.95))
            row[f"{col}_mean"] = float(np.mean(arr))
            row[f"{col}_median"] = float(np.median(arr))
            row[f"{col}_q05"] = q05
            row[f"{col}_q25"] = q25
            row[f"{col}_q75"] = q75
            row[f"{col}_q95"] = q95
            row[f"{col}_iqr"] = float(q75 - q25)

        rows.append(row)

    return pd.DataFrame(rows)


def run_blocked_cv_phase3c_smoke(
    all_voxel_wise_dose_df: pd.DataFrame,
    semivariogram_df: pd.DataFrame,
    *,
    output_dir: Path,
    figs_dir: Path,
    csv_dir: Path,
    config: BlockedCVConfig,
) -> dict:
    """
    Phase 3C/3D blocked_CV path:
    - uses fold map from 3B logic,
    - runs strict train-only fit/predict for configured kernel specs,
    - writes centralized *_all CSV outputs (plus optional per-kernel slices).
    """
    del semivariogram_df  # phase 3C uses per-fold train-only semivariograms
    csv_subdirs = _blocked_cv_csv_subdirs(csv_dir)
    fold_map_df, fold_summary_df = build_blocked_cv_fold_map(
        all_voxel_wise_dose_df,
        config=config,
    )
    fold_map_path = csv_subdirs["folds"].joinpath("blocked_cv_fold_map.csv")
    fold_summary_path = csv_subdirs["folds"].joinpath("blocked_cv_fold_summary.csv")
    if not fold_map_path.exists():
        fold_map_df.to_csv(fold_map_path, index=False)
    if not fold_summary_path.exists():
        fold_summary_df.to_csv(fold_summary_path, index=False)

    kernel_specs_raw = list(config.kernel_specs) if config.kernel_specs is not None else _default_kernel_specs()
    if not kernel_specs_raw:
        raise ValueError("blocked_CV Phase 3C needs at least one kernel spec.")
    kernel_specs: list[tuple[str, float | None, str]] = []
    for spec in kernel_specs_raw:
        if len(spec) == 3:
            k_name, k_param, k_label = spec
        elif len(spec) == 2:
            k_name, k_param = spec
            k_label = f"{k_name}_{k_param}" if k_param is not None else str(k_name)
        else:
            raise ValueError(
                "Each blocked_CV kernel spec must be (kernel_name, kernel_param, kernel_label) "
                "or (kernel_name, kernel_param)."
            )
        kernel_specs.append((str(k_name), None if k_param is None else float(k_param), str(k_label)))
    primary_mode, scored_modes = _resolve_variance_modes(config)

    pred_rows = []
    pred_compare_rows = []
    fold_status_rows = []

    grp_cols = ["Patient ID", "Bx index", "fold_id"]
    for kernel_name, kernel_param, kernel_label in kernel_specs:
        for (patient_id, bx_index, fold_id), fold_rows in fold_map_df.groupby(grp_cols):
            g_bx = all_voxel_wise_dose_df[
                (all_voxel_wise_dose_df["Patient ID"] == patient_id)
                & (all_voxel_wise_dose_df["Bx index"] == bx_index)
            ].copy()
            bx_id_value = g_bx["Bx ID"].dropna().iloc[0] if ("Bx ID" in g_bx.columns and g_bx["Bx ID"].notna().any()) else None
            cv_block_length_mm_value = float(config.block_length_mm) if config.block_length_mm is not None else np.nan
            fold_split_strategy = (
                str(fold_rows["split_strategy"].iloc[0])
                if ("split_strategy" in fold_rows.columns and not fold_rows.empty)
                else ""
            )
            fold_rebalanced_two_fold_split = (
                bool(fold_rows["rebalanced_two_fold_split"].iloc[0])
                if ("rebalanced_two_fold_split" in fold_rows.columns and not fold_rows.empty)
                else False
            )
            n_total_voxels_bx = int(g_bx["Voxel index"].nunique()) if "Voxel index" in g_bx.columns else None
            n_train_expected = int(fold_rows["n_train"].iloc[0]) if ("n_train" in fold_rows.columns and not fold_rows.empty) else None
            n_test_expected = int(fold_rows["n_test"].iloc[0]) if ("n_test" in fold_rows.columns and not fold_rows.empty) else None
            n_train_plus_test_expected = (
                int(n_train_expected + n_test_expected)
                if n_train_expected is not None and n_test_expected is not None
                else None
            )
            expected_train_test_total_match = (
                bool(n_train_plus_test_expected == n_total_voxels_bx)
                if n_train_plus_test_expected is not None and n_total_voxels_bx is not None
                else None
            )
            test_voxels = set(
                fold_rows.loc[fold_rows["is_test"], "Voxel index"].astype(int).tolist()
            )
            if not test_voxels:
                fold_status_rows.append(
                    {
                        "Patient ID": patient_id,
                        "Bx ID": bx_id_value,
                        "Bx index": bx_index,
                        "fold_id": int(fold_id),
                        "kernel_label": kernel_label,
                        "kernel_name": kernel_name,
                        "kernel_param": kernel_param,
                        "primary_predictive_variance_mode": primary_mode,
                        "variance_modes_scored": "|".join(scored_modes),
                        "block_mode": config.block_mode,
                        "position_mode": config.position_mode,
                        "cv_n_folds": int(config.n_folds),
                        "cv_block_length_mm": cv_block_length_mm_value,
                        "cv_min_derived_block_mm": float(config.min_derived_block_mm),
                        "cv_merge_tiny_tail_folds": bool(config.merge_tiny_tail_folds),
                        "cv_rebalance_two_fold_splits": bool(config.rebalance_two_fold_splits),
                        "cv_min_test_voxels": int(config.min_test_voxels),
                        "cv_min_test_block_mm": float(config.min_test_block_mm),
                        "cv_min_effective_folds_after_merge": int(config.min_effective_folds_after_merge),
                        "split_strategy": fold_split_strategy,
                        "rebalanced_two_fold_split": fold_rebalanced_two_fold_split,
                        "status": "skipped",
                        "message": "no test voxels",
                        "n_total_voxels_bx": n_total_voxels_bx,
                        "n_train_voxels_expected_from_fold_map": n_train_expected,
                        "n_test_voxels_expected_from_fold_map": n_test_expected,
                        "n_train_plus_test_expected": n_train_plus_test_expected,
                        "expected_train_test_total_match": expected_train_test_total_match,
                    }
                )
                continue
            train_df = g_bx[~g_bx["Voxel index"].isin(test_voxels)].copy()
            test_df = g_bx[g_bx["Voxel index"].isin(test_voxels)].copy()
            n_train_actual = None
            n_test_actual = None
            n_train_plus_test_actual = None
            train_test_total_match = None
            train_count_matches_fold_map = None
            test_count_matches_fold_map = None

            try:
                X_train, y_train, var_n_train, _pv_train = gpr_pf.build_voxel_targets_and_noise(
                    train_df,
                    patient_id=patient_id,
                    bx_index=bx_index,
                    target_stat=config.target_stat,
                    position_mode=config.position_mode,
                )
                X_test, y_test, var_n_test, pv_test = gpr_pf.build_voxel_targets_and_noise(
                    test_df,
                    patient_id=patient_id,
                    bx_index=bx_index,
                    target_stat=config.target_stat,
                    position_mode=config.position_mode,
                )
                n_train_actual = int(len(X_train))
                n_test_actual = int(len(X_test))
                n_train_plus_test_actual = int(n_train_actual + n_test_actual)
                train_test_total_match = (
                    bool(n_train_plus_test_actual == n_total_voxels_bx)
                    if n_total_voxels_bx is not None
                    else None
                )
                train_count_matches_fold_map = (
                    bool(n_train_actual == n_train_expected)
                    if n_train_expected is not None
                    else None
                )
                test_count_matches_fold_map = (
                    bool(n_test_actual == n_test_expected)
                    if n_test_expected is not None
                    else None
                )
                if n_train_actual < 3 or n_test_actual < 1:
                    fold_status_rows.append(
                        {
                            "Patient ID": patient_id,
                            "Bx ID": bx_id_value,
                            "Bx index": bx_index,
                            "fold_id": int(fold_id),
                            "kernel_label": kernel_label,
                            "kernel_name": kernel_name,
                            "kernel_param": kernel_param,
                            "primary_predictive_variance_mode": primary_mode,
                            "variance_modes_scored": "|".join(scored_modes),
                            "block_mode": config.block_mode,
                            "position_mode": config.position_mode,
                            "cv_n_folds": int(config.n_folds),
                            "cv_block_length_mm": cv_block_length_mm_value,
                            "cv_min_derived_block_mm": float(config.min_derived_block_mm),
                            "cv_merge_tiny_tail_folds": bool(config.merge_tiny_tail_folds),
                            "cv_rebalance_two_fold_splits": bool(config.rebalance_two_fold_splits),
                            "cv_min_test_voxels": int(config.min_test_voxels),
                            "cv_min_test_block_mm": float(config.min_test_block_mm),
                            "cv_min_effective_folds_after_merge": int(config.min_effective_folds_after_merge),
                            "split_strategy": fold_split_strategy,
                            "rebalanced_two_fold_split": fold_rebalanced_two_fold_split,
                            "status": "skipped",
                            "message": f"insufficient voxels (train={n_train_actual}, test={n_test_actual})",
                            "n_total_voxels_bx": n_total_voxels_bx,
                            "n_train_voxels_expected_from_fold_map": n_train_expected,
                            "n_test_voxels_expected_from_fold_map": n_test_expected,
                            "n_train_plus_test_expected": n_train_plus_test_expected,
                            "expected_train_test_total_match": expected_train_test_total_match,
                            "n_train_voxels": n_train_actual,
                            "n_test_voxels": n_test_actual,
                            "n_train_plus_test_voxels": n_train_plus_test_actual,
                            "train_test_total_match": train_test_total_match,
                            "train_count_matches_fold_map": train_count_matches_fold_map,
                            "test_count_matches_fold_map": test_count_matches_fold_map,
                        }
                    )
                    continue

                sv_train = gpr_sv.compute_semivariogram_pairwise(
                    train_df,
                    voxel_size_mm=float(config.semivariogram_voxel_size_mm),
                    max_lag_voxels=None,
                    position_mode=config.position_mode,
                    lag_bin_width_mm=config.semivariogram_lag_bin_width_mm,
                )
                sv_train["Patient ID"] = patient_id
                sv_train["Bx index"] = bx_index
                sv_train = sv_train[np.isfinite(sv_train["semivariance"])].copy()
                if len(sv_train) < 3:
                    fold_status_rows.append(
                        {
                            "Patient ID": patient_id,
                            "Bx ID": bx_id_value,
                            "Bx index": bx_index,
                            "fold_id": int(fold_id),
                            "kernel_label": kernel_label,
                            "kernel_name": kernel_name,
                            "kernel_param": kernel_param,
                            "primary_predictive_variance_mode": primary_mode,
                            "variance_modes_scored": "|".join(scored_modes),
                            "block_mode": config.block_mode,
                            "position_mode": config.position_mode,
                            "cv_n_folds": int(config.n_folds),
                            "cv_block_length_mm": cv_block_length_mm_value,
                            "cv_min_derived_block_mm": float(config.min_derived_block_mm),
                            "cv_merge_tiny_tail_folds": bool(config.merge_tiny_tail_folds),
                            "cv_rebalance_two_fold_splits": bool(config.rebalance_two_fold_splits),
                            "cv_min_test_voxels": int(config.min_test_voxels),
                            "cv_min_test_block_mm": float(config.min_test_block_mm),
                            "cv_min_effective_folds_after_merge": int(config.min_effective_folds_after_merge),
                            "split_strategy": fold_split_strategy,
                            "rebalanced_two_fold_split": fold_rebalanced_two_fold_split,
                            "status": "skipped",
                            "message": f"insufficient semivariogram bins (n={len(sv_train)})",
                            "n_total_voxels_bx": n_total_voxels_bx,
                            "n_train_voxels_expected_from_fold_map": n_train_expected,
                            "n_test_voxels_expected_from_fold_map": n_test_expected,
                            "n_train_plus_test_expected": n_train_plus_test_expected,
                            "expected_train_test_total_match": expected_train_test_total_match,
                            "n_train_voxels": n_train_actual,
                            "n_test_voxels": n_test_actual,
                            "n_train_plus_test_voxels": n_train_plus_test_actual,
                            "train_test_total_match": train_test_total_match,
                            "train_count_matches_fold_map": train_count_matches_fold_map,
                            "test_count_matches_fold_map": test_count_matches_fold_map,
                        }
                    )
                    continue

                hyp = _fit_hyperparams_from_train_sv(
                    sv_train,
                    patient_id=patient_id,
                    bx_index=bx_index,
                    kernel_name=kernel_name,
                    kernel_param=kernel_param,
                )
                mu_test, sd_test = gpr_pf.gp_posterior(
                    X_train,
                    y_train,
                    var_n_train,
                    hyp,
                    X_star=X_test,
                    mean_mode=config.mean_mode,
                )
                resid = y_test - mu_test
                mode_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                for mode in scored_modes:
                    pred_var_mode = _predictive_variance_for_mode(
                        latent_sd=sd_test,
                        obs_var=var_n_test,
                        nugget=float(getattr(hyp, "nugget", 0.0)),
                        mode=mode,
                    )
                    pred_sd_mode = np.sqrt(np.maximum(pred_var_mode, 1e-12))
                    rstd_mode = resid / pred_sd_mode
                    mode_arrays[mode] = (pred_var_mode, pred_sd_mode, rstd_mode)
                pred_var_primary, pred_sd_primary, rstd_primary = mode_arrays[primary_mode]

                pv_test = pv_test.sort_values("x_mm").reset_index(drop=True)
                for i in range(len(X_test)):
                    abs_res = float(np.abs(resid[i]))
                    abs_res_over_sd_latent = abs_res / max(float(sd_test[i]), 1e-12)
                    abs_res_over_sd_used = abs_res / max(float(pred_sd_primary[i]), 1e-12)
                    pred_rows.append(
                        {
                            "Patient ID": patient_id,
                            "Bx ID": bx_id_value,
                            "Bx index": bx_index,
                            "fold_id": int(fold_id),
                            "kernel_label": kernel_label,
                            "kernel_name": kernel_name,
                            "kernel_param": kernel_param,
                            "Voxel index": int(pv_test.loc[i, "Voxel index"]),
                            "x_mm": float(X_test[i]),
                            "y_test": float(y_test[i]),
                            "mu_test": float(mu_test[i]),
                            "sd_test_latent": float(sd_test[i]),
                            "var_obs_test": float(var_n_test[i]),
                            "var_pred_used": float(pred_var_primary[i]),
                            "sd_pred_used": float(pred_sd_primary[i]),
                            "residual": float(resid[i]),
                            "rstd": float(rstd_primary[i]),
                            "abs_res_over_sd_latent": abs_res_over_sd_latent,
                            "abs_res_over_sd_used": abs_res_over_sd_used,
                            "gp_mean_mode": config.mean_mode,
                            "target_stat": config.target_stat,
                            "block_mode": config.block_mode,
                            "position_mode": config.position_mode,
                            "cv_n_folds": int(config.n_folds),
                            "cv_block_length_mm": cv_block_length_mm_value,
                            "cv_min_derived_block_mm": float(config.min_derived_block_mm),
                            "cv_merge_tiny_tail_folds": bool(config.merge_tiny_tail_folds),
                            "cv_rebalance_two_fold_splits": bool(config.rebalance_two_fold_splits),
                            "cv_min_test_voxels": int(config.min_test_voxels),
                            "cv_min_test_block_mm": float(config.min_test_block_mm),
                            "cv_min_effective_folds_after_merge": int(config.min_effective_folds_after_merge),
                            "primary_predictive_variance_mode": primary_mode,
                            "predictive_variance_mode": primary_mode,
                            "variance_modes_scored": "|".join(scored_modes),
                            "split_strategy": fold_split_strategy,
                            "rebalanced_two_fold_split": fold_rebalanced_two_fold_split,
                            "n_train_voxels": n_train_actual,
                            "n_test_voxels": n_test_actual,
                            "n_total_voxels_bx": n_total_voxels_bx,
                            "n_train_plus_test_voxels": n_train_plus_test_actual,
                            "train_test_total_match": train_test_total_match,
                            "n_train_voxels_expected_from_fold_map": n_train_expected,
                            "n_test_voxels_expected_from_fold_map": n_test_expected,
                            "n_train_plus_test_expected": n_train_plus_test_expected,
                            "expected_train_test_total_match": expected_train_test_total_match,
                            "train_count_matches_fold_map": train_count_matches_fold_map,
                            "test_count_matches_fold_map": test_count_matches_fold_map,
                            "ell": float(getattr(hyp, "ell", np.nan)),
                            "sigma_f2": float(getattr(hyp, "sigma_f2", np.nan)),
                            "nugget": float(getattr(hyp, "nugget", np.nan)),
                            "nu": float(getattr(hyp, "nu", np.nan)),
                        }
                    )
                    if config.compare_variance_modes:
                        for mode in scored_modes:
                            var_m, sd_m, rstd_m = mode_arrays[mode]
                            pred_compare_rows.append(
                                {
                                    "Patient ID": patient_id,
                                    "Bx ID": bx_id_value,
                                    "Bx index": bx_index,
                                    "fold_id": int(fold_id),
                                    "kernel_label": kernel_label,
                                    "kernel_name": kernel_name,
                                    "kernel_param": kernel_param,
                                    "Voxel index": int(pv_test.loc[i, "Voxel index"]),
                                    "x_mm": float(X_test[i]),
                                    "y_test": float(y_test[i]),
                                    "mu_test": float(mu_test[i]),
                                    "sd_test_latent": float(sd_test[i]),
                                    "var_obs_test": float(var_n_test[i]),
                                    "variance_mode": mode,
                                    "var_pred_used": float(var_m[i]),
                                    "sd_pred_used": float(sd_m[i]),
                                    "residual": float(resid[i]),
                                    "rstd": float(rstd_m[i]),
                                    "abs_res_over_sd_latent": abs_res_over_sd_latent,
                                    "abs_res_over_sd_used": abs_res / max(float(sd_m[i]), 1e-12),
                                    "gp_mean_mode": config.mean_mode,
                                    "target_stat": config.target_stat,
                                    "block_mode": config.block_mode,
                                    "position_mode": config.position_mode,
                                    "cv_n_folds": int(config.n_folds),
                                    "cv_block_length_mm": cv_block_length_mm_value,
                                    "cv_min_derived_block_mm": float(config.min_derived_block_mm),
                                    "cv_merge_tiny_tail_folds": bool(config.merge_tiny_tail_folds),
                                    "cv_rebalance_two_fold_splits": bool(config.rebalance_two_fold_splits),
                                    "cv_min_test_voxels": int(config.min_test_voxels),
                                    "cv_min_test_block_mm": float(config.min_test_block_mm),
                                    "cv_min_effective_folds_after_merge": int(config.min_effective_folds_after_merge),
                                    "primary_predictive_variance_mode": primary_mode,
                                    "split_strategy": fold_split_strategy,
                                    "rebalanced_two_fold_split": fold_rebalanced_two_fold_split,
                                    "n_train_voxels": n_train_actual,
                                    "n_test_voxels": n_test_actual,
                                    "n_total_voxels_bx": n_total_voxels_bx,
                                    "n_train_plus_test_voxels": n_train_plus_test_actual,
                                    "train_test_total_match": train_test_total_match,
                                    "n_train_voxels_expected_from_fold_map": n_train_expected,
                                    "n_test_voxels_expected_from_fold_map": n_test_expected,
                                    "n_train_plus_test_expected": n_train_plus_test_expected,
                                    "expected_train_test_total_match": expected_train_test_total_match,
                                    "train_count_matches_fold_map": train_count_matches_fold_map,
                                    "test_count_matches_fold_map": test_count_matches_fold_map,
                                    "ell": float(getattr(hyp, "ell", np.nan)),
                                    "sigma_f2": float(getattr(hyp, "sigma_f2", np.nan)),
                                    "nugget": float(getattr(hyp, "nugget", np.nan)),
                                    "nu": float(getattr(hyp, "nu", np.nan)),
                                }
                            )

                fold_status_rows.append(
                    {
                        "Patient ID": patient_id,
                        "Bx ID": bx_id_value,
                        "Bx index": bx_index,
                        "fold_id": int(fold_id),
                        "kernel_label": kernel_label,
                        "kernel_name": kernel_name,
                        "kernel_param": kernel_param,
                        "primary_predictive_variance_mode": primary_mode,
                        "variance_modes_scored": "|".join(scored_modes),
                        "block_mode": config.block_mode,
                        "position_mode": config.position_mode,
                        "cv_n_folds": int(config.n_folds),
                        "cv_block_length_mm": cv_block_length_mm_value,
                        "cv_min_derived_block_mm": float(config.min_derived_block_mm),
                        "cv_merge_tiny_tail_folds": bool(config.merge_tiny_tail_folds),
                        "cv_rebalance_two_fold_splits": bool(config.rebalance_two_fold_splits),
                        "cv_min_test_voxels": int(config.min_test_voxels),
                        "cv_min_test_block_mm": float(config.min_test_block_mm),
                        "cv_min_effective_folds_after_merge": int(config.min_effective_folds_after_merge),
                        "split_strategy": fold_split_strategy,
                        "rebalanced_two_fold_split": fold_rebalanced_two_fold_split,
                        "status": "ok",
                        "message": "",
                        "n_total_voxels_bx": n_total_voxels_bx,
                        "n_train_voxels_expected_from_fold_map": n_train_expected,
                        "n_test_voxels_expected_from_fold_map": n_test_expected,
                        "n_train_plus_test_expected": n_train_plus_test_expected,
                        "expected_train_test_total_match": expected_train_test_total_match,
                        "n_train_voxels": n_train_actual,
                        "n_test_voxels": n_test_actual,
                        "n_train_plus_test_voxels": n_train_plus_test_actual,
                        "train_test_total_match": train_test_total_match,
                        "train_count_matches_fold_map": train_count_matches_fold_map,
                        "test_count_matches_fold_map": test_count_matches_fold_map,
                    }
                )

            except Exception as e:  # keep run resilient and auditable
                fold_status_rows.append(
                    {
                        "Patient ID": patient_id,
                        "Bx ID": bx_id_value,
                        "Bx index": bx_index,
                        "fold_id": int(fold_id),
                        "kernel_label": kernel_label,
                        "kernel_name": kernel_name,
                        "kernel_param": kernel_param,
                        "primary_predictive_variance_mode": primary_mode,
                        "variance_modes_scored": "|".join(scored_modes),
                        "block_mode": config.block_mode,
                        "position_mode": config.position_mode,
                        "cv_n_folds": int(config.n_folds),
                        "cv_block_length_mm": cv_block_length_mm_value,
                        "cv_min_derived_block_mm": float(config.min_derived_block_mm),
                        "cv_merge_tiny_tail_folds": bool(config.merge_tiny_tail_folds),
                        "cv_rebalance_two_fold_splits": bool(config.rebalance_two_fold_splits),
                        "cv_min_test_voxels": int(config.min_test_voxels),
                        "cv_min_test_block_mm": float(config.min_test_block_mm),
                        "cv_min_effective_folds_after_merge": int(config.min_effective_folds_after_merge),
                        "split_strategy": fold_split_strategy,
                        "rebalanced_two_fold_split": fold_rebalanced_two_fold_split,
                        "status": "error",
                        "message": str(e),
                        "n_total_voxels_bx": n_total_voxels_bx,
                        "n_train_voxels_expected_from_fold_map": n_train_expected,
                        "n_test_voxels_expected_from_fold_map": n_test_expected,
                        "n_train_plus_test_expected": n_train_plus_test_expected,
                        "expected_train_test_total_match": expected_train_test_total_match,
                        "n_train_voxels": n_train_actual,
                        "n_test_voxels": n_test_actual,
                        "n_train_plus_test_voxels": n_train_plus_test_actual,
                        "train_test_total_match": train_test_total_match,
                        "train_count_matches_fold_map": train_count_matches_fold_map,
                        "test_count_matches_fold_map": test_count_matches_fold_map,
                    }
                )

    pred_df = pd.DataFrame(pred_rows)
    fold_status_df = pd.DataFrame(fold_status_rows)
    if not fold_status_df.empty:
        fold_status_cols = [
            "Patient ID",
            "Bx ID",
            "Bx index",
            "fold_id",
            "kernel_label",
            "kernel_name",
            "kernel_param",
            "primary_predictive_variance_mode",
            "variance_modes_scored",
            "block_mode",
            "position_mode",
            "cv_n_folds",
            "cv_block_length_mm",
            "cv_min_derived_block_mm",
            "cv_merge_tiny_tail_folds",
            "cv_rebalance_two_fold_splits",
            "cv_min_test_voxels",
            "cv_min_test_block_mm",
            "cv_min_effective_folds_after_merge",
            "split_strategy",
            "rebalanced_two_fold_split",
            "status",
            "message",
            "n_train_voxels",
            "n_test_voxels",
            "n_total_voxels_bx",
            "n_train_plus_test_voxels",
            "train_test_total_match",
            "n_train_voxels_expected_from_fold_map",
            "n_test_voxels_expected_from_fold_map",
            "n_train_plus_test_expected",
            "expected_train_test_total_match",
            "train_count_matches_fold_map",
            "test_count_matches_fold_map",
        ]
        extra_cols = [c for c in fold_status_df.columns if c not in fold_status_cols]
        fold_status_df = fold_status_df.reindex(columns=fold_status_cols + extra_cols)
    pred_path = csv_subdirs["predictions"].joinpath("blocked_cv_point_predictions_all.csv")
    status_path = csv_subdirs["diagnostics"].joinpath("blocked_cv_fold_fit_status_all.csv")
    pred_df.to_csv(pred_path, index=False)
    fold_status_df.to_csv(status_path, index=False)

    compare_path = None
    compare_summary_path = None
    compare_df = pd.DataFrame()
    compare_summary_df = pd.DataFrame()
    if config.compare_variance_modes:
        compare_df = pd.DataFrame(pred_compare_rows)
        compare_path = csv_subdirs["predictions"].joinpath("blocked_cv_point_predictions_variance_compare_all.csv")
        compare_df.to_csv(compare_path, index=False)

        summary_rows = []
        if not compare_df.empty and "variance_mode" in compare_df.columns:
            grp_cols = ["kernel_label", "kernel_name", "kernel_param", "variance_mode"]
            # Keep NaN kernel_param groups (rbf/exp have kernel_param=None).
            for (k_label, k_name, k_param, mode), g_mode in compare_df.groupby(
                grp_cols,
                sort=True,
                dropna=False,
            ):
                rstd = pd.to_numeric(g_mode["rstd"], errors="coerce").to_numpy(float)
                valid = np.isfinite(rstd)
                rstd_v = rstd[valid]
                if rstd_v.size:
                    mean_rstd = float(np.mean(rstd_v))
                    sd_rstd = float(np.std(rstd_v, ddof=0))
                    pct_abs_le1 = float(np.mean(np.abs(rstd_v) <= 1.0) * 100.0)
                    pct_abs_le2 = float(np.mean(np.abs(rstd_v) <= 2.0) * 100.0)
                else:
                    mean_rstd = np.nan
                    sd_rstd = np.nan
                    pct_abs_le1 = np.nan
                    pct_abs_le2 = np.nan
                abs_ratio = pd.to_numeric(
                    g_mode.get("abs_res_over_sd_used", pd.Series(dtype=float)),
                    errors="coerce",
                ).to_numpy(float)
                abs_ratio_v = abs_ratio[np.isfinite(abs_ratio)]
                n_unique_folds = int(
                    g_mode[["Patient ID", "Bx index", "fold_id"]].drop_duplicates().shape[0]
                ) if {"Patient ID", "Bx index", "fold_id"}.issubset(g_mode.columns) else 0
                n_unique_biopsies = int(
                    g_mode[["Patient ID", "Bx index"]].drop_duplicates().shape[0]
                ) if {"Patient ID", "Bx index"}.issubset(g_mode.columns) else 0
                summary_rows.append(
                    {
                        "kernel_label": k_label,
                        "kernel_name": k_name,
                        "kernel_param": k_param,
                        "variance_mode": mode,
                        "n_points": int(rstd_v.size),
                        "n_unique_folds": n_unique_folds,
                        "n_unique_biopsies": n_unique_biopsies,
                        "mean_rstd": mean_rstd,
                        "sd_rstd": sd_rstd,
                        "pct_abs_le1": pct_abs_le1,
                        "pct_abs_le2": pct_abs_le2,
                        "median_abs_res_over_sd_used": float(np.median(abs_ratio_v)) if abs_ratio_v.size else np.nan,
                    }
                )
        compare_summary_df = pd.DataFrame(summary_rows)
        compare_summary_path = csv_subdirs["metrics"].joinpath("blocked_cv_variance_mode_summary_all.csv")
        compare_summary_df.to_csv(compare_summary_path, index=False)

    # Phase 4B: fold-level performance metrics from held-out prediction rows.
    fold_metrics_df = _compute_fold_metrics_from_predictions(
        pred_df,
        variance_mode_col="predictive_variance_mode",
    )
    fold_metrics_path = csv_subdirs["metrics"].joinpath("blocked_cv_fold_metrics_all.csv")
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    compare_fold_metrics_df = pd.DataFrame()
    compare_fold_metrics_path = None
    if config.compare_variance_modes:
        compare_fold_metrics_df = _compute_fold_metrics_from_predictions(
            compare_df,
            variance_mode_col="variance_mode",
        )
        compare_fold_metrics_path = csv_subdirs["metrics"].joinpath(
            "blocked_cv_fold_metrics_variance_compare_all.csv"
        )
        compare_fold_metrics_df.to_csv(compare_fold_metrics_path, index=False)

    # Phase 4C: biopsy-level pooled metrics (concatenate held-out points across folds).
    biopsy_metrics_df = _compute_biopsy_metrics_from_predictions(
        pred_df,
        variance_mode_col="predictive_variance_mode",
    )
    biopsy_metrics_path = csv_subdirs["metrics"].joinpath("blocked_cv_biopsy_metrics_all.csv")
    biopsy_metrics_df.to_csv(biopsy_metrics_path, index=False)

    compare_biopsy_metrics_df = pd.DataFrame()
    compare_biopsy_metrics_path = None
    if config.compare_variance_modes:
        compare_biopsy_metrics_df = _compute_biopsy_metrics_from_predictions(
            compare_df,
            variance_mode_col="variance_mode",
        )
        compare_biopsy_metrics_path = csv_subdirs["metrics"].joinpath(
            "blocked_cv_biopsy_metrics_variance_compare_all.csv"
        )
        compare_biopsy_metrics_df.to_csv(compare_biopsy_metrics_path, index=False)

    # Phase 4D: cohort summaries from biopsy-level metrics.
    cohort_summary_df = _compute_cohort_summary_from_biopsy_metrics(biopsy_metrics_df)
    cohort_summary_path = csv_subdirs["metrics"].joinpath("blocked_cv_cohort_summary_all.csv")
    cohort_summary_df.to_csv(cohort_summary_path, index=False)

    compare_cohort_summary_df = pd.DataFrame()
    compare_cohort_summary_path = None
    if config.compare_variance_modes:
        compare_cohort_summary_df = _compute_cohort_summary_from_biopsy_metrics(compare_biopsy_metrics_df)
        compare_cohort_summary_path = csv_subdirs["metrics"].joinpath(
            "blocked_cv_cohort_summary_variance_compare_all.csv"
        )
        compare_cohort_summary_df.to_csv(compare_cohort_summary_path, index=False)

    # Optional per-kernel slices (subsets of centralized *_all outputs).
    kernel_labels_run = sorted(pd.unique(pred_df["kernel_label"])) if not pred_df.empty else []
    if config.write_per_kernel_predictions_csvs and not pred_df.empty:
        for k_label in kernel_labels_run:
            pred_df.loc[pred_df["kernel_label"] == k_label].to_csv(
                csv_subdirs["predictions"].joinpath(f"blocked_cv_point_predictions_{k_label}.csv"),
                index=False,
            )
    if config.write_per_kernel_fit_status_csvs and not fold_status_df.empty:
        for k_label in sorted(pd.unique(fold_status_df["kernel_label"])):
            fold_status_df.loc[fold_status_df["kernel_label"] == k_label].to_csv(
                csv_subdirs["diagnostics"].joinpath(f"blocked_cv_fold_fit_status_{k_label}.csv"),
                index=False,
            )
    if config.compare_variance_modes and config.write_per_kernel_variance_compare_csvs and not compare_df.empty:
        for k_label in sorted(pd.unique(compare_df["kernel_label"])):
            compare_df.loc[compare_df["kernel_label"] == k_label].to_csv(
                csv_subdirs["predictions"].joinpath(f"blocked_cv_point_predictions_variance_compare_{k_label}.csv"),
                index=False,
            )
    if config.compare_variance_modes and config.write_per_kernel_variance_summary_csvs and compare_summary_path is not None:
        if not compare_summary_df.empty and "kernel_label" in compare_summary_df.columns:
            for k_label in sorted(pd.unique(compare_summary_df["kernel_label"])):
                compare_summary_df.loc[compare_summary_df["kernel_label"] == k_label].to_csv(
                    csv_subdirs["metrics"].joinpath(f"blocked_cv_variance_mode_summary_{k_label}.csv"),
                    index=False,
                )

    status = {
        "phase": "3D_all_kernels_fit_predict",
        "status": "ready",
        "blocked_cv_root": str(output_dir),
        "blocked_cv_figs_dir": str(figs_dir),
        "blocked_cv_csv_dir": str(csv_dir),
        "blocked_cv_csv_folds_dir": str(csv_subdirs["folds"]),
        "blocked_cv_csv_predictions_dir": str(csv_subdirs["predictions"]),
        "blocked_cv_csv_metrics_dir": str(csv_subdirs["metrics"]),
        "blocked_cv_csv_diagnostics_dir": str(csv_subdirs["diagnostics"]),
        "kernels_run": kernel_labels_run,
        "n_kernels_run": int(len(kernel_labels_run)),
        "primary_predictive_variance_mode": primary_mode,
        "variance_modes_scored": scored_modes,
        "point_predictions_csv": str(pred_path),
        "fold_fit_status_csv": str(status_path),
        "fold_metrics_csv": str(fold_metrics_path),
        "biopsy_metrics_csv": str(biopsy_metrics_path),
        "cohort_summary_csv": str(cohort_summary_path),
        "point_predictions_compare_csv": str(compare_path) if compare_path is not None else None,
        "fold_metrics_compare_csv": str(compare_fold_metrics_path) if compare_fold_metrics_path is not None else None,
        "biopsy_metrics_compare_csv": str(compare_biopsy_metrics_path) if compare_biopsy_metrics_path is not None else None,
        "cohort_summary_compare_csv": str(compare_cohort_summary_path) if compare_cohort_summary_path is not None else None,
        "variance_mode_summary_csv": str(compare_summary_path) if compare_summary_path is not None else None,
        "n_point_prediction_rows": int(len(pred_df)),
        "n_fold_metrics_rows": int(len(fold_metrics_df)),
        "n_biopsy_metrics_rows": int(len(biopsy_metrics_df)),
        "n_cohort_summary_rows": int(len(cohort_summary_df)),
        "n_point_prediction_compare_rows": int(len(compare_df)),
        "n_fold_metrics_compare_rows": int(len(compare_fold_metrics_df)),
        "n_biopsy_metrics_compare_rows": int(len(compare_biopsy_metrics_df)),
        "n_cohort_summary_compare_rows": int(len(compare_cohort_summary_df)),
        "n_fold_status_rows": int(len(fold_status_df)),
        "n_fold_status_ok": int((fold_status_df.get("status", pd.Series(dtype=str)) == "ok").sum()) if len(fold_status_df) else 0,
        "n_fold_status_error": int((fold_status_df.get("status", pd.Series(dtype=str)) == "error").sum()) if len(fold_status_df) else 0,
    }
    return status
