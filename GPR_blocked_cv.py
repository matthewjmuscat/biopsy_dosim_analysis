from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import GPR_analysis_pipeline_functions as gpr_pf
import GPR_semivariogram as gpr_sv
import GPR_production_plots as gpr_pp
import matplotlib.pyplot as plt


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
    write_debug_csvs: bool = True
    write_eligible_views: bool = True
    write_per_kernel_predictions_csvs: bool = False
    write_per_kernel_fit_status_csvs: bool = False
    write_per_kernel_variance_compare_csvs: bool = False
    write_per_kernel_variance_summary_csvs: bool = False
    plot_patient_bx_list: Iterable[Tuple[str, int]] | None = None
    plot_grid_ncols: int = 2
    plot_grid_label_map: dict[Tuple[str, int], str] | None = None
    plot_fold_ids: Iterable[int] | None = None
    plot_max_folds_per_biopsy: int | None = None
    plot_fold_sort_mode: str = "fold_id"
    plot_include_merged_tail_folds: bool = True
    plot_include_rebalanced_two_fold_splits: bool = True
    plot_kernel_labels: Iterable[str] | None = None
    plot_variance_mode: str = "primary"
    plot_make_paired_semivariogram_profile: bool = True
    plot_make_semivariogram_grids: bool = True
    plot_make_profile_grids: bool = True
    plot_semivariogram_show_n_pairs_paired: bool | None = None
    plot_semivariogram_show_n_pairs_grids: bool | None = None
    plot_semivariogram_n_pairs_fontsize: float = 5.0
    plot_make_report_calibration_scatter: bool = False
    plot_make_report_calibration_distributions: bool = False
    plot_make_report_performance_distributions: bool = False
    plot_make_report_variance_mode_comparison: bool = False
    plot_report_distribution_modes_list: Iterable[Iterable[str]] | Iterable[str] | str | None = None
    plot_report_distribution_kde_bw_scale: float | None = None
    plot_write_report_figures: bool = True
    plot_write_diagnostic_figures: bool = False


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


def _write_blocked_cv_readme(
    blocked_cv_root: Path,
    *,
    config: BlockedCVConfig,
) -> Path:
    """
    Write a concise README describing blocked_CV output organization and policy.
    """
    readme_path = blocked_cv_root.joinpath("README.md")
    lines = [
        "# blocked_CV outputs",
        "",
        "This folder contains blocked cross-validation artifacts for the GPR along-core pipeline.",
        "",
        "## Directory layout",
        "- `csv/folds/`: fold assignment and fold-structure summaries.",
        "- `csv/predictions/`: point-level held-out predictions (largest tables).",
        "- `csv/metrics/`: fold/biopsy/cohort aggregated performance tables.",
        "- `csv/diagnostics/`: fit status and integrity/audit tables.",
        "- `figures/`: blocked_CV figures (when enabled in later phases).",
        "",
        "## Output policy",
        "- Report/repro tables are always written.",
        "- Large debug tables are controlled by `write_blocked_cv_debug_csvs`.",
        "- Optional eligible-only views are controlled by `write_blocked_cv_eligible_views`.",
        f"- Current run setting: `write_blocked_cv_debug_csvs = {bool(config.write_debug_csvs)}`.",
        f"- Current run setting: `write_blocked_cv_eligible_views = {bool(config.write_eligible_views)}`.",
        "",
        "## Report-facing CSVs (always produced)",
        "- `csv/metrics/blocked_cv_cohort_summary_all.csv`",
        "- `csv/metrics/blocked_cv_biopsy_metrics_all.csv`",
        "- `csv/metrics/blocked_cv_fold_metrics_all.csv`",
        "- `csv/diagnostics/blocked_cv_fold_fit_status_all.csv`",
        "- `csv/folds/blocked_cv_fold_summary.csv`",
        "",
        "## Additional compare-mode CSVs (when enabled)",
        "- `csv/metrics/blocked_cv_cohort_summary_variance_compare_all.csv`",
        "- `csv/metrics/blocked_cv_biopsy_metrics_variance_compare_all.csv`",
        "- `csv/metrics/blocked_cv_fold_metrics_variance_compare_all.csv`",
        "- `csv/metrics/blocked_cv_variance_mode_summary_all.csv`",
        "",
        "## Eligible-view CSVs (when `write_blocked_cv_eligible_views=True`)",
        "- `csv/metrics/blocked_cv_fold_metrics_eligible.csv`",
        "- `csv/metrics/blocked_cv_biopsy_metrics_eligible.csv`",
        "- `csv/metrics/blocked_cv_cohort_summary_eligible.csv`",
        "- `csv/metrics/blocked_cv_fold_metrics_variance_compare_eligible.csv` (compare mode)",
        "- `csv/metrics/blocked_cv_biopsy_metrics_variance_compare_eligible.csv` (compare mode)",
        "- `csv/metrics/blocked_cv_cohort_summary_variance_compare_eligible.csv` (compare mode)",
        "- `csv/diagnostics/blocked_cv_eligibility_exclusions_all.csv`",
        "",
        "## Debug CSVs (produced only if `write_blocked_cv_debug_csvs=True`)",
        "- `csv/folds/blocked_cv_fold_map.csv`",
        "- `csv/predictions/blocked_cv_point_predictions_all.csv`",
        "- `csv/predictions/blocked_cv_point_predictions_variance_compare_all.csv`",
        "",
        "## Notes",
        "- `Patient ID` + `Bx index` define biopsy identity in this pipeline; `Bx ID` is also carried for traceability.",
        "- Kernel-specific slice files may be produced when per-kernel toggles are enabled.",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


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
        "phase": "blocked_cv_scaffold",
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
        "write_debug_csvs": bool(config.write_debug_csvs),
        "write_eligible_views": bool(config.write_eligible_views),
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
    if config.write_debug_csvs:
        fold_map_df.to_csv(fold_map_path, index=False)
    fold_summary_df.to_csv(fold_summary_path, index=False)
    readme_path = _write_blocked_cv_readme(output_dir, config=config)
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
        "phase": "blocked_cv_fold_mapping",
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
        "write_debug_csvs": bool(config.write_debug_csvs),
        "write_eligible_views": bool(config.write_eligible_views),
        "fold_map_csv": str(fold_map_path) if config.write_debug_csvs else None,
        "fold_summary_csv": str(fold_summary_path),
        "readme_path": str(readme_path),
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
        var_obs_test = pd.to_numeric(g.get("var_obs_test", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        sd_test_latent = pd.to_numeric(g.get("sd_test_latent", pd.Series(dtype=float)), errors="coerce").to_numpy(float)

        resid_v = residual[np.isfinite(residual)]
        rstd_v = rstd[np.isfinite(rstd)]
        valid_nlpd = np.isfinite(residual) & np.isfinite(var_pred_used)
        indep_sd_test = np.sqrt(np.maximum(var_obs_test, 0.0))
        valid_delta_sd = np.isfinite(indep_sd_test) & np.isfinite(sd_test_latent)
        if np.any(valid_delta_sd):
            mean_mc_sd_test = float(np.mean(indep_sd_test[valid_delta_sd]))
            mean_gp_sd_test_latent = float(np.mean(sd_test_latent[valid_delta_sd]))
            pct_reduction_mean_sd_test_latent = (
                float(100.0 * (1.0 - mean_gp_sd_test_latent / mean_mc_sd_test))
                if mean_mc_sd_test > 0
                else np.nan
            )
        else:
            mean_mc_sd_test = np.nan
            mean_gp_sd_test_latent = np.nan
            pct_reduction_mean_sd_test_latent = np.nan
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
                "mean_mc_sd_test": mean_mc_sd_test,
                "mean_gp_sd_test_latent": mean_gp_sd_test_latent,
                "pct_reduction_mean_sd_test_latent": pct_reduction_mean_sd_test_latent,
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
        var_obs_test = pd.to_numeric(g.get("var_obs_test", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        sd_test_latent = pd.to_numeric(g.get("sd_test_latent", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        resid_v = residual[np.isfinite(residual)]
        rstd_v = rstd[np.isfinite(rstd)]
        indep_sd_test = np.sqrt(np.maximum(var_obs_test, 0.0))
        valid_delta_sd = np.isfinite(indep_sd_test) & np.isfinite(sd_test_latent)
        if np.any(valid_delta_sd):
            mean_mc_sd_test = float(np.mean(indep_sd_test[valid_delta_sd]))
            mean_gp_sd_test_latent = float(np.mean(sd_test_latent[valid_delta_sd]))
            pct_reduction_mean_sd_test_latent = (
                float(100.0 * (1.0 - mean_gp_sd_test_latent / mean_mc_sd_test))
                if mean_mc_sd_test > 0
                else np.nan
            )
        else:
            mean_mc_sd_test = np.nan
            mean_gp_sd_test_latent = np.nan
            pct_reduction_mean_sd_test_latent = np.nan

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
                "mean_mc_sd_test": mean_mc_sd_test,
                "mean_gp_sd_test_latent": mean_gp_sd_test_latent,
                "pct_reduction_mean_sd_test_latent": pct_reduction_mean_sd_test_latent,
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
        "mean_mc_sd_test",
        "mean_gp_sd_test_latent",
        "pct_reduction_mean_sd_test_latent",
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


def _normalize_bool_series(s: pd.Series, default: bool = False) -> pd.Series:
    """Normalize mixed-type truthy/falsy column to boolean series."""
    if s is None:
        return pd.Series([], dtype=bool)
    if s.dtype == bool:
        return s.fillna(default)
    txt = s.astype(str).str.strip().str.lower()
    true_vals = {"true", "1", "yes", "y", "t"}
    false_vals = {"false", "0", "no", "n", "f", "nan", "none", ""}
    out = pd.Series(default, index=s.index, dtype=bool)
    out[txt.isin(true_vals)] = True
    out[txt.isin(false_vals)] = False
    return out


def _attach_fold_summary_flags_to_metrics(
    metrics_df: pd.DataFrame,
    fold_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach fold-level structural flags from fold summary onto metrics rows."""
    if metrics_df.empty or fold_summary_df.empty:
        return metrics_df
    keys = ["Patient ID", "Bx index", "fold_id"]
    if not all(k in metrics_df.columns for k in keys):
        return metrics_df
    if not all(k in fold_summary_df.columns for k in keys):
        return metrics_df
    add_cols = [
        "contiguous_test_block",
        "test_meets_min_voxels_threshold",
        "test_meets_min_span_threshold",
        "merged_tail_fold",
        "rebalanced_two_fold_split",
        "split_strategy",
    ]
    use_cols = [c for c in add_cols if c in fold_summary_df.columns]
    if not use_cols:
        return metrics_df
    fold_flags = fold_summary_df[keys + use_cols].drop_duplicates(subset=keys)
    # keep metrics-side values when already present
    existing = set(metrics_df.columns)
    drop_right = [c for c in use_cols if c in existing]
    fold_flags_for_merge = fold_flags.drop(columns=drop_right) if drop_right else fold_flags
    return metrics_df.merge(fold_flags_for_merge, on=keys, how="left")


def _evaluate_fold_eligibility(
    fold_metrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add eligibility flags and reasons to fold-level metrics.

    Returns
    -------
    fold_metrics_with_eligibility:
        input table with columns `eligible_for_primary` and `exclude_reason`.
    exclusions_df:
        subset where `eligible_for_primary == False`.
    """
    if fold_metrics_df.empty:
        out = fold_metrics_df.copy()
        out["eligible_for_primary"] = False
        out["exclude_reason"] = ""
        return out, out

    out = fold_metrics_df.copy()
    reasons_per_row: list[str] = []
    eligible = np.ones(len(out), dtype=bool)

    check_cols = [
        ("train_test_total_match", "train_test_total_mismatch"),
        ("train_count_matches_fold_map", "train_count_mismatch"),
        ("test_count_matches_fold_map", "test_count_mismatch"),
        ("contiguous_test_block", "non_contiguous_test_block"),
        ("test_meets_min_voxels_threshold", "below_min_test_voxels"),
        ("test_meets_min_span_threshold", "below_min_test_span_mm"),
    ]
    checks = []
    for col, reason in check_cols:
        if col in out.columns:
            v = _normalize_bool_series(out[col], default=False).to_numpy(bool)
            checks.append((v, reason))
        else:
            checks.append((np.ones(len(out), dtype=bool), reason))

    n_rstd = pd.to_numeric(out.get("n_rstd_valid", pd.Series(dtype=float)), errors="coerce")
    has_rstd = np.isfinite(n_rstd.to_numpy(float)) & (n_rstd.to_numpy(float) > 0)
    checks.append((has_rstd, "no_valid_rstd"))

    for i in range(len(out)):
        row_reasons = []
        row_ok = True
        for mask, reason in checks:
            if not bool(mask[i]):
                row_ok = False
                row_reasons.append(reason)
        eligible[i] = row_ok
        reasons_per_row.append(";".join(row_reasons))

    out["eligible_for_primary"] = eligible
    out["exclude_reason"] = reasons_per_row
    exclusions = out.loc[~out["eligible_for_primary"]].copy()
    return out, exclusions


def _eligible_fold_keys_from_fold_metrics(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return unique eligible fold keys for filtering point-level tables."""
    if fold_metrics_df.empty or "eligible_for_primary" not in fold_metrics_df.columns:
        return pd.DataFrame(columns=["Patient ID", "Bx index", "fold_id", "kernel_label"])
    keys = ["Patient ID", "Bx index", "fold_id", "kernel_label"]
    keys = [k for k in keys if k in fold_metrics_df.columns]
    return fold_metrics_df.loc[fold_metrics_df["eligible_for_primary"], keys].drop_duplicates()


def _filter_predictions_by_eligible_folds(pred_df: pd.DataFrame, eligible_keys_df: pd.DataFrame) -> pd.DataFrame:
    """Filter prediction rows to eligible fold keys."""
    if pred_df.empty or eligible_keys_df.empty:
        return pred_df.iloc[0:0].copy()
    merge_keys = [k for k in ["Patient ID", "Bx index", "fold_id", "kernel_label"] if k in pred_df.columns and k in eligible_keys_df.columns]
    if not merge_keys:
        return pred_df.iloc[0:0].copy()
    return pred_df.merge(eligible_keys_df[merge_keys].drop_duplicates(), on=merge_keys, how="inner")


def run_blocked_cv_fit_predict(
    all_voxel_wise_dose_df: pd.DataFrame,
    semivariogram_df: pd.DataFrame,
    *,
    output_dir: Path,
    figs_dir: Path,
    csv_dir: Path,
    config: BlockedCVConfig,
) -> dict:
    """
    blocked_CV fit/predict path:
    - uses fold map from 3B logic,
    - runs strict train-only fit/predict for configured kernel specs,
    - writes centralized *_all CSV outputs (plus optional per-kernel slices).

    Returns a dict with:
    - status: compact status/path/count metadata for logging
    - artifacts: in-memory dataframes for downstream plotting without CSV rereads
    """
    del semivariogram_df  # phase 3C uses per-fold train-only semivariograms
    csv_subdirs = _blocked_cv_csv_subdirs(csv_dir)
    fold_map_df, fold_summary_df = build_blocked_cv_fold_map(
        all_voxel_wise_dose_df,
        config=config,
    )
    fold_map_path = csv_subdirs["folds"].joinpath("blocked_cv_fold_map.csv")
    fold_summary_path = csv_subdirs["folds"].joinpath("blocked_cv_fold_summary.csv")
    if config.write_debug_csvs and (not fold_map_path.exists()):
        fold_map_df.to_csv(fold_map_path, index=False)
    if not fold_summary_path.exists():
        fold_summary_df.to_csv(fold_summary_path, index=False)
    readme_path = _write_blocked_cv_readme(output_dir, config=config)

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
    if config.write_debug_csvs:
        pred_df.to_csv(pred_path, index=False)
    fold_status_df.to_csv(status_path, index=False)

    compare_path = None
    compare_summary_path = None
    compare_df = pd.DataFrame()
    compare_summary_df = pd.DataFrame()
    if config.compare_variance_modes:
        compare_df = pd.DataFrame(pred_compare_rows)
        compare_path = csv_subdirs["predictions"].joinpath("blocked_cv_point_predictions_variance_compare_all.csv")
        if config.write_debug_csvs:
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
    fold_metrics_df = _attach_fold_summary_flags_to_metrics(fold_metrics_df, fold_summary_df)
    fold_metrics_df, eligibility_exclusions_df = _evaluate_fold_eligibility(fold_metrics_df)
    fold_metrics_path = csv_subdirs["metrics"].joinpath("blocked_cv_fold_metrics_all.csv")
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    compare_fold_metrics_df = pd.DataFrame()
    compare_fold_metrics_path = None
    if config.compare_variance_modes:
        compare_fold_metrics_df = _compute_fold_metrics_from_predictions(
            compare_df,
            variance_mode_col="variance_mode",
        )
        compare_fold_metrics_df = _attach_fold_summary_flags_to_metrics(compare_fold_metrics_df, fold_summary_df)
        if (not compare_fold_metrics_df.empty) and (not fold_metrics_df.empty):
            join_keys = [k for k in ["Patient ID", "Bx index", "fold_id", "kernel_label"] if k in compare_fold_metrics_df.columns and k in fold_metrics_df.columns]
            if join_keys:
                elig_cols = join_keys + [c for c in ["eligible_for_primary", "exclude_reason"] if c in fold_metrics_df.columns]
                compare_fold_metrics_df = compare_fold_metrics_df.merge(
                    fold_metrics_df[elig_cols].drop_duplicates(),
                    on=join_keys,
                    how="left",
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

    # Phase 4E: optional eligible-only views for report-facing filtering.
    eligibility_exclusions_path = None
    fold_metrics_eligible_df = pd.DataFrame()
    fold_metrics_eligible_path = None
    biopsy_metrics_eligible_df = pd.DataFrame()
    biopsy_metrics_eligible_path = None
    cohort_summary_eligible_df = pd.DataFrame()
    cohort_summary_eligible_path = None
    compare_fold_metrics_eligible_df = pd.DataFrame()
    compare_fold_metrics_eligible_path = None
    compare_biopsy_metrics_eligible_df = pd.DataFrame()
    compare_biopsy_metrics_eligible_path = None
    compare_cohort_summary_eligible_df = pd.DataFrame()
    compare_cohort_summary_eligible_path = None
    if config.write_eligible_views:
        eligibility_exclusions_path = csv_subdirs["diagnostics"].joinpath("blocked_cv_eligibility_exclusions_all.csv")
        eligibility_exclusions_df.to_csv(eligibility_exclusions_path, index=False)

        fold_metrics_eligible_df = fold_metrics_df.loc[
            _normalize_bool_series(fold_metrics_df.get("eligible_for_primary", pd.Series(dtype=bool)), default=False)
        ].copy()
        fold_metrics_eligible_path = csv_subdirs["metrics"].joinpath("blocked_cv_fold_metrics_eligible.csv")
        fold_metrics_eligible_df.to_csv(fold_metrics_eligible_path, index=False)

        eligible_keys_df = _eligible_fold_keys_from_fold_metrics(fold_metrics_df)
        pred_eligible_df = _filter_predictions_by_eligible_folds(pred_df, eligible_keys_df)
        biopsy_metrics_eligible_df = _compute_biopsy_metrics_from_predictions(
            pred_eligible_df,
            variance_mode_col="predictive_variance_mode",
        )
        biopsy_metrics_eligible_path = csv_subdirs["metrics"].joinpath("blocked_cv_biopsy_metrics_eligible.csv")
        biopsy_metrics_eligible_df.to_csv(biopsy_metrics_eligible_path, index=False)

        cohort_summary_eligible_df = _compute_cohort_summary_from_biopsy_metrics(biopsy_metrics_eligible_df)
        cohort_summary_eligible_path = csv_subdirs["metrics"].joinpath("blocked_cv_cohort_summary_eligible.csv")
        cohort_summary_eligible_df.to_csv(cohort_summary_eligible_path, index=False)

        if config.compare_variance_modes:
            compare_fold_metrics_eligible_df = compare_fold_metrics_df.loc[
                _normalize_bool_series(compare_fold_metrics_df.get("eligible_for_primary", pd.Series(dtype=bool)), default=False)
            ].copy()
            compare_fold_metrics_eligible_path = csv_subdirs["metrics"].joinpath(
                "blocked_cv_fold_metrics_variance_compare_eligible.csv"
            )
            compare_fold_metrics_eligible_df.to_csv(compare_fold_metrics_eligible_path, index=False)

            compare_pred_eligible_df = _filter_predictions_by_eligible_folds(compare_df, eligible_keys_df)
            compare_biopsy_metrics_eligible_df = _compute_biopsy_metrics_from_predictions(
                compare_pred_eligible_df,
                variance_mode_col="variance_mode",
            )
            compare_biopsy_metrics_eligible_path = csv_subdirs["metrics"].joinpath(
                "blocked_cv_biopsy_metrics_variance_compare_eligible.csv"
            )
            compare_biopsy_metrics_eligible_df.to_csv(compare_biopsy_metrics_eligible_path, index=False)

            compare_cohort_summary_eligible_df = _compute_cohort_summary_from_biopsy_metrics(
                compare_biopsy_metrics_eligible_df
            )
            compare_cohort_summary_eligible_path = csv_subdirs["metrics"].joinpath(
                "blocked_cv_cohort_summary_variance_compare_eligible.csv"
            )
            compare_cohort_summary_eligible_df.to_csv(compare_cohort_summary_eligible_path, index=False)

    # Optional per-kernel slices (subsets of centralized *_all outputs).
    kernel_labels_run = sorted(pd.unique(pred_df["kernel_label"])) if not pred_df.empty else []
    if config.write_debug_csvs and config.write_per_kernel_predictions_csvs and not pred_df.empty:
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
    if config.write_debug_csvs and config.compare_variance_modes and config.write_per_kernel_variance_compare_csvs and not compare_df.empty:
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

    artifacts = {
        "fold_map_df": fold_map_df,
        "fold_summary_df": fold_summary_df,
        "pred_df": pred_df,
        "compare_df": compare_df,
        "fold_status_df": fold_status_df,
        "fold_metrics_df": fold_metrics_df,
        "compare_fold_metrics_df": compare_fold_metrics_df,
        "biopsy_metrics_df": biopsy_metrics_df,
        "compare_biopsy_metrics_df": compare_biopsy_metrics_df,
        "cohort_summary_df": cohort_summary_df,
        "compare_cohort_summary_df": compare_cohort_summary_df,
        "fold_metrics_eligible_df": fold_metrics_eligible_df,
        "compare_fold_metrics_eligible_df": compare_fold_metrics_eligible_df,
        "biopsy_metrics_eligible_df": biopsy_metrics_eligible_df,
        "compare_biopsy_metrics_eligible_df": compare_biopsy_metrics_eligible_df,
        "cohort_summary_eligible_df": cohort_summary_eligible_df,
        "compare_cohort_summary_eligible_df": compare_cohort_summary_eligible_df,
        "eligibility_exclusions_df": eligibility_exclusions_df,
    }

    status = {
        "phase": "blocked_cv_fit_predict",
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
        "write_debug_csvs": bool(config.write_debug_csvs),
        "write_eligible_views": bool(config.write_eligible_views),
        "readme_path": str(readme_path),
        "point_predictions_csv": str(pred_path) if config.write_debug_csvs else None,
        "fold_fit_status_csv": str(status_path),
        "fold_metrics_csv": str(fold_metrics_path),
        "biopsy_metrics_csv": str(biopsy_metrics_path),
        "cohort_summary_csv": str(cohort_summary_path),
        "eligibility_exclusions_csv": str(eligibility_exclusions_path) if eligibility_exclusions_path is not None else None,
        "fold_metrics_eligible_csv": str(fold_metrics_eligible_path) if fold_metrics_eligible_path is not None else None,
        "biopsy_metrics_eligible_csv": str(biopsy_metrics_eligible_path) if biopsy_metrics_eligible_path is not None else None,
        "cohort_summary_eligible_csv": str(cohort_summary_eligible_path) if cohort_summary_eligible_path is not None else None,
        "point_predictions_compare_csv": (str(compare_path) if (compare_path is not None and config.write_debug_csvs) else None),
        "fold_metrics_compare_csv": str(compare_fold_metrics_path) if compare_fold_metrics_path is not None else None,
        "biopsy_metrics_compare_csv": str(compare_biopsy_metrics_path) if compare_biopsy_metrics_path is not None else None,
        "cohort_summary_compare_csv": str(compare_cohort_summary_path) if compare_cohort_summary_path is not None else None,
        "variance_mode_summary_csv": str(compare_summary_path) if compare_summary_path is not None else None,
        "fold_metrics_compare_eligible_csv": str(compare_fold_metrics_eligible_path) if compare_fold_metrics_eligible_path is not None else None,
        "biopsy_metrics_compare_eligible_csv": str(compare_biopsy_metrics_eligible_path) if compare_biopsy_metrics_eligible_path is not None else None,
        "cohort_summary_compare_eligible_csv": str(compare_cohort_summary_eligible_path) if compare_cohort_summary_eligible_path is not None else None,
        "n_point_prediction_rows": int(len(pred_df)),
        "n_fold_metrics_rows": int(len(fold_metrics_df)),
        "n_biopsy_metrics_rows": int(len(biopsy_metrics_df)),
        "n_cohort_summary_rows": int(len(cohort_summary_df)),
        "n_fold_metrics_eligible_rows": int(len(fold_metrics_eligible_df)),
        "n_biopsy_metrics_eligible_rows": int(len(biopsy_metrics_eligible_df)),
        "n_cohort_summary_eligible_rows": int(len(cohort_summary_eligible_df)),
        "n_eligibility_exclusion_rows": int(len(eligibility_exclusions_df)),
        "n_point_prediction_compare_rows": int(len(compare_df)),
        "n_fold_metrics_compare_rows": int(len(compare_fold_metrics_df)),
        "n_biopsy_metrics_compare_rows": int(len(compare_biopsy_metrics_df)),
        "n_cohort_summary_compare_rows": int(len(compare_cohort_summary_df)),
        "n_fold_metrics_compare_eligible_rows": int(len(compare_fold_metrics_eligible_df)),
        "n_biopsy_metrics_compare_eligible_rows": int(len(compare_biopsy_metrics_eligible_df)),
        "n_cohort_summary_compare_eligible_rows": int(len(compare_cohort_summary_eligible_df)),
        "n_fold_status_rows": int(len(fold_status_df)),
        "n_fold_status_ok": int((fold_status_df.get("status", pd.Series(dtype=str)) == "ok").sum()) if len(fold_status_df) else 0,
        "n_fold_status_error": int((fold_status_df.get("status", pd.Series(dtype=str)) == "error").sum()) if len(fold_status_df) else 0,
    }
    return {"status": status, "artifacts": artifacts}


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
    Backward-compatible alias returning status-only payload.
    """
    result = run_blocked_cv_fit_predict(
        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
        semivariogram_df=semivariogram_df,
        output_dir=output_dir,
        figs_dir=figs_dir,
        csv_dir=csv_dir,
        config=config,
    )
    return result["status"]


def _sanitize_token(value) -> str:
    """Filesystem-safe token for figure names/directories."""
    txt = str(value)
    chars = []
    for ch in txt:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out if out else "na"


def _blocked_cv_biopsy_label(
    patient_id,
    bx_index,
    label_map: dict[tuple[str, int], str] | None,
) -> str:
    key = (patient_id, int(bx_index))
    if label_map and key in label_map:
        return str(label_map[key])
    return f"P{patient_id} Bx{int(bx_index)}"


def _build_blocked_cv_fold_plot_payload(
    *,
    all_voxel_wise_dose_df: pd.DataFrame,
    fold_map_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame | None,
    pred_group_df: pd.DataFrame,
    patient_id,
    bx_index: int,
    fold_id: int,
    kernel_name: str,
    kernel_param,
    config: BlockedCVConfig,
) -> dict | None:
    """
    Build train-fold GP objects used by blocked_CV plotters for one fold.
    """
    g_bx = all_voxel_wise_dose_df[
        (all_voxel_wise_dose_df["Patient ID"] == patient_id)
        & (all_voxel_wise_dose_df["Bx index"] == bx_index)
    ].copy()
    if g_bx.empty:
        return None

    fold_rows = fold_map_df[
        (fold_map_df["Patient ID"] == patient_id)
        & (fold_map_df["Bx index"] == bx_index)
        & (fold_map_df["fold_id"] == fold_id)
    ].copy()
    if fold_rows.empty or "is_test" not in fold_rows.columns:
        return None

    fold_metrics_row = None
    if isinstance(fold_metrics_df, pd.DataFrame) and not fold_metrics_df.empty:
        fm = fold_metrics_df[
            (fold_metrics_df["Patient ID"] == patient_id)
            & (fold_metrics_df["Bx index"] == bx_index)
            & (fold_metrics_df["fold_id"].astype(int) == int(fold_id))
            & (fold_metrics_df["kernel_label"].astype(str) == str(pred_group_df["kernel_label"].iloc[0]))
        ].copy()
        if "predictive_variance_mode" in fm.columns:
            fm = fm[fm["predictive_variance_mode"].astype(str) == str(config.primary_predictive_variance_mode)]
        if not fm.empty:
            fold_metrics_row = fm.iloc[0]

    is_test = _normalize_bool_series(fold_rows["is_test"], default=False)
    test_voxels = set(fold_rows.loc[is_test, "Voxel index"].astype(int).tolist())
    if not test_voxels:
        return None

    train_df = g_bx[~g_bx["Voxel index"].isin(test_voxels)].copy()
    test_df = g_bx[g_bx["Voxel index"].isin(test_voxels)].copy()
    if train_df.empty or test_df.empty:
        return None

    X_train, y_train, var_n_train, _pv_train = gpr_pf.build_voxel_targets_and_noise(
        train_df,
        patient_id=patient_id,
        bx_index=bx_index,
        target_stat=config.target_stat,
        position_mode=config.position_mode,
    )
    X_test, y_test, var_n_test, _pv_test = gpr_pf.build_voxel_targets_and_noise(
        test_df,
        patient_id=patient_id,
        bx_index=bx_index,
        target_stat=config.target_stat,
        position_mode=config.position_mode,
    )
    X_all, _y_all, _var_n_all, _pv_all = gpr_pf.build_voxel_targets_and_noise(
        g_bx,
        patient_id=patient_id,
        bx_index=bx_index,
        target_stat=config.target_stat,
        position_mode=config.position_mode,
    )
    if len(X_train) < 3 or len(X_all) < 1:
        return None

    row0 = pred_group_df.iloc[0]
    ell = float(row0.get("ell", np.nan))
    sigma_f2 = float(row0.get("sigma_f2", np.nan))
    nugget = float(row0.get("nugget", np.nan))
    nu = float(row0.get("nu", np.nan))
    if not np.isfinite(ell) or not np.isfinite(sigma_f2):
        return None
    if not np.isfinite(nugget):
        nugget = 0.0
    if not np.isfinite(nu):
        if str(kernel_name) == "matern" and pd.notna(kernel_param):
            nu = float(kernel_param)
        elif str(kernel_name) == "exp":
            nu = 0.5
        else:
            nu = 1.5

    hyp = gpr_pf.GPHyperparams(
        sigma_f2=float(sigma_f2),
        ell=float(ell),
        nugget=float(nugget),
        nu=float(nu),
        kernel=str(kernel_name),
    )
    mu_star, sd_star = gpr_pf.gp_posterior(
        X_train,
        y_train,
        var_n_train,
        hyp,
        X_star=X_all,
        mean_mode=config.mean_mode,
    )
    mu_X, sd_X = gpr_pf.gp_posterior(
        X_train,
        y_train,
        var_n_train,
        hyp,
        X_star=X_train,
        mean_mode=config.mean_mode,
    )
    gp_res = {
        "X_star": X_all,
        "mu_star": mu_star,
        "sd_star": sd_star,
        "X": X_train,
        "y": y_train,
        "var_n": var_n_train,
        "mu_X": mu_X,
        "sd_X": sd_X,
        "hyperparams": hyp,
    }

    sv_train = gpr_sv.compute_semivariogram_pairwise(
        train_df,
        voxel_size_mm=float(config.semivariogram_voxel_size_mm),
        max_lag_voxels=None,
        position_mode=config.position_mode,
        lag_bin_width_mm=config.semivariogram_lag_bin_width_mm,
    )
    sv_train = sv_train[np.isfinite(sv_train["semivariance"])].copy()
    if sv_train.empty:
        return None
    sv_train["Patient ID"] = patient_id
    sv_train["Bx index"] = bx_index

    x_test_min = float(np.min(X_test)) if len(X_test) else np.nan
    x_test_max = float(np.max(X_test)) if len(X_test) else np.nan

    return {
        "gp_res": gp_res,
        "sv_train": sv_train,
        "ell": float(ell),
        "nugget": float(nugget),
        "X_test": X_test,
        "y_test": y_test,
        "var_n_test": var_n_test,
        "x_test_min": x_test_min,
        "x_test_max": x_test_max,
        "fold_metrics_row": fold_metrics_row,
    }


def _plot_blocked_cv_variogram_profile_pair(
    *,
    all_voxel_wise_dose_df: pd.DataFrame,
    fold_map_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame | None,
    pred_group_df: pd.DataFrame,
    patient_id,
    bx_index: int,
    fold_id: int,
    kernel_label: str,
    kernel_name: str,
    kernel_param,
    config: BlockedCVConfig,
    save_dir: Path,
    file_name_base: str,
    title_label: str,
) -> list[Path]:
    """
    Render one blocked_CV paired semivariogram/profile figure for one fold.
    """
    payload = _build_blocked_cv_fold_plot_payload(
        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
        fold_map_df=fold_map_df,
        fold_metrics_df=fold_metrics_df,
        pred_group_df=pred_group_df,
        patient_id=patient_id,
        bx_index=bx_index,
        fold_id=fold_id,
        kernel_name=kernel_name,
        kernel_param=kernel_param,
        config=config,
    )
    if payload is None:
        return []
    gp_res = payload["gp_res"]
    sv_train = payload["sv_train"]
    ell = payload["ell"]
    nugget = payload["nugget"]
    fold_metrics_row = payload.get("fold_metrics_row", None)
    X_test = np.asarray(payload.get("X_test", np.array([])), dtype=float)
    y_test = np.asarray(payload.get("y_test", np.array([])), dtype=float)
    x_test_min = payload.get("x_test_min", np.nan)
    x_test_max = payload.get("x_test_max", np.nan)

    metrics_left = (
        rf"$\hat{{\ell}}_{{b,-f}} = {ell:.1f}~\mathrm{{mm}},\ "
        rf"\hat{{\tau}}_{{b,-f}}^2 = {gpr_pp._format_nugget(nugget)}\ \mathrm{{Gy}}^2$"
    )
    metrics_right = None
    if isinstance(fold_metrics_row, pd.Series):
        delta_test = float(pd.to_numeric(fold_metrics_row.get("pct_reduction_mean_sd_test_latent", np.nan), errors="coerce"))
        n_test_pts = float(pd.to_numeric(fold_metrics_row.get("n_test_points", np.nan), errors="coerce"))
        if np.isfinite(delta_test):
            if np.isfinite(n_test_pts):
                metrics_right = (
                    rf"$n_{{b,f}}^{{\mathrm{{test}}}} = {int(n_test_pts)},\ "
                    rf"{gpr_pp._delta_sd_symbol(test_fold_latent=True)} = {delta_test:.1f}\%$"
                )
            else:
                metrics_right = rf"${gpr_pp._delta_sd_symbol(test_fold_latent=True)} = {delta_test:.1f}\%$"

    out_paths = gpr_pp.plot_variogram_and_profile_pair(
        sv_train,
        patient_id,
        bx_index,
        gp_res,
        save_dir=save_dir,
        file_name_base=file_name_base,
        save_formats=("pdf", "svg"),
        title_label=title_label,
        metrics_row=pd.Series({"ell": ell, "nugget": nugget}),
        include_kernel_legend=True,
        kernel_legend_label=kernel_label,
        annotate_semivariogram_n_pairs=_show_semivariogram_n_pairs_paired(config),
        semivariogram_n_pairs_fontsize=float(config.plot_semivariogram_n_pairs_fontsize),
        title_fontsize=max(float(getattr(gpr_pp, "TITLE_FONTSIZE", 14)) - 2.0, 1.0),
        create_subdir_for_stem=False,
        X_test=X_test,
        y_test=y_test,
        x_test_min=None if not np.isfinite(x_test_min) else float(x_test_min),
        x_test_max=None if not np.isfinite(x_test_max) else float(x_test_max),
        metrics_left_override=metrics_left,
        metrics_right_override=metrics_right,
    )
    return [Path(p) for p in out_paths]


def _selection_token(values, *, none_token: str = "all") -> str:
    """Compact token for filename suffixes from optional list-like selectors."""
    if values is None:
        return none_token
    vals = [str(v) for v in values]
    if not vals:
        return "none"
    if len(vals) <= 4:
        return _sanitize_token("-".join(vals))
    return _sanitize_token(f"n{len(vals)}")


def _show_semivariogram_n_pairs_paired(config: BlockedCVConfig) -> bool:
    """Resolve paired semivariogram n-pairs toggle."""
    if config.plot_semivariogram_show_n_pairs_paired is None:
        return False
    return bool(config.plot_semivariogram_show_n_pairs_paired)


def _show_semivariogram_n_pairs_grids(config: BlockedCVConfig) -> bool:
    """Resolve semivariogram-grid n-pairs toggle."""
    if config.plot_semivariogram_show_n_pairs_grids is None:
        return False
    return bool(config.plot_semivariogram_show_n_pairs_grids)


def _annotate_semivariogram_n_pairs(
    ax,
    *,
    h: np.ndarray,
    gamma_hat: np.ndarray,
    n_pairs: np.ndarray,
    fontsize: float,
    model_h: np.ndarray | None = None,
    model_gamma: np.ndarray | None = None,
) -> None:
    """Delegate to shared robust dynamic placement used in production plots."""
    gpr_pp._annotate_semivariogram_n_pairs_dynamic(
        ax,
        h=np.asarray(h, dtype=float),
        gamma_hat=np.asarray(gamma_hat, dtype=float),
        n_pairs=np.asarray(n_pairs, dtype=float),
        fontsize=float(fontsize),
        model_h=None if model_h is None else np.asarray(model_h, dtype=float),
        model_gamma=None if model_gamma is None else np.asarray(model_gamma, dtype=float),
    )


def _draw_blocked_cv_profile_axis(
    ax,
    *,
    payload: dict,
    kernel_label: str,
    title_label: str,
) -> tuple[list, list]:
    """Draw one blocked_CV profile panel on an existing axis."""
    gp_res = payload["gp_res"]
    X_star = gp_res["X_star"]
    mu_star = gp_res["mu_star"]
    sd_star = gp_res["sd_star"]
    X = gp_res["X"]
    y = gp_res["y"]
    indep_sd = np.sqrt(np.maximum(gp_res["var_n"], 0))
    X_test = np.asarray(payload.get("X_test", np.array([])), dtype=float)
    y_test = np.asarray(payload.get("y_test", np.array([])), dtype=float)

    gp_mean_label = gpr_pp._gp_mean_legend_label(
        include_kernel_legend=True,
        kernel_legend_label=kernel_label,
    )
    ax.plot(X_star, mu_star, lw=2.0, color=gpr_pp.PRIMARY_LINE_COLOR, label=gp_mean_label, zorder=3)
    ax.fill_between(X_star, mu_star - 1.96 * sd_star, mu_star + 1.96 * sd_star, alpha=0.12, color=gpr_pp.PRIMARY_LINE_COLOR, label="95% band", zorder=1)
    ax.fill_between(X_star, mu_star - 1.0 * sd_star, mu_star + 1.0 * sd_star, alpha=0.22, color=gpr_pp.PRIMARY_LINE_COLOR, label="68% band", zorder=2)
    sigma_mc_voxel = gpr_pp._sigma_mc_symbol(mean=False)
    ax.errorbar(X, y, yerr=2 * indep_sd, fmt="s", ms=3.0, lw=1.0, color="#1b8a5a", label=rf"$\widetilde{{D}}_{{b,v}}\pm2{sigma_mc_voxel}$", zorder=4)
    ax.errorbar(X, y, yerr=indep_sd, fmt="o", ms=3.0, lw=1.0, color="#c75000", label=rf"$\widetilde{{D}}_{{b,v}}\pm{sigma_mc_voxel}$", zorder=5)
    if X_test.size:
        ax.plot(X_test, y_test, "x", ms=4.0, mew=1.0, color="black", label=r"Held-out $\widetilde{D}_{b,v}$", zorder=6)
        x_min = float(np.nanmin(X_test))
        x_max = float(np.nanmax(X_test))
        if np.isfinite(x_min) and np.isfinite(x_max):
            # Keep fold span context without adding an extra legend entry.
            ax.axvspan(x_min, x_max, color="#d0d0d0", alpha=0.08, zorder=0)

    ax.set_xlabel(r"Axial position along biopsy $z$ (mm)", fontsize=gpr_pp._fs_label())
    ax.set_ylabel(r"Dose along core $D_b(z)$ (Gy)", fontsize=gpr_pp._fs_label())
    ymin, ymax = ax.get_ylim()
    if np.isfinite(ymax):
        ax.set_ylim(bottom=0 if ymin < 0 else ymin, top=ymax)
    elif ymin < 0:
        ax.set_ylim(bottom=0)
    gpr_pp._apply_axis_style(ax)
    gpr_pp._apply_per_biopsy_ticks(ax)

    fold_metrics_row = payload.get("fold_metrics_row", None)
    delta_test = np.nan
    n_test_pts = np.nan
    if isinstance(fold_metrics_row, pd.Series):
        delta_test = float(pd.to_numeric(fold_metrics_row.get("pct_reduction_mean_sd_test_latent", np.nan), errors="coerce"))
        n_test_pts = float(pd.to_numeric(fold_metrics_row.get("n_test_points", np.nan), errors="coerce"))

    if np.isfinite(delta_test):
        if np.isfinite(n_test_pts):
            metrics_str = (
                rf"$n_{{b,f}}^{{\mathrm{{test}}}} = {int(n_test_pts)},\ "
                rf"{gpr_pp._delta_sd_symbol(test_fold_latent=True)} = {delta_test:.1f}\%$"
            )
        else:
            metrics_str = rf"${gpr_pp._delta_sd_symbol(test_fold_latent=True)} = {delta_test:.1f}\%$"
    else:
        # Fallback if fold-level held-out summary is unavailable.
        mean_dose = float(np.nanmean(gp_res["mu_X"])) if gp_res.get("mu_X") is not None else np.nan
        shrink = 100.0 * (1 - np.nanmean(gp_res["sd_X"]) / np.nanmean(indep_sd)) if np.nanmean(indep_sd) > 0 else np.nan
        metrics_str = (
            rf"$\overline{{\mu}}^{{\mathrm{{GP}}}}_b = {mean_dose:.2f}\ \mathrm{{Gy}},\ "
            rf"{gpr_pp._delta_sd_symbol()} = {shrink:.1f}\%$"
        )
    ax.text(
        0.98,
        1.04,
        metrics_str,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=gpr_pp._fs_legend(),
        bbox=gpr_pp.ANNOT_BBOX,
    )
    title_fs = max(float(getattr(gpr_pp, "TITLE_FONTSIZE", 14)) - 2.0, 1.0)
    ax.set_title(title_label, fontsize=title_fs, loc="left")
    return ax.get_legend_handles_labels()


def _draw_blocked_cv_semivariogram_axis(
    ax,
    *,
    payload: dict,
    title_label: str,
    show_n_pairs: bool = False,
    n_pairs_fontsize: float = 5.0,
) -> tuple[list, list]:
    """Draw one blocked_CV semivariogram panel on an existing axis."""
    sv = payload["sv_train"].sort_values("h_mm")
    h = sv["h_mm"].to_numpy(float)
    gamma_hat = sv["semivariance"].to_numpy(float)
    hyperparams = payload["gp_res"]["hyperparams"]
    kernel = getattr(hyperparams, "kernel", "matern")
    if kernel == "rbf":
        gamma_model = gpr_pf.rbf_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0) + hyperparams.nugget
        label_model = r"Fitted $\gamma_b(h)$ (RBF)"
    elif kernel == "exp":
        gamma_model = gpr_pf.exp_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0) + hyperparams.nugget
        label_model = r"Fitted $\gamma_b(h)$ (Exp)"
    else:
        gamma_model = gpr_pf.matern_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0, hyperparams.nu) + hyperparams.nugget
        label_model = rf"Fitted $\gamma_b(h)$ (Matérn, $\nu={hyperparams.nu}$)"

    ax.plot(h, gamma_hat, "o", ms=4, color=gpr_pp.PRIMARY_LINE_COLOR, label=r"Empirical $\widehat{\gamma}_b(h)$")
    ax.plot(h, gamma_model, "-", lw=2.0, color=gpr_pp.OVERLAY_LINE_COLOR, label=label_model)
    if show_n_pairs and ("n_pairs" in sv.columns):
        n_pairs = pd.to_numeric(sv["n_pairs"], errors="coerce").to_numpy(float)
        _annotate_semivariogram_n_pairs(
            ax,
            h=h,
            gamma_hat=gamma_hat,
            n_pairs=n_pairs,
            fontsize=float(n_pairs_fontsize),
            model_h=h,
            model_gamma=gamma_model,
        )
    ax.set_xlabel(r"Lag $h\ \text{(mm)}$", fontsize=gpr_pp._fs_label())
    ax.set_ylabel(r"Semivariance $\gamma_b(h)$ (Gy$^2$)", fontsize=gpr_pp._fs_label())
    gpr_pp._apply_axis_style(ax)
    gpr_pp._apply_per_biopsy_ticks(ax)
    gpr_pp._enforce_integer_major_xticks(ax)
    metrics_str = (
        rf"$\hat{{\ell}}_{{b,-f}} = {payload['ell']:.1f}~\mathrm{{mm}},\ "
        rf"\hat{{\tau}}_{{b,-f}}^2 = {gpr_pp._format_nugget(payload['nugget'])}\ \mathrm{{Gy}}^2$"
    )
    ax.text(
        0.98,
        1.04,
        metrics_str,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=gpr_pp._fs_legend(),
        bbox=gpr_pp.ANNOT_BBOX,
    )
    title_fs = max(float(getattr(gpr_pp, "TITLE_FONTSIZE", 14)) - 2.0, 1.0)
    ax.set_title(title_label, fontsize=title_fs, loc="left")
    return ax.get_legend_handles_labels()


def run_blocked_cv_plots(
    *,
    fit_predict_artifacts: dict,
    all_voxel_wise_dose_df: pd.DataFrame,
    output_dir: Path,
    figs_dir: Path,
    csv_dir: Path,
    config: BlockedCVConfig,
) -> dict:
    """
    blocked_CV plotting lane.

    First implemented figure family:
    - paired semivariogram/profile figures per (patient, biopsy, fold, kernel)
    """
    def _plot_progress(msg: str) -> None:
        print(f"[blocked_CV][plots] {msg}")

    del output_dir, csv_dir
    pred_df = fit_predict_artifacts.get("pred_df", pd.DataFrame()) if isinstance(fit_predict_artifacts, dict) else pd.DataFrame()
    fold_map_df = fit_predict_artifacts.get("fold_map_df", pd.DataFrame()) if isinstance(fit_predict_artifacts, dict) else pd.DataFrame()
    fold_summary_df = fit_predict_artifacts.get("fold_summary_df", pd.DataFrame()) if isinstance(fit_predict_artifacts, dict) else pd.DataFrame()
    fold_metrics_df = fit_predict_artifacts.get("fold_metrics_df", pd.DataFrame()) if isinstance(fit_predict_artifacts, dict) else pd.DataFrame()
    biopsy_metrics_df = fit_predict_artifacts.get("biopsy_metrics_df", pd.DataFrame()) if isinstance(fit_predict_artifacts, dict) else pd.DataFrame()
    compare_biopsy_metrics_df = fit_predict_artifacts.get("compare_biopsy_metrics_df", pd.DataFrame()) if isinstance(fit_predict_artifacts, dict) else pd.DataFrame()
    n_pred = int(len(pred_df)) if isinstance(pred_df, pd.DataFrame) else 0

    paired_saved = []
    paired_errors = []
    profile_grid_saved = []
    profile_grid_errors = []
    semivariogram_grid_saved = []
    semivariogram_grid_errors = []
    report_calibration_saved = []
    report_calibration_errors = []
    report_performance_saved = []
    report_performance_errors = []
    report_variance_compare_saved = []
    report_variance_compare_errors = []
    n_candidates = 0
    n_selected = 0
    plot_keys_df = pd.DataFrame()
    need_plot_keys = (
        config.plot_write_report_figures
        and (
            config.plot_make_paired_semivariogram_profile
            or config.plot_make_profile_grids
            or config.plot_make_semivariogram_grids
        )
    )
    _plot_progress("building plot key selection table")

    def _filter_report_biopsy_df(df: pd.DataFrame, *, use_variance_mode: bool) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if config.plot_kernel_labels is not None and "kernel_label" in out.columns:
            allowed = {str(k) for k in config.plot_kernel_labels}
            out = out[out["kernel_label"].astype(str).isin(allowed)].copy()
        if use_variance_mode and "variance_mode" in out.columns:
            mode_sel = str(config.plot_variance_mode)
            if mode_sel == "primary":
                mode_sel = str(config.primary_predictive_variance_mode)
            out = out[out["variance_mode"].astype(str) == mode_sel].copy()
        return out

    if (
        isinstance(pred_df, pd.DataFrame)
        and isinstance(fold_map_df, pd.DataFrame)
        and not pred_df.empty
        and not fold_map_df.empty
        and need_plot_keys
    ):
        keys = ["Patient ID", "Bx index", "fold_id", "kernel_label", "kernel_name", "kernel_param"]
        plot_keys_df = pred_df[keys].drop_duplicates()
        n_candidates = int(len(plot_keys_df))

        if config.plot_kernel_labels is not None:
            allowed_kernels = {str(k) for k in config.plot_kernel_labels}
            plot_keys_df = plot_keys_df[plot_keys_df["kernel_label"].astype(str).isin(allowed_kernels)]

        if config.plot_fold_ids is not None:
            allowed_folds = {int(f) for f in config.plot_fold_ids}
            plot_keys_df = plot_keys_df[plot_keys_df["fold_id"].astype(int).isin(allowed_folds)]

        if not fold_summary_df.empty:
            fs_cols = ["Patient ID", "Bx index", "fold_id", "merged_tail_fold", "rebalanced_two_fold_split", "test_z_min_mm"]
            fs_cols = [c for c in fs_cols if c in fold_summary_df.columns]
            if {"Patient ID", "Bx index", "fold_id"}.issubset(fs_cols):
                plot_keys_df = plot_keys_df.merge(
                    fold_summary_df[fs_cols].drop_duplicates(subset=["Patient ID", "Bx index", "fold_id"]),
                    on=["Patient ID", "Bx index", "fold_id"],
                    how="left",
                )
            if ("merged_tail_fold" in plot_keys_df.columns) and (not config.plot_include_merged_tail_folds):
                plot_keys_df = plot_keys_df[~_normalize_bool_series(plot_keys_df["merged_tail_fold"], default=False)]
            if ("rebalanced_two_fold_split" in plot_keys_df.columns) and (not config.plot_include_rebalanced_two_fold_splits):
                plot_keys_df = plot_keys_df[~_normalize_bool_series(plot_keys_df["rebalanced_two_fold_split"], default=False)]

        if config.plot_fold_sort_mode == "z_start_mm" and "test_z_min_mm" in plot_keys_df.columns:
            plot_keys_df = plot_keys_df.sort_values(["Patient ID", "Bx index", "kernel_label", "test_z_min_mm", "fold_id"])
        else:
            plot_keys_df = plot_keys_df.sort_values(["Patient ID", "Bx index", "kernel_label", "fold_id"])

        if config.plot_max_folds_per_biopsy is not None:
            n_keep = max(int(config.plot_max_folds_per_biopsy), 1)
            plot_keys_df = (
                plot_keys_df.groupby(["Patient ID", "Bx index", "kernel_label"], sort=False, dropna=False)
                .head(n_keep)
                .copy()
            )
        n_selected = int(len(plot_keys_df))
        _plot_progress(f"plot keys selected: {n_selected}/{n_candidates}")
    else:
        if not need_plot_keys:
            _plot_progress("plot key selection skipped (all report plot families disabled)")
        elif not isinstance(pred_df, pd.DataFrame) or pred_df.empty:
            _plot_progress("plot key selection skipped (prediction table empty)")
        elif not isinstance(fold_map_df, pd.DataFrame) or fold_map_df.empty:
            _plot_progress("plot key selection skipped (fold map table empty)")
        else:
            _plot_progress("plot key selection skipped")

    if config.plot_make_paired_semivariogram_profile and config.plot_write_report_figures and not plot_keys_df.empty:
        _plot_progress("paired semivariogram+profile: starting")
        for _, row in plot_keys_df.iterrows():
            patient_id = row["Patient ID"]
            bx_index = int(row["Bx index"])
            fold_id = int(row["fold_id"])
            kernel_label = str(row["kernel_label"])
            kernel_name = str(row["kernel_name"])
            kernel_param = row["kernel_param"]

            pred_group_df = pred_df[
                (pred_df["Patient ID"] == patient_id)
                & (pred_df["Bx index"] == bx_index)
                & (pred_df["fold_id"] == fold_id)
                & (pred_df["kernel_label"] == kernel_label)
            ].copy()
            if pred_group_df.empty:
                continue

            patient_tok = _sanitize_token(patient_id)
            kernel_tok = _sanitize_token(kernel_label)
            fold_tok = _sanitize_token(fold_id)
            bx_tok = _sanitize_token(bx_index)
            file_name_base = (
                f"blocked_cv_variogram_profile_pair_patient_{patient_tok}"
                f"_bx_{bx_tok}_fold_{fold_tok}"
            )
            biopsy_label = _blocked_cv_biopsy_label(
                patient_id=patient_id,
                bx_index=bx_index,
                label_map=config.plot_grid_label_map,
            )
            title_label = f"{biopsy_label} | Fold {fold_id}"
            save_dir = (
                figs_dir
                .joinpath("report")
                .joinpath("patient")
                .joinpath(f"patient_{patient_tok}")
                .joinpath(f"bx_{bx_tok}")
                .joinpath("paired_semivariogram_profile")
                .joinpath(f"kernel_{kernel_tok}")
            )
            try:
                out_paths = _plot_blocked_cv_variogram_profile_pair(
                    all_voxel_wise_dose_df=all_voxel_wise_dose_df,
                    fold_map_df=fold_map_df,
                    fold_metrics_df=fold_metrics_df,
                    pred_group_df=pred_group_df,
                    patient_id=patient_id,
                    bx_index=bx_index,
                    fold_id=fold_id,
                    kernel_label=kernel_label,
                    kernel_name=kernel_name,
                    kernel_param=kernel_param,
                    config=config,
                    save_dir=save_dir,
                    file_name_base=file_name_base,
                    title_label=title_label,
                )
                paired_saved.extend([str(p) for p in out_paths])
            except Exception as e:
                paired_errors.append(
                    f"patient={patient_id}, bx={bx_index}, fold={fold_id}, kernel={kernel_label}: {e}"
                )
        _plot_progress(
            f"paired semivariogram+profile: complete "
            f"(saved={len(paired_saved)}, errors={len(paired_errors)})"
        )
    else:
        _plot_progress("paired semivariogram+profile: skipped")

    if config.plot_make_profile_grids and config.plot_write_report_figures and not plot_keys_df.empty:
        _plot_progress("profile grids: starting")
        gpr_pp._setup_matplotlib_defaults()
        kernels = sorted(pd.unique(plot_keys_df["kernel_label"]))
        if config.plot_patient_bx_list is None:
            bx_order = [
                (r["Patient ID"], int(r["Bx index"]))
                for _, r in plot_keys_df[["Patient ID", "Bx index"]].drop_duplicates().sort_values(["Patient ID", "Bx index"]).iterrows()
            ]
        else:
            available = {
                (r["Patient ID"], int(r["Bx index"]))
                for _, r in plot_keys_df[["Patient ID", "Bx index"]].drop_duplicates().iterrows()
            }
            bx_order = []
            for pid, bx in config.plot_patient_bx_list:
                key = (pid, int(bx))
                if key in available:
                    bx_order.append(key)
            if not bx_order:
                bx_order = [
                    (r["Patient ID"], int(r["Bx index"]))
                    for _, r in plot_keys_df[["Patient ID", "Bx index"]].drop_duplicates().sort_values(["Patient ID", "Bx index"]).iterrows()
                ]

        # Mode A: one biopsy across folds
        for kernel_label in kernels:
            kernel_tok = _sanitize_token(kernel_label)
            for patient_id, bx_index in bx_order:
                panel_df = plot_keys_df[
                    (plot_keys_df["Patient ID"] == patient_id)
                    & (plot_keys_df["Bx index"] == bx_index)
                    & (plot_keys_df["kernel_label"] == kernel_label)
                ].copy()
                if panel_df.empty:
                    continue
                if config.plot_fold_sort_mode == "z_start_mm" and "test_z_min_mm" in panel_df.columns:
                    panel_df = panel_df.sort_values(["test_z_min_mm", "fold_id"])
                else:
                    panel_df = panel_df.sort_values(["fold_id"])

                n_rows, n_cols = gpr_pp._grid_shape(len(panel_df), int(config.plot_grid_ncols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(gpr_pp.GRID_PER_BIOPSY_FIGSIZE[0] * n_cols, gpr_pp.GRID_PER_BIOPSY_FIGSIZE[1] * n_rows))
                axes = np.atleast_1d(axes).ravel()
                shared_handles, shared_labels = None, None
                for ax, (_, prow) in zip(axes, panel_df.iterrows()):
                    fold_id = int(prow["fold_id"])
                    kernel_name = str(prow["kernel_name"])
                    kernel_param = prow["kernel_param"]
                    pred_group_df = pred_df[
                        (pred_df["Patient ID"] == patient_id)
                        & (pred_df["Bx index"] == bx_index)
                        & (pred_df["fold_id"] == fold_id)
                        & (pred_df["kernel_label"] == kernel_label)
                    ].copy()
                    payload = _build_blocked_cv_fold_plot_payload(
                        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
                        fold_map_df=fold_map_df,
                        fold_metrics_df=fold_metrics_df,
                        pred_group_df=pred_group_df,
                        patient_id=patient_id,
                        bx_index=bx_index,
                        fold_id=fold_id,
                        kernel_name=kernel_name,
                        kernel_param=kernel_param,
                        config=config,
                    )
                    if payload is None:
                        ax.text(0.5, 0.5, f"Missing fold {fold_id}", ha="center", va="center")
                        ax.axis("off")
                        continue
                    biopsy_label = _blocked_cv_biopsy_label(patient_id, bx_index, config.plot_grid_label_map)
                    title_label = f"{biopsy_label} | Fold {fold_id}"
                    h, l = _draw_blocked_cv_profile_axis(
                        ax,
                        payload=payload,
                        kernel_label=kernel_label,
                        title_label=title_label,
                    )
                    if shared_handles is None and h:
                        shared_handles, shared_labels = h, l
                for ax in axes[len(panel_df):]:
                    ax.axis("off")
                if shared_handles:
                    fig.legend(
                        shared_handles,
                        shared_labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 1.07),
                        ncol=min(len(shared_handles), 6),
                        frameon=False,
                        fancybox=False,
                        fontsize=gpr_pp._fs_legend(),
                    )
                fig.tight_layout(rect=[0, 0, 1, 0.86])
                patient_tok = _sanitize_token(patient_id)
                bx_tok = _sanitize_token(bx_index)
                fold_sel_token = _selection_token(config.plot_fold_ids, none_token="allfolds")
                save_dir = (
                    figs_dir
                    .joinpath("report")
                    .joinpath("cohort")
                    .joinpath("grid_profiles")
                    .joinpath("by_biopsy_across_folds")
                    .joinpath(f"kernel_{kernel_tok}")
                )
                file_name_base = (
                    f"blocked_cv_profile_grid_by_biopsy_patient_{patient_tok}"
                    f"_bx_{bx_tok}_kernel_{kernel_tok}_foldsel_{fold_sel_token}"
                )
                try:
                    out_paths = gpr_pp._save_figure(
                        fig,
                        save_dir.joinpath(file_name_base),
                        formats=("pdf", "svg"),
                        dpi=400,
                        show=False,
                        create_subdir_for_stem=False,
                    )
                    profile_grid_saved.extend([str(p) for p in out_paths])
                except Exception as e:
                    plt.close(fig)
                    profile_grid_errors.append(
                        f"grid_by_biopsy patient={patient_id}, bx={bx_index}, kernel={kernel_label}: {e}"
                    )

        # Mode B: one fold across biopsies
        if config.plot_fold_ids is not None:
            fold_values = [int(f) for f in config.plot_fold_ids]
        else:
            fold_values = sorted(int(v) for v in pd.unique(plot_keys_df["fold_id"]))
        bx_order_map = {bx_key: i for i, bx_key in enumerate(bx_order)}
        for kernel_label in kernels:
            kernel_tok = _sanitize_token(kernel_label)
            for fold_id in fold_values:
                panel_df = plot_keys_df[
                    (plot_keys_df["fold_id"].astype(int) == int(fold_id))
                    & (plot_keys_df["kernel_label"] == kernel_label)
                ].copy()
                if panel_df.empty:
                    continue
                panel_df["__ord"] = panel_df.apply(
                    lambda r: bx_order_map.get((r["Patient ID"], int(r["Bx index"])), 10**9),
                    axis=1,
                )
                panel_df = panel_df.sort_values(["__ord", "Patient ID", "Bx index"]).drop(columns=["__ord"])
                if panel_df.empty:
                    continue
                n_rows, n_cols = gpr_pp._grid_shape(len(panel_df), int(config.plot_grid_ncols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(gpr_pp.GRID_PER_BIOPSY_FIGSIZE[0] * n_cols, gpr_pp.GRID_PER_BIOPSY_FIGSIZE[1] * n_rows))
                axes = np.atleast_1d(axes).ravel()
                shared_handles, shared_labels = None, None
                for ax, (_, prow) in zip(axes, panel_df.iterrows()):
                    patient_id = prow["Patient ID"]
                    bx_index = int(prow["Bx index"])
                    kernel_name = str(prow["kernel_name"])
                    kernel_param = prow["kernel_param"]
                    pred_group_df = pred_df[
                        (pred_df["Patient ID"] == patient_id)
                        & (pred_df["Bx index"] == bx_index)
                        & (pred_df["fold_id"] == int(fold_id))
                        & (pred_df["kernel_label"] == kernel_label)
                    ].copy()
                    payload = _build_blocked_cv_fold_plot_payload(
                        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
                        fold_map_df=fold_map_df,
                        fold_metrics_df=fold_metrics_df,
                        pred_group_df=pred_group_df,
                        patient_id=patient_id,
                        bx_index=bx_index,
                        fold_id=int(fold_id),
                        kernel_name=kernel_name,
                        kernel_param=kernel_param,
                        config=config,
                    )
                    if payload is None:
                        ax.text(0.5, 0.5, f"Missing {patient_id},{bx_index}", ha="center", va="center")
                        ax.axis("off")
                        continue
                    biopsy_label = _blocked_cv_biopsy_label(patient_id, bx_index, config.plot_grid_label_map)
                    title_label = f"{biopsy_label} | Fold {int(fold_id)}"
                    h, l = _draw_blocked_cv_profile_axis(
                        ax,
                        payload=payload,
                        kernel_label=kernel_label,
                        title_label=title_label,
                    )
                    if shared_handles is None and h:
                        shared_handles, shared_labels = h, l
                for ax in axes[len(panel_df):]:
                    ax.axis("off")
                if shared_handles:
                    fig.legend(
                        shared_handles,
                        shared_labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 1.07),
                        ncol=min(len(shared_handles), 6),
                        frameon=False,
                        fancybox=False,
                        fontsize=gpr_pp._fs_legend(),
                    )
                fig.tight_layout(rect=[0, 0, 1, 0.86])
                bx_sel_token = _selection_token(
                    [f"{_sanitize_token(pid)}_{int(bx)}" for pid, bx in bx_order] if bx_order else None,
                    none_token="allbx",
                )
                save_dir = (
                    figs_dir
                    .joinpath("report")
                    .joinpath("cohort")
                    .joinpath("grid_profiles")
                    .joinpath("by_fold_across_biopsies")
                    .joinpath(f"kernel_{kernel_tok}")
                )
                file_name_base = (
                    f"blocked_cv_profile_grid_by_fold_fold_{_sanitize_token(fold_id)}"
                    f"_kernel_{kernel_tok}_bxsel_{bx_sel_token}"
                )
                try:
                    out_paths = gpr_pp._save_figure(
                        fig,
                        save_dir.joinpath(file_name_base),
                        formats=("pdf", "svg"),
                        dpi=400,
                        show=False,
                        create_subdir_for_stem=False,
                    )
                    profile_grid_saved.extend([str(p) for p in out_paths])
                except Exception as e:
                    plt.close(fig)
                    profile_grid_errors.append(
                        f"grid_by_fold fold={fold_id}, kernel={kernel_label}: {e}"
                    )
        _plot_progress(
            f"profile grids: complete "
            f"(saved={len(profile_grid_saved)}, errors={len(profile_grid_errors)})"
        )
    else:
        _plot_progress("profile grids: skipped")

    if config.plot_make_semivariogram_grids and config.plot_write_report_figures and not plot_keys_df.empty:
        _plot_progress("semivariogram grids: starting")
        gpr_pp._setup_matplotlib_defaults()
        kernels = sorted(pd.unique(plot_keys_df["kernel_label"]))
        if config.plot_patient_bx_list is None:
            bx_order = [
                (r["Patient ID"], int(r["Bx index"]))
                for _, r in plot_keys_df[["Patient ID", "Bx index"]].drop_duplicates().sort_values(["Patient ID", "Bx index"]).iterrows()
            ]
        else:
            available = {
                (r["Patient ID"], int(r["Bx index"]))
                for _, r in plot_keys_df[["Patient ID", "Bx index"]].drop_duplicates().iterrows()
            }
            bx_order = []
            for pid, bx in config.plot_patient_bx_list:
                key = (pid, int(bx))
                if key in available:
                    bx_order.append(key)
            if not bx_order:
                bx_order = [
                    (r["Patient ID"], int(r["Bx index"]))
                    for _, r in plot_keys_df[["Patient ID", "Bx index"]].drop_duplicates().sort_values(["Patient ID", "Bx index"]).iterrows()
                ]

        # Mode A: one biopsy across folds
        for kernel_label in kernels:
            kernel_tok = _sanitize_token(kernel_label)
            for patient_id, bx_index in bx_order:
                panel_df = plot_keys_df[
                    (plot_keys_df["Patient ID"] == patient_id)
                    & (plot_keys_df["Bx index"] == bx_index)
                    & (plot_keys_df["kernel_label"] == kernel_label)
                ].copy()
                if panel_df.empty:
                    continue
                if config.plot_fold_sort_mode == "z_start_mm" and "test_z_min_mm" in panel_df.columns:
                    panel_df = panel_df.sort_values(["test_z_min_mm", "fold_id"])
                else:
                    panel_df = panel_df.sort_values(["fold_id"])

                n_rows, n_cols = gpr_pp._grid_shape(len(panel_df), int(config.plot_grid_ncols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(gpr_pp.GRID_PER_BIOPSY_FIGSIZE[0] * n_cols, gpr_pp.GRID_PER_BIOPSY_FIGSIZE[1] * n_rows))
                axes = np.atleast_1d(axes).ravel()
                shared_handles, shared_labels = None, None
                for ax, (_, prow) in zip(axes, panel_df.iterrows()):
                    fold_id = int(prow["fold_id"])
                    kernel_name = str(prow["kernel_name"])
                    kernel_param = prow["kernel_param"]
                    pred_group_df = pred_df[
                        (pred_df["Patient ID"] == patient_id)
                        & (pred_df["Bx index"] == bx_index)
                        & (pred_df["fold_id"] == fold_id)
                        & (pred_df["kernel_label"] == kernel_label)
                    ].copy()
                    payload = _build_blocked_cv_fold_plot_payload(
                        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
                        fold_map_df=fold_map_df,
                        fold_metrics_df=fold_metrics_df,
                        pred_group_df=pred_group_df,
                        patient_id=patient_id,
                        bx_index=bx_index,
                        fold_id=fold_id,
                        kernel_name=kernel_name,
                        kernel_param=kernel_param,
                        config=config,
                    )
                    if payload is None:
                        ax.text(0.5, 0.5, f"Missing fold {fold_id}", ha="center", va="center")
                        ax.axis("off")
                        continue
                    biopsy_label = _blocked_cv_biopsy_label(patient_id, bx_index, config.plot_grid_label_map)
                    title_label = f"{biopsy_label} | Fold {fold_id}"
                    h, l = _draw_blocked_cv_semivariogram_axis(
                        ax,
                        payload=payload,
                        title_label=title_label,
                        show_n_pairs=_show_semivariogram_n_pairs_grids(config),
                        n_pairs_fontsize=float(config.plot_semivariogram_n_pairs_fontsize),
                    )
                    if shared_handles is None and h:
                        shared_handles, shared_labels = h, l
                for ax in axes[len(panel_df):]:
                    ax.axis("off")
                if shared_handles:
                    fig.legend(
                        shared_handles,
                        shared_labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 1.07),
                        ncol=min(len(shared_handles), 6),
                        frameon=False,
                        fancybox=False,
                        fontsize=gpr_pp._fs_legend(),
                    )
                fig.tight_layout(rect=[0, 0, 1, 0.86])
                patient_tok = _sanitize_token(patient_id)
                bx_tok = _sanitize_token(bx_index)
                fold_sel_token = _selection_token(config.plot_fold_ids, none_token="allfolds")
                save_dir = (
                    figs_dir
                    .joinpath("report")
                    .joinpath("cohort")
                    .joinpath("grid_semivariograms")
                    .joinpath("by_biopsy_across_folds")
                    .joinpath(f"kernel_{kernel_tok}")
                )
                file_name_base = (
                    f"blocked_cv_semivariogram_grid_by_biopsy_patient_{patient_tok}"
                    f"_bx_{bx_tok}_kernel_{kernel_tok}_foldsel_{fold_sel_token}"
                )
                try:
                    out_paths = gpr_pp._save_figure(
                        fig,
                        save_dir.joinpath(file_name_base),
                        formats=("pdf", "svg"),
                        dpi=400,
                        show=False,
                        create_subdir_for_stem=False,
                    )
                    semivariogram_grid_saved.extend([str(p) for p in out_paths])
                except Exception as e:
                    plt.close(fig)
                    semivariogram_grid_errors.append(
                        f"semivar_grid_by_biopsy patient={patient_id}, bx={bx_index}, kernel={kernel_label}: {e}"
                    )

        # Mode B: one fold across biopsies
        if config.plot_fold_ids is not None:
            fold_values = [int(f) for f in config.plot_fold_ids]
        else:
            fold_values = sorted(int(v) for v in pd.unique(plot_keys_df["fold_id"]))
        bx_order_map = {bx_key: i for i, bx_key in enumerate(bx_order)}
        for kernel_label in kernels:
            kernel_tok = _sanitize_token(kernel_label)
            for fold_id in fold_values:
                panel_df = plot_keys_df[
                    (plot_keys_df["fold_id"].astype(int) == int(fold_id))
                    & (plot_keys_df["kernel_label"] == kernel_label)
                ].copy()
                if panel_df.empty:
                    continue
                panel_df["__ord"] = panel_df.apply(
                    lambda r: bx_order_map.get((r["Patient ID"], int(r["Bx index"])), 10**9),
                    axis=1,
                )
                panel_df = panel_df.sort_values(["__ord", "Patient ID", "Bx index"]).drop(columns=["__ord"])
                if panel_df.empty:
                    continue
                n_rows, n_cols = gpr_pp._grid_shape(len(panel_df), int(config.plot_grid_ncols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(gpr_pp.GRID_PER_BIOPSY_FIGSIZE[0] * n_cols, gpr_pp.GRID_PER_BIOPSY_FIGSIZE[1] * n_rows))
                axes = np.atleast_1d(axes).ravel()
                shared_handles, shared_labels = None, None
                for ax, (_, prow) in zip(axes, panel_df.iterrows()):
                    patient_id = prow["Patient ID"]
                    bx_index = int(prow["Bx index"])
                    kernel_name = str(prow["kernel_name"])
                    kernel_param = prow["kernel_param"]
                    pred_group_df = pred_df[
                        (pred_df["Patient ID"] == patient_id)
                        & (pred_df["Bx index"] == bx_index)
                        & (pred_df["fold_id"] == int(fold_id))
                        & (pred_df["kernel_label"] == kernel_label)
                    ].copy()
                    payload = _build_blocked_cv_fold_plot_payload(
                        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
                        fold_map_df=fold_map_df,
                        fold_metrics_df=fold_metrics_df,
                        pred_group_df=pred_group_df,
                        patient_id=patient_id,
                        bx_index=bx_index,
                        fold_id=int(fold_id),
                        kernel_name=kernel_name,
                        kernel_param=kernel_param,
                        config=config,
                    )
                    if payload is None:
                        ax.text(0.5, 0.5, f"Missing {patient_id},{bx_index}", ha="center", va="center")
                        ax.axis("off")
                        continue
                    biopsy_label = _blocked_cv_biopsy_label(patient_id, bx_index, config.plot_grid_label_map)
                    title_label = f"{biopsy_label} | Fold {int(fold_id)}"
                    h, l = _draw_blocked_cv_semivariogram_axis(
                        ax,
                        payload=payload,
                        title_label=title_label,
                        show_n_pairs=_show_semivariogram_n_pairs_grids(config),
                        n_pairs_fontsize=float(config.plot_semivariogram_n_pairs_fontsize),
                    )
                    if shared_handles is None and h:
                        shared_handles, shared_labels = h, l
                for ax in axes[len(panel_df):]:
                    ax.axis("off")
                if shared_handles:
                    fig.legend(
                        shared_handles,
                        shared_labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 1.07),
                        ncol=min(len(shared_handles), 6),
                        frameon=False,
                        fancybox=False,
                        fontsize=gpr_pp._fs_legend(),
                    )
                fig.tight_layout(rect=[0, 0, 1, 0.86])
                bx_sel_token = _selection_token(
                    [f"{_sanitize_token(pid)}_{int(bx)}" for pid, bx in bx_order] if bx_order else None,
                    none_token="allbx",
                )
                save_dir = (
                    figs_dir
                    .joinpath("report")
                    .joinpath("cohort")
                    .joinpath("grid_semivariograms")
                    .joinpath("by_fold_across_biopsies")
                    .joinpath(f"kernel_{kernel_tok}")
                )
                file_name_base = (
                    f"blocked_cv_semivariogram_grid_by_fold_fold_{_sanitize_token(fold_id)}"
                    f"_kernel_{kernel_tok}_bxsel_{bx_sel_token}"
                )
                try:
                    out_paths = gpr_pp._save_figure(
                        fig,
                        save_dir.joinpath(file_name_base),
                        formats=("pdf", "svg"),
                        dpi=400,
                        show=False,
                        create_subdir_for_stem=False,
                    )
                    semivariogram_grid_saved.extend([str(p) for p in out_paths])
                except Exception as e:
                    plt.close(fig)
                    semivariogram_grid_errors.append(
                        f"semivar_grid_by_fold fold={fold_id}, kernel={kernel_label}: {e}"
                    )
        _plot_progress(
            f"semivariogram grids: complete "
            f"(saved={len(semivariogram_grid_saved)}, errors={len(semivariogram_grid_errors)})"
        )
    else:
        _plot_progress("semivariogram grids: skipped")

    # Normalize distribution mode families for report hist/KDE figures.
    # Accepted forms:
    # - None -> default (("histogram", "kde"),)
    # - ("histogram", "kde") -> one combined family
    # - [("histogram",), ("histogram", "kde"), ("kde",)] -> multiple families
    # - "kde" -> one KDE-only family
    raw_mode_families = config.plot_report_distribution_modes_list
    if raw_mode_families is None:
        report_dist_modes_list = (("histogram", "kde"),)
    elif isinstance(raw_mode_families, str):
        report_dist_modes_list = ((raw_mode_families,),)
    else:
        mode_items = list(raw_mode_families)
        if mode_items and all(isinstance(item, str) for item in mode_items):
            report_dist_modes_list = (tuple(mode_items),)
        else:
            normalized_families = []
            for mode_item in mode_items:
                if isinstance(mode_item, str):
                    normalized_families.append((mode_item,))
                else:
                    mode_tuple = tuple(mode_item)
                    if mode_tuple:
                        normalized_families.append(mode_tuple)
            report_dist_modes_list = (
                tuple(normalized_families)
                if normalized_families
                else (("histogram", "kde"),)
            )

    do_report_calib = (
        config.plot_write_report_figures
        and (config.plot_make_report_calibration_scatter or config.plot_make_report_calibration_distributions)
    )
    if do_report_calib:
        _plot_progress("report calibration: starting")
        try:
            calib_df = _filter_report_biopsy_df(biopsy_metrics_df, use_variance_mode=True)
            if not calib_df.empty:
                calib_mode_label = str(config.plot_variance_mode)
                if calib_mode_label == "primary":
                    calib_mode_label = str(config.primary_predictive_variance_mode)
                calib_save_dir = (
                    figs_dir
                    .joinpath("report")
                    .joinpath("cohort")
                    .joinpath("metrics")
                    .joinpath("calibration")
                )
                kernel_suffix = None
                if "kernel_label" in calib_df.columns:
                    kernel_vals = pd.unique(calib_df["kernel_label"].dropna())
                    if len(kernel_vals) == 1:
                        kernel_suffix = str(kernel_vals[0])
                report_calibration_saved.extend(
                    gpr_pp.plot_blocked_cv_calibration_report(
                        biopsy_metrics_df=calib_df,
                        save_dir=calib_save_dir,
                        save_formats=("pdf", "svg"),
                        modes=report_dist_modes_list[0],
                        modes_list=report_dist_modes_list,
                        kde_bw_scale=config.plot_report_distribution_kde_bw_scale,
                        kernel_suffix=kernel_suffix,
                        rstd_label_mode=calib_mode_label,
                        make_histograms=bool(config.plot_make_report_calibration_distributions),
                        make_scatter=bool(config.plot_make_report_calibration_scatter),
                    )
                )
            else:
                _plot_progress("report calibration: no rows after kernel/variance filtering")
        except Exception as e:
            report_calibration_errors.append(f"{type(e).__name__}: {e!r}")
        _plot_progress(
            f"report calibration: complete "
            f"(saved={len(report_calibration_saved)}, errors={len(report_calibration_errors)})"
        )
    else:
        _plot_progress("report calibration: skipped")

    do_report_perf = config.plot_write_report_figures and config.plot_make_report_performance_distributions
    if do_report_perf:
        _plot_progress("report performance distributions: starting")
        try:
            perf_df = _filter_report_biopsy_df(biopsy_metrics_df, use_variance_mode=True)
            if not perf_df.empty:
                perf_save_dir = (
                    figs_dir
                    .joinpath("report")
                    .joinpath("cohort")
                    .joinpath("metrics")
                    .joinpath("performance")
                )
                kernel_suffix = None
                if "kernel_label" in perf_df.columns:
                    kernel_vals = pd.unique(perf_df["kernel_label"].dropna())
                    if len(kernel_vals) == 1:
                        kernel_suffix = str(kernel_vals[0])
                report_performance_saved.extend(
                    gpr_pp.plot_blocked_cv_performance_distributions(
                        biopsy_metrics_df=perf_df,
                        save_dir=perf_save_dir,
                        save_formats=("pdf", "svg"),
                        modes=report_dist_modes_list[0],
                        modes_list=report_dist_modes_list,
                        kde_bw_scale=config.plot_report_distribution_kde_bw_scale,
                        kernel_suffix=kernel_suffix,
                    )
                )
            else:
                _plot_progress("report performance distributions: no rows after kernel/variance filtering")
        except Exception as e:
            report_performance_errors.append(f"{type(e).__name__}: {e!r}")
        _plot_progress(
            f"report performance distributions: complete "
            f"(saved={len(report_performance_saved)}, errors={len(report_performance_errors)})"
        )
    else:
        _plot_progress("report performance distributions: skipped")

    do_report_varcmp = config.plot_write_report_figures and config.plot_make_report_variance_mode_comparison
    if do_report_varcmp:
        _plot_progress("report variance-mode comparison: starting")
        try:
            cmp_df = _filter_report_biopsy_df(compare_biopsy_metrics_df, use_variance_mode=False)
            if not cmp_df.empty:
                varcmp_save_dir = (
                    figs_dir
                    .joinpath("report")
                    .joinpath("cohort")
                    .joinpath("metrics")
                    .joinpath("variance_compare")
                )
                primary_mode = str(config.primary_predictive_variance_mode)
                if primary_mode != "latent":
                    latent_mode = "latent"
                    observed_mode = primary_mode
                else:
                    compare_modes = [str(m) for m in (config.variance_modes_to_compare or [])]
                    observed_mode = next((m for m in compare_modes if m != "latent"), "observed_mc")
                    latent_mode = "latent"
                kernel_suffix = None
                if "kernel_label" in cmp_df.columns:
                    kernel_vals = pd.unique(cmp_df["kernel_label"].dropna())
                    if len(kernel_vals) == 1:
                        kernel_suffix = str(kernel_vals[0])
                variance_mode_families = list(report_dist_modes_list)
                for family_idx, dist_mode_family in enumerate(variance_mode_families):
                    report_variance_compare_saved.extend(
                        gpr_pp.plot_blocked_cv_variance_mode_comparison(
                            compare_biopsy_metrics_df=cmp_df,
                            save_dir=varcmp_save_dir,
                            latent_mode=latent_mode,
                            observed_mode=observed_mode,
                            save_formats=("pdf", "svg"),
                            make_scatter=(family_idx == 0),
                            make_delta_distributions=True,
                            delta_modes=dist_mode_family,
                            delta_kde_bw_scale=config.plot_report_distribution_kde_bw_scale,
                            kernel_suffix=kernel_suffix,
                        )
                    )
            else:
                _plot_progress("report variance-mode comparison: compare table empty after kernel filtering")
        except Exception as e:
            report_variance_compare_errors.append(f"{type(e).__name__}: {e!r}")
        _plot_progress(
            f"report variance-mode comparison: complete "
            f"(saved={len(report_variance_compare_saved)}, errors={len(report_variance_compare_errors)})"
        )
    else:
        _plot_progress("report variance-mode comparison: skipped")

    unimplemented = []
    if config.plot_write_diagnostic_figures:
        unimplemented.append("diagnostic_figure_lane")

    any_errors = bool(
        paired_errors
        or profile_grid_errors
        or semivariogram_grid_errors
        or report_calibration_errors
        or report_performance_errors
        or report_variance_compare_errors
    )
    return {
        "phase": "blocked_cv_plots",
        "status": "ready" if not any_errors else "partial",
        "figs_dir": str(figs_dir),
        "n_point_prediction_rows_available": n_pred,
        "n_paired_candidates": n_candidates,
        "n_paired_selected": n_selected,
        "n_paired_saved_files": int(len(paired_saved)),
        "n_paired_errors": int(len(paired_errors)),
        "paired_example_paths": paired_saved[:4],
        "paired_error_examples": paired_errors[:4],
        "n_profile_grid_saved_files": int(len(profile_grid_saved)),
        "n_profile_grid_errors": int(len(profile_grid_errors)),
        "profile_grid_example_paths": profile_grid_saved[:4],
        "profile_grid_error_examples": profile_grid_errors[:4],
        "n_semivariogram_grid_saved_files": int(len(semivariogram_grid_saved)),
        "n_semivariogram_grid_errors": int(len(semivariogram_grid_errors)),
        "semivariogram_grid_example_paths": semivariogram_grid_saved[:4],
        "semivariogram_grid_error_examples": semivariogram_grid_errors[:4],
        "n_report_calibration_saved_files": int(len(report_calibration_saved)),
        "n_report_calibration_errors": int(len(report_calibration_errors)),
        "report_calibration_example_paths": report_calibration_saved[:4],
        "report_calibration_error_examples": report_calibration_errors[:4],
        "n_report_performance_saved_files": int(len(report_performance_saved)),
        "n_report_performance_errors": int(len(report_performance_errors)),
        "report_performance_example_paths": report_performance_saved[:4],
        "report_performance_error_examples": report_performance_errors[:4],
        "n_report_variance_compare_saved_files": int(len(report_variance_compare_saved)),
        "n_report_variance_compare_errors": int(len(report_variance_compare_errors)),
        "report_variance_compare_example_paths": report_variance_compare_saved[:4],
        "report_variance_compare_error_examples": report_variance_compare_errors[:4],
        "unimplemented_plot_families": unimplemented,
        "plot_patient_bx_list": (list(config.plot_patient_bx_list) if config.plot_patient_bx_list is not None else None),
        "plot_grid_ncols": int(config.plot_grid_ncols),
        "plot_grid_label_map_size": (len(config.plot_grid_label_map) if config.plot_grid_label_map is not None else 0),
        "plot_fold_ids": (list(config.plot_fold_ids) if config.plot_fold_ids is not None else None),
        "plot_max_folds_per_biopsy": (int(config.plot_max_folds_per_biopsy) if config.plot_max_folds_per_biopsy is not None else None),
        "plot_fold_sort_mode": config.plot_fold_sort_mode,
        "plot_include_merged_tail_folds": bool(config.plot_include_merged_tail_folds),
        "plot_include_rebalanced_two_fold_splits": bool(config.plot_include_rebalanced_two_fold_splits),
        "plot_kernel_labels": (list(config.plot_kernel_labels) if config.plot_kernel_labels is not None else None),
        "plot_variance_mode": config.plot_variance_mode,
        "plot_make_paired_semivariogram_profile": bool(config.plot_make_paired_semivariogram_profile),
        "plot_make_semivariogram_grids": bool(config.plot_make_semivariogram_grids),
        "plot_make_profile_grids": bool(config.plot_make_profile_grids),
        "plot_semivariogram_show_n_pairs_paired": _show_semivariogram_n_pairs_paired(config),
        "plot_semivariogram_show_n_pairs_grids": _show_semivariogram_n_pairs_grids(config),
        "plot_semivariogram_n_pairs_fontsize": float(config.plot_semivariogram_n_pairs_fontsize),
        "plot_make_report_calibration_scatter": bool(config.plot_make_report_calibration_scatter),
        "plot_make_report_calibration_distributions": bool(config.plot_make_report_calibration_distributions),
        "plot_make_report_performance_distributions": bool(config.plot_make_report_performance_distributions),
        "plot_make_report_variance_mode_comparison": bool(config.plot_make_report_variance_mode_comparison),
        "plot_report_distribution_modes_list": [list(m) for m in report_dist_modes_list],
        "plot_report_distribution_kde_bw_scale": (
            None if config.plot_report_distribution_kde_bw_scale is None else float(config.plot_report_distribution_kde_bw_scale)
        ),
        "n_biopsy_metrics_rows_available": int(len(biopsy_metrics_df)) if isinstance(biopsy_metrics_df, pd.DataFrame) else 0,
        "n_compare_biopsy_metrics_rows_available": int(len(compare_biopsy_metrics_df)) if isinstance(compare_biopsy_metrics_df, pd.DataFrame) else 0,
        "plot_write_report_figures": bool(config.plot_write_report_figures),
        "plot_write_diagnostic_figures": bool(config.plot_write_diagnostic_figures),
    }
