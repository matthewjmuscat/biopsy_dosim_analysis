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
    min_block_mm: float = 5.0
    position_mode: str = "begin"
    target_stat: str = "median"
    mean_mode: str = "ordinary"
    predictive_variance_mode: str = "observed_mc"
    kernel_specs: Iterable[Tuple[str, float | None, str]] | None = None
    semivariogram_voxel_size_mm: float = 1.0
    semivariogram_lag_bin_width_mm: float | None = None
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
    Phase 3C smoke path:
    - uses fold map from 3B logic,
    - runs strict train-only fit/predict for a single kernel,
    - writes point-level held-out prediction CSV + fold status CSV.
    """
    del semivariogram_df  # phase 3C uses per-fold train-only semivariograms
    fold_map_df, fold_summary_df = build_blocked_cv_fold_map(
        all_voxel_wise_dose_df,
        config=config,
    )
    fold_map_path = csv_dir.joinpath("blocked_cv_fold_map.csv")
    fold_summary_path = csv_dir.joinpath("blocked_cv_fold_summary.csv")
    if not fold_map_path.exists():
        fold_map_df.to_csv(fold_map_path, index=False)
    if not fold_summary_path.exists():
        fold_summary_df.to_csv(fold_summary_path, index=False)

    kernel_specs = list(config.kernel_specs) if config.kernel_specs is not None else _default_kernel_specs()
    if not kernel_specs:
        raise ValueError("blocked_CV Phase 3C needs at least one kernel spec.")
    kernel_name, kernel_param, kernel_label = kernel_specs[0]

    pred_rows = []
    fold_status_rows = []

    grp_cols = ["Patient ID", "Bx index", "fold_id"]
    for (patient_id, bx_index, fold_id), fold_rows in fold_map_df.groupby(grp_cols):
        g_bx = all_voxel_wise_dose_df[
            (all_voxel_wise_dose_df["Patient ID"] == patient_id)
            & (all_voxel_wise_dose_df["Bx index"] == bx_index)
        ].copy()
        test_voxels = set(
            fold_rows.loc[fold_rows["is_test"], "Voxel index"].astype(int).tolist()
        )
        if not test_voxels:
            fold_status_rows.append(
                {
                    "Patient ID": patient_id,
                    "Bx index": bx_index,
                    "fold_id": int(fold_id),
                    "kernel_label": kernel_label,
                    "status": "skipped",
                    "message": "no test voxels",
                }
            )
            continue
        train_df = g_bx[~g_bx["Voxel index"].isin(test_voxels)].copy()
        test_df = g_bx[g_bx["Voxel index"].isin(test_voxels)].copy()

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
            if len(X_train) < 3 or len(X_test) < 1:
                fold_status_rows.append(
                    {
                        "Patient ID": patient_id,
                        "Bx index": bx_index,
                        "fold_id": int(fold_id),
                        "kernel_label": kernel_label,
                        "status": "skipped",
                        "message": f"insufficient voxels (train={len(X_train)}, test={len(X_test)})",
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
                        "Bx index": bx_index,
                        "fold_id": int(fold_id),
                        "kernel_label": kernel_label,
                        "status": "skipped",
                        "message": f"insufficient semivariogram bins (n={len(sv_train)})",
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
            pred_var = _predictive_variance_for_mode(
                latent_sd=sd_test,
                obs_var=var_n_test,
                nugget=float(getattr(hyp, "nugget", 0.0)),
                mode=config.predictive_variance_mode,
            )
            pred_sd = np.sqrt(np.maximum(pred_var, 1e-12))
            resid = y_test - mu_test
            rstd = resid / pred_sd

            pv_test = pv_test.sort_values("x_mm").reset_index(drop=True)
            for i in range(len(X_test)):
                pred_rows.append(
                    {
                        "Patient ID": patient_id,
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
                        "var_pred_used": float(pred_var[i]),
                        "sd_pred_used": float(pred_sd[i]),
                        "residual": float(resid[i]),
                        "rstd": float(rstd[i]),
                        "gp_mean_mode": config.mean_mode,
                        "target_stat": config.target_stat,
                        "predictive_variance_mode": config.predictive_variance_mode,
                        "n_train_voxels": int(len(X_train)),
                        "n_test_voxels": int(len(X_test)),
                        "ell": float(getattr(hyp, "ell", np.nan)),
                        "sigma_f2": float(getattr(hyp, "sigma_f2", np.nan)),
                        "nugget": float(getattr(hyp, "nugget", np.nan)),
                        "nu": float(getattr(hyp, "nu", np.nan)),
                    }
                )

            fold_status_rows.append(
                {
                    "Patient ID": patient_id,
                    "Bx index": bx_index,
                    "fold_id": int(fold_id),
                    "kernel_label": kernel_label,
                    "status": "ok",
                    "message": "",
                    "n_train_voxels": int(len(X_train)),
                    "n_test_voxels": int(len(X_test)),
                }
            )

        except Exception as e:  # keep smoke run resilient and auditable
            fold_status_rows.append(
                {
                    "Patient ID": patient_id,
                    "Bx index": bx_index,
                    "fold_id": int(fold_id),
                    "kernel_label": kernel_label,
                    "status": "error",
                    "message": str(e),
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    fold_status_df = pd.DataFrame(fold_status_rows)
    pred_path = csv_dir.joinpath("blocked_cv_point_predictions_smoke_all.csv")
    status_path = csv_dir.joinpath("blocked_cv_fold_fit_status_smoke_all.csv")
    pred_df.to_csv(pred_path, index=False)
    fold_status_df.to_csv(status_path, index=False)

    status = {
        "phase": "3C_smoke_fit_predict",
        "status": "ready",
        "blocked_cv_root": str(output_dir),
        "blocked_cv_figs_dir": str(figs_dir),
        "blocked_cv_csv_dir": str(csv_dir),
        "kernel_label": kernel_label,
        "point_predictions_csv": str(pred_path),
        "fold_fit_status_csv": str(status_path),
        "n_point_prediction_rows": int(len(pred_df)),
        "n_fold_status_rows": int(len(fold_status_df)),
        "n_fold_status_ok": int((fold_status_df.get("status", pd.Series(dtype=str)) == "ok").sum()) if len(fold_status_df) else 0,
        "n_fold_status_error": int((fold_status_df.get("status", pd.Series(dtype=str)) == "error").sum()) if len(fold_status_df) else 0,
    }
    return status
