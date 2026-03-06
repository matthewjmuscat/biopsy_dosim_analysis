from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd


def _voxel_position_map(
    df: pd.DataFrame,
    *,
    position_mode: Literal["begin", "center"] = "begin",
) -> pd.Series:
    """Return per-voxel axial position (mm), indexed by voxel index."""
    if {"Voxel begin (Z)", "Voxel end (Z)"}.issubset(df.columns):
        pos = df.groupby("Voxel index").agg(
            x_mm_begin=("Voxel begin (Z)", "first"),
            x_mm_end=("Voxel end (Z)", "first"),
        )
        if position_mode == "center":
            x_mm = 0.5 * (pos["x_mm_begin"] + pos["x_mm_end"])
        else:
            x_mm = pos["x_mm_begin"]
        return x_mm.astype(float)

    # Fallback when begin/end bounds are unavailable
    z_col = "Z (Bx frame)"
    agg_fn = "mean" if position_mode == "center" else "min"
    pos = df.groupby("Voxel index").agg(x_mm=(z_col, agg_fn))
    return pos["x_mm"].astype(float)


def compute_semivariogram_shift(
    df: pd.DataFrame,
    voxel_size_mm: float = 1.0,
    max_lag_voxels: int | None = None,
) -> pd.DataFrame:
    """
    Legacy shift-based empirical semivariogram.

    Assumes contiguous voxel columns and integer lag steps.
    """
    M = df.pivot_table(index="MC trial num", columns="Voxel index", values="Dose (Gy)", aggfunc="first")
    M = M.sort_index(axis=1)
    D = M.values
    T, N = D.shape
    if N < 2:
        return pd.DataFrame(columns=["lag_voxels", "h_mm", "semivariance", "n_pairs"])
    if max_lag_voxels is None:
        max_lag_voxels = N - 1

    lags = np.arange(1, max_lag_voxels + 1)
    gamma = np.empty_like(lags, dtype=float)
    npairs = np.empty_like(lags, dtype=int)
    for idx, L in enumerate(lags):
        diffs = D[:, L:] - D[:, :-L]
        gamma[idx] = 0.5 * np.mean(diffs**2)
        npairs[idx] = T * (N - L)

    return pd.DataFrame(
        {
            "lag_voxels": lags,
            "h_mm": lags * voxel_size_mm,
            "semivariance": gamma,
            "n_pairs": npairs,
        }
    )


def compute_semivariogram_pairwise(
    df: pd.DataFrame,
    voxel_size_mm: float = 1.0,
    max_lag_voxels: int | None = None,
    *,
    position_mode: Literal["begin", "center"] = "begin",
    lag_bin_width_mm: float | None = None,
) -> pd.DataFrame:
    """
    Gap-safe empirical semivariogram using pairwise voxel distances.

    Voxels are binned by physical lag around centers h = L * voxel_size_mm.
    This remains valid when interior voxel segments are missing.
    """
    M = df.pivot_table(index="MC trial num", columns="Voxel index", values="Dose (Gy)", aggfunc="first")
    M = M.sort_index(axis=1)
    D = M.values
    voxel_ids = M.columns.to_numpy()
    T, N = D.shape
    if N < 2:
        return pd.DataFrame(columns=["lag_voxels", "h_mm", "semivariance", "n_pairs"])

    pos_map = _voxel_position_map(df, position_mode=position_mode)
    x = pos_map.reindex(voxel_ids).to_numpy(dtype=float)

    if lag_bin_width_mm is None:
        lag_bin_width_mm = float(voxel_size_mm)
    half_width = 0.5 * float(lag_bin_width_mm)

    ii, jj = np.triu_indices(N, k=1)
    dists = np.abs(x[jj] - x[ii])

    if max_lag_voxels is None:
        max_h = float(np.nanmax(dists)) if dists.size else 0.0
        max_lag_voxels = int(np.floor(max_h / voxel_size_mm))
        max_lag_voxels = max(max_lag_voxels, 1)

    lags = np.arange(1, max_lag_voxels + 1)
    gamma = np.full(lags.shape, np.nan, dtype=float)
    npairs = np.zeros(lags.shape, dtype=int)

    for idx, L in enumerate(lags):
        h_center = float(L * voxel_size_mm)
        msk = np.isfinite(dists) & (np.abs(dists - h_center) <= half_width + 1e-12)
        if not np.any(msk):
            continue
        i_sel = ii[msk]
        j_sel = jj[msk]
        diffs = D[:, j_sel] - D[:, i_sel]
        gamma[idx] = 0.5 * np.mean(diffs**2)
        npairs[idx] = int(T * i_sel.size)

    return pd.DataFrame(
        {
            "lag_voxels": lags,
            "h_mm": lags * voxel_size_mm,
            "semivariance": gamma,
            "n_pairs": npairs,
        }
    )


def compare_semivariogram_methods_by_biopsy(
    all_df: pd.DataFrame,
    *,
    voxel_size_mm: float = 1.0,
    max_lag_voxels: int | None = None,
    position_mode: Literal["begin", "center"] = "begin",
    lag_bin_width_mm: float | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare shift-based vs pairwise semivariograms per biopsy.

    Returns
    -------
    summary_df:
        One row per biopsy with aggregate agreement stats.
    detail_df:
        Per-lag comparison with absolute differences.
    """
    summary_rows = []
    detail_rows = []

    for (patient_id, bx_index), g in all_df.groupby(["Patient ID", "Bx index"]):
        sv_shift = compute_semivariogram_shift(
            g, voxel_size_mm=voxel_size_mm, max_lag_voxels=max_lag_voxels
        )
        sv_pair = compute_semivariogram_pairwise(
            g,
            voxel_size_mm=voxel_size_mm,
            max_lag_voxels=max_lag_voxels,
            position_mode=position_mode,
            lag_bin_width_mm=lag_bin_width_mm,
        )
        merged = sv_shift.merge(
            sv_pair,
            on=["lag_voxels", "h_mm"],
            how="outer",
            suffixes=("_shift", "_pairwise"),
        )
        merged["Patient ID"] = patient_id
        merged["Bx index"] = bx_index
        merged["abs_diff_semivariance"] = np.abs(
            merged["semivariance_shift"] - merged["semivariance_pairwise"]
        )
        merged["abs_diff_n_pairs"] = np.abs(
            merged["n_pairs_shift"].fillna(0) - merged["n_pairs_pairwise"].fillna(0)
        )
        detail_rows.append(merged)

        valid = merged["abs_diff_semivariance"].dropna()
        summary_rows.append(
            {
                "Patient ID": patient_id,
                "Bx index": bx_index,
                "n_lags_compared": int(valid.size),
                "max_abs_diff_semivariance": float(valid.max()) if valid.size else np.nan,
                "mean_abs_diff_semivariance": float(valid.mean()) if valid.size else np.nan,
                "median_abs_diff_semivariance": float(valid.median()) if valid.size else np.nan,
                "max_abs_diff_n_pairs": float(merged["abs_diff_n_pairs"].max(skipna=True)),
                "mean_abs_diff_n_pairs": float(merged["abs_diff_n_pairs"].mean(skipna=True)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("max_abs_diff_semivariance", ascending=False)

    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return summary_df, detail_df

