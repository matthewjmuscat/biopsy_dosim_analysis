from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from typing import Sequence, Literal


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


import numpy as np
import pandas as pd

# existing helper
# def _flatten_columns(...):


def _robust_group_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Group df by group_cols and compute robust summary stats for value_cols:
    count, mean, std, median, q05, q25, q75, q95, IQR, IPR90.

    Returns a DataFrame with one row per group and flattened column names like
    'Dose (Gy) mean', 'Dose (Gy) q25', etc.
    """
    df = df.copy()

    group = df.groupby(group_cols, dropna=False)

    # Define named helper functions so their __name__ becomes the column label.
    def q05(x):
        return np.nanpercentile(x, 5)
    q05.__name__ = "q05"

    def q25(x):
        return np.nanpercentile(x, 25)
    q25.__name__ = "q25"

    def q75(x):
        return np.nanpercentile(x, 75)
    q75.__name__ = "q75"

    def q95(x):
        return np.nanpercentile(x, 95)
    q95.__name__ = "q95"

    def IQR(x):
        return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    IQR.__name__ = "IQR"

    def IPR90(x):
        return np.nanpercentile(x, 95) - np.nanpercentile(x, 5)
    IPR90.__name__ = "IPR90"

    agg_funcs = ["count", "mean", "std", "median", q05, q25, q75, q95, IQR, IPR90]

    # This yields a MultiIndex on columns: (value_col, agg_func_name)
    out = group[value_cols].agg(agg_funcs)

    # Flatten MultiIndex columns: ('Dose (Gy)', 'mean') -> 'Dose (Gy) mean'
    out.columns = [
        f"{metric} {stat}".strip()
        for metric, stat in out.columns.to_flat_index()
    ]

    out = out.reset_index()

    return out


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten 2-level column structures or tuple-valued columns into single strings.

    ('Dose (Gy)', 'mean')  -> 'Dose (Gy) mean'
    ('Patient ID', '')     -> 'Patient ID'

    If columns are already simple strings and not tuples, the DataFrame is
    returned unchanged (aside from a shallow copy).
    """
    cols = df.columns

    # If we have neither a MultiIndex nor any tuple columns, just return a copy.
    if not isinstance(cols, pd.MultiIndex) and not any(
        isinstance(c, tuple) for c in cols
    ):
        return df.copy()

    new_cols: list[str] = []
    for col in cols:
        # True MultiIndex element or tuple sitting in a plain Index
        if isinstance(col, tuple):
            top = str(col[0]) if len(col) > 0 else ""
            sub = str(col[1]) if len(col) > 1 else ""
            if sub in ("", "None"):
                new_cols.append(top)
            else:
                new_cols.append(f"{top} {sub}")
        else:
            # already a simple label
            new_cols.append(str(col))

    out = df.copy()
    out.columns = new_cols
    return out




# -------------------------------------------------------------------------
# Correlation summaries
# -------------------------------------------------------------------------

def compute_delta_vs_predictor_correlations(
    design_df: pd.DataFrame,
    delta_col: str = "log1p|Delta|",
    key_cols: Sequence[str] | None = None,
    group_col: str = "Delta kind",
    min_n: int = 20,
    exclude_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compute (Pearson) correlations between a chosen Δ column and every
    numeric predictor, stratified by delta kind (or any grouping column).

    Only columns with a numeric dtype (np.number) are considered as
    candidate predictors.

    Parameters
    ----------
    design_df
        Long-format design matrix (e.g. from
        `make_long_delta_design_from_delta_design`).
    delta_col
        Column containing the Δ quantity to use in the correlation
        (typically '|Delta| (Gy)', '|Delta|', or 'log1p|Delta|').
    key_cols
        Identifier / bookkeeping columns that should be excluded from
        the predictor list (e.g. 'Patient ID', 'Bx index', 'Voxel index').
        They are *not* used for grouping; they are only auto-excluded
        from the predictors. If None, sensible defaults are used.
    group_col
        Column used to stratify correlations (usually 'Delta kind').
    min_n
        Minimum number of non-missing (x, y) pairs required for a
        predictor in a given group; predictors with fewer pairs are
        skipped for that group.
    exclude_cols
        Additional columns to exclude from the predictor list, on top of
        `key_cols`, `group_col`, and the standard delta columns.

    Returns
    -------
    corr_df : pd.DataFrame
        One row per (group_col, predictor) with:
          * group_col – the group label (e.g. Δ_mode / Δ_median / Δ_mean)
          * predictor – predictor column name
          * n         – number of paired observations
          * r         – Pearson correlation coefficient
          * p_value   – two-sided p-value
        The rows are sorted within each group by |r| (largest first).
    """
    # Default key columns (ID-like columns to auto-exclude)
    if key_cols is None:
        key_cols = (
            "Patient ID",
            "Bx ID",
            "Bx index",
            "Voxel index",
        )

    # Extra explicit exclusions
    if exclude_cols is None:
        exclude_cols = ()

    # Base set of columns that should never be treated as predictors
    base_exclude = {
        group_col,
        "Delta (Gy)",      # signed Δ
        "|Delta| (Gy)",    # magnitude with units in name
        "|Delta|",         # magnitude without units in name
        "log1p|Delta|",    # transformed magnitude
    }

    exclude = set(key_cols) | set(exclude_cols) | base_exclude

    # Only numeric predictors that are not in the exclusion set
    numeric_predictors: list[str] = [
        c
        for c in design_df.columns
        if c not in exclude and np.issubdtype(design_df[c].dtype, np.number)
    ]

    rows: list[dict] = []
    groups = design_df[group_col].unique()

    for g in groups:
        df_g = design_df[design_df[group_col] == g]

        # y: chosen delta column
        y = df_g[delta_col].to_numpy()
        mask_y = np.isfinite(y)

        for pred in numeric_predictors:
            x = df_g[pred].to_numpy()
            mask_x = np.isfinite(x)
            mask = mask_x & mask_y

            n = int(mask.sum())
            if n < min_n:
                continue

            r, p = stats.pearsonr(x[mask], y[mask])
            rows.append(
                {
                    group_col: g,
                    "predictor": pred,
                    "n": n,
                    "r": r,
                    "p_value": p,
                }
            )

    corr_df = pd.DataFrame(rows)
    if not corr_df.empty:
        corr_df["abs_r"] = corr_df["r"].abs()
        corr_df = (
            corr_df.sort_values(
                by=[group_col, "abs_r"],
                ascending=[True, False],
            )
            .drop(columns="abs_r")
        )

    return corr_df






def compute_delta_vs_predictor_correlation_generalized(
    design_df: pd.DataFrame,
    delta_col: str = "log1p|Delta|",
    key_cols: Sequence[str] | None = None,
    group_col: str = "Delta kind",
    min_n: int = 20,
    exclude_cols: Sequence[str] | None = None,
    # NEW: choose how to rank/sort results (both r and rho are always computed)
    rank_by: Literal["pearson", "spearman", "max_abs"] | None = "spearman",
) -> pd.DataFrame:
    """
    Compute correlations between delta_col and every numeric predictor,
    stratified by group_col.

    Always computes:
      - Pearson r (and p-value)
      - Spearman rho (and p-value)

    Ranking/sorting is controlled by rank_by:
      - "pearson": sort by abs(Pearson r)
      - "spearman": sort by abs(Spearman rho)
      - "max_abs": sort by max(abs(r), abs(rho))
    """
    if key_cols is None:
        key_cols = ("Patient ID", "Bx ID", "Bx index", "Voxel index")

    if exclude_cols is None:
        exclude_cols = ()


    if rank_by is None:
        rank_by = "spearman"

    rank_by = str(rank_by).lower()
    if rank_by not in {"pearson", "spearman", "max_abs"}:
        raise ValueError("rank_by must be one of: 'pearson', 'spearman', 'max_abs'.")

    base_exclude = {
        group_col,
        "Delta (Gy)",
        "|Delta| (Gy)",
        "|Delta|",
        "log1p|Delta|",
    }
    exclude = set(key_cols) | set(exclude_cols) | base_exclude

    numeric_predictors: list[str] = [
        c
        for c in design_df.columns
        if c not in exclude and np.issubdtype(design_df[c].dtype, np.number)
    ]

    rows: list[dict] = []
    groups = pd.unique(design_df[group_col])

    for g in groups:
        df_g = design_df.loc[design_df[group_col] == g]

        y = df_g[delta_col].to_numpy()
        mask_y = np.isfinite(y)

        for pred in numeric_predictors:
            x = df_g[pred].to_numpy()
            mask_x = np.isfinite(x)
            mask = mask_x & mask_y

            n = int(mask.sum())
            if n < min_n:
                continue

            # Pearson
            r_pearson, p_pearson = stats.pearsonr(x[mask], y[mask])

            # Spearman
            rho_spearman, p_spearman = stats.spearmanr(x[mask], y[mask])

            rows.append(
                {
                    group_col: g,
                    "predictor": pred,
                    "n": n,
                    "r_pearson": r_pearson,
                    "p_pearson": p_pearson,
                    "rho_spearman": rho_spearman,
                    "p_spearman": p_spearman,
                }
            )

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df

    # Build ranking metric
    if rank_by == "pearson":
        corr_df["_rank_abs"] = corr_df["r_pearson"].abs()
        corr_df["ranked_by"] = "pearson"
    elif rank_by == "spearman":
        corr_df["_rank_abs"] = corr_df["rho_spearman"].abs()
        corr_df["ranked_by"] = "spearman"
    else:  # "max_abs"
        corr_df["_rank_abs"] = np.maximum(
            corr_df["r_pearson"].abs(),
            corr_df["rho_spearman"].abs(),
        )
        corr_df["ranked_by"] = "max_abs"

    corr_df = (
        corr_df.sort_values(by=[group_col, "_rank_abs"], ascending=[True, False])
        .drop(columns=["_rank_abs"])
    )

    return corr_df






def compute_interdelta_correlations(
    wide_deltas_df: pd.DataFrame,
    delta_cols: Sequence[str] = (
        "Δ_mode (Gy)",
        "Δ_median (Gy)",
        "Δ_mean (Gy)",
    ),
) -> pd.DataFrame:
    """
    Compute Pearson correlations between the different voxel-level Δ
    definitions (Δ_mode, Δ_median, Δ_mean), using one row per
    (Patient ID, Bx index, Voxel index).

    Parameters
    ----------
    wide_deltas_df
        Wide-format dataframe `combined_wide_deltas_vs_gradient` from
        `build_deltas_vs_gradient_df_with_abs`.
    delta_cols
        Names of the Δ columns to correlate.

    Returns
    -------
    corr_df
        One row per pair (delta_a, delta_b) with:
          * n       – number of paired observations
          * r       – Pearson correlation coefficient
          * p_value – two-sided p-value
    """
    df = wide_deltas_df.loc[:, list(delta_cols)].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    rows: list[dict] = []
    for i, a in enumerate(delta_cols):
        for b in delta_cols[i + 1 :]:
            x = df[a].to_numpy()
            y = df[b].to_numpy()
            n = len(df)
            if n == 0:
                continue

            r, p = stats.pearsonr(x, y)
            rows.append(
                {
                    "delta_a": a,
                    "delta_b": b,
                    "n": n,
                    "r": r,
                    "p_value": p,
                }
            )

    corr_df = pd.DataFrame(rows)
    if not corr_df.empty:
        corr_df = corr_df.sort_values(
            by="r", key=lambda s: s.abs(), ascending=False
        )
    return corr_df




def compute_interdelta_correlations_generalized(
    wide_deltas_df: pd.DataFrame,
    delta_cols: Sequence[str] = (
        "Δ_mode (Gy)",
        "Δ_median (Gy)",
        "Δ_mean (Gy)",
    ),
) -> pd.DataFrame:
    """
    Compute correlations between different voxel-level Δ definitions
    (e.g. Δ_mode, Δ_median, Δ_mean), using one row per pair
    (delta_a, delta_b).

    For each pair it returns:
        * n            – number of paired observations
        * r_pearson    – Pearson correlation coefficient
        * p_pearson    – two-sided p-value for Pearson r
        * rho_spearman – Spearman rank correlation coefficient
        * p_spearman   – two-sided p-value for Spearman rho
    """
    # Work only with the requested columns; clean infs to NaN
    df = wide_deltas_df.loc[:, list(delta_cols)].replace(
        [np.inf, -np.inf], np.nan
    )

    rows: list[dict] = []

    for i, a in enumerate(delta_cols):
        for b in delta_cols[i + 1:]:
            pair = df[[a, b]].dropna(how="any")
            n = len(pair)
            if n < 2:
                # Not enough data points to define a correlation
                continue

            x = pair[a].to_numpy()
            y = pair[b].to_numpy()

            # Default to NaN, then fill if we can compute
            r_pearson = p_pearson = np.nan
            rho_spearman = p_spearman = np.nan

            # Guard against constant inputs which make correlations undefined
            if np.unique(x).size > 1 and np.unique(y).size > 1:
                try:
                    r_pearson, p_pearson = stats.pearsonr(x, y)
                except Exception:
                    pass

                try:
                    rho_spearman, p_spearman = stats.spearmanr(x, y)
                except Exception:
                    pass

            rows.append(
                {
                    "delta_a": a,
                    "delta_b": b,
                    "n": n,
                    "r_pearson": r_pearson,
                    "p_pearson": p_pearson,
                    "rho_spearman": rho_spearman,
                    "p_spearman": p_spearman,
                }
            )

    corr_df = pd.DataFrame(rows)

    if not corr_df.empty:
        # Sort by absolute Pearson r (you can switch to rho if you prefer)
        corr_df = corr_df.sort_values(
            by="r_pearson", key=lambda s: s.abs(), ascending=False
        )

    return corr_df





def make_long_delta_design_from_delta_design(
    delta_design_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Take the wide delta_design_df produced by
    helper_funcs.build_deltas_with_spatial_radiomics_and_distances and
    reshape it into a long design matrix suitable for correlation analysis.

    One row per (voxel, Δ kind):

      * Delta kind  ∈ { 'Δ_mode', 'Δ_median', 'Δ_mean' }
      * Delta (Gy)  – signed nominal minus aggregate
      * |Delta|     – absolute value of the above
      * log1p|Delta| – log(1 + |Delta|)

    All spatial / radiomics / distance / nominal predictors are carried along.
    """

    # Column names in delta_design_df
    signed_cols = {
        "Δ_mode": "Dose (Gy) deltas nominal_minus_mode",
        "Δ_median": "Dose (Gy) deltas nominal_minus_q50",
        "Δ_mean": "Dose (Gy) deltas nominal_minus_mean",
    }

    abs_cols = {
        "Δ_mode": "Dose (Gy) abs deltas abs_nominal_minus_mode",
        "Δ_median": "Dose (Gy) abs deltas abs_nominal_minus_q50",
        "Δ_mean": "Dose (Gy) abs deltas abs_nominal_minus_mean",
    }

    # Sanity check: only keep mappings that actually exist
    valid_kinds: list[str] = []
    for kind in signed_cols:
        s_col = signed_cols[kind]
        a_col = abs_cols[kind]
        if s_col in delta_design_df.columns and a_col in delta_design_df.columns:
            valid_kinds.append(kind)

    if not valid_kinds:
        raise ValueError(
            "No matching signed/absolute delta columns found in delta_design_df."
        )

    drop_cols = [signed_cols[k] for k in valid_kinds] + [abs_cols[k] for k in valid_kinds]

    # Base frame: all predictors / IDs but no per-aggregate delta columns
    base = delta_design_df.drop(columns=drop_cols).copy()

    long_frames: list[pd.DataFrame] = []

    for kind in valid_kinds:
        s_col = signed_cols[kind]
        a_col = abs_cols[kind]

        tmp = base.copy()
        tmp["Delta kind"] = kind

        # Attach signed and absolute deltas
        tmp["Delta (Gy)"] = delta_design_df[s_col].astype(float).to_numpy()
        tmp["|Delta|"] = delta_design_df[a_col].astype(float).to_numpy()

        # log1p(|Δ|)
        tmp["log1p|Delta|"] = np.log1p(np.abs(tmp["|Delta|"]))

        long_frames.append(tmp)

    design_long = pd.concat(long_frames, ignore_index=True)

    return design_long


def summarize_mc_dose_grad_by_sextant(
    all_voxel_wise_dose_df: pd.DataFrame,
    sextant_df: pd.DataFrame,
    sextant_cols: tuple[str, str, str] = (
        "Bx voxel prostate sextant (LR)",
        "Bx voxel prostate sextant (AP)",
        "Bx voxel prostate sextant (SI)",
    ),
) -> pd.DataFrame:
    """
    Robust summary of dose and dose gradient across all MC trials,
    stratified by per voxel prostate double sextant.

    all_voxel_wise_dose_df columns include:
        ['Voxel index', 'MC trial num', 'Dose (Gy)', 'Dose grad (Gy/mm)',
         'X (Bx frame)', 'Y (Bx frame)', 'Z (Bx frame)', 'R (Bx frame)',
         'Simulated bool', 'Simulated type', 'Bx index',
         'Bx ID', 'Patient ID', 'Voxel begin (Z)', 'Voxel end (Z)']

    sextant_df columns include:
        ['Patient ID', 'Bx ID', 'Bx index', 'Voxel index', 'Simulated type',
         'Simulated bool', 
         'Bx voxel prostate sextant (LR)',
         'Bx voxel prostate sextant (AP)',
         'Bx voxel prostate sextant (SI)']
    """
    join_keys = ["Patient ID", "Bx ID", "Bx index", "Voxel index"]

    sextant_sub = sextant_df.loc[
        :,
        join_keys + list(sextant_cols),
    ].drop_duplicates()

    merged = all_voxel_wise_dose_df.merge(
        sextant_sub,
        on=join_keys,
        how="inner",
        validate="m:1",  # many trials per voxel, single sextant per voxel
    )

    value_cols = [c for c in ["Dose (Gy)", "Dose grad (Gy/mm)"] if c in merged.columns]
    if not value_cols:
        raise ValueError("Dose / gradient columns not found in all_voxel_wise_dose_df")

    group_cols = list(sextant_cols)

    # Main robust summaries for dose / grad
    summary = _robust_group_summary(
        df=merged,
        group_cols=group_cols,
        value_cols=value_cols,
    )

    # ----- Add counts: n_voxels, n_trials, n_datapoints -----

    # Create a unique voxel id per Patient/Bx/Voxel
    tmp = merged.copy()
    tmp["_voxel_id"] = (
        tmp["Patient ID"].astype(str)
        + "|"
        + tmp["Bx ID"].astype(str)
        + "|"
        + tmp["Bx index"].astype(str)
        + "|"
        + tmp["Voxel index"].astype(str)
    )

    group = tmp.groupby(group_cols, dropna=False)

    n_voxels = group["_voxel_id"].nunique().rename("n_voxels").reset_index()
    n_trials = group["MC trial num"].nunique().rename("n_trials").reset_index()
    n_datapoints = group.size().rename("n_datapoints").reset_index()

    # Merge counts into the summary
    summary = (
        summary
        .merge(n_voxels, on=group_cols, how="left")
        .merge(n_trials, on=group_cols, how="left")
        .merge(n_datapoints, on=group_cols, how="left")
    )

    return summary


def summarize_mc_deltas_by_sextant(
    mc_deltas: pd.DataFrame,
    sextant_df: pd.DataFrame,
    sextant_cols: tuple[str, str, str] = (
        "Bx voxel prostate sextant (LR)",
        "Bx voxel prostate sextant (AP)",
        "Bx voxel prostate sextant (SI)",
    ),
) -> pd.DataFrame:
    """
    Robust summary of trial wise deltas, stratified by voxel sextant.

    mc_deltas columns (MultiIndex or tuple-valued) include for example:
        ('Dose (Gy) deltas', 'nominal_minus_trial'),
        ('Dose (Gy) abs deltas', 'abs_nominal_minus_trial'),
        ('Dose grad (Gy/mm) deltas', 'nominal_minus_trial'),
        ('Dose grad (Gy/mm) abs deltas', 'abs_nominal_minus_trial'),
    plus ID columns: Patient ID, Bx index, Bx ID, Voxel index.
    """
    mc_flat = _flatten_columns(mc_deltas)

    join_keys = ["Patient ID", "Bx ID", "Bx index", "Voxel index"]

    sextant_sub = sextant_df.loc[
        :,
        join_keys + list(sextant_cols),
    ].drop_duplicates()

    merged = mc_flat.merge(
        sextant_sub,
        on=join_keys,
        how="inner",
        validate="m:1",
    )

    candidate_cols = [
        "Dose (Gy) deltas nominal_minus_trial",
        "Dose (Gy) abs deltas abs_nominal_minus_trial",
        "Dose grad (Gy/mm) deltas nominal_minus_trial",
        "Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial",
    ]
    value_cols = [c for c in candidate_cols if c in merged.columns]
    if not value_cols:
        raise ValueError("Expected trial wise delta columns not found in mc_deltas")

    group_cols = list(sextant_cols)

    # Main robust summaries for trial deltas
    summary = _robust_group_summary(
        df=merged,
        group_cols=group_cols,
        value_cols=value_cols,
    )

    # ----- Add counts: n_voxels, n_trials, n_datapoints -----

    tmp = merged.copy()
    tmp["_voxel_id"] = (
        tmp["Patient ID"].astype(str)
        + "|"
        + tmp["Bx ID"].astype(str)
        + "|"
        + tmp["Bx index"].astype(str)
        + "|"
        + tmp["Voxel index"].astype(str)
    )

    group = tmp.groupby(group_cols, dropna=False)

    n_voxels = group["_voxel_id"].nunique().rename("n_voxels").reset_index()
    n_trials = group["MC trial num"].nunique().rename("n_trials").reset_index()
    n_datapoints = group.size().rename("n_datapoints").reset_index()

    summary = (
        summary
        .merge(n_voxels, on=group_cols, how="left")
        .merge(n_trials, on=group_cols, how="left")
        .merge(n_datapoints, on=group_cols, how="left")
    )

    return summary



def summarize_nominal_deltas_by_sextant(
    nominal_deltas_df_with_abs: pd.DataFrame,
    sextant_df: pd.DataFrame,
    sextant_cols: tuple[str, str, str] = (
        "Bx voxel prostate sextant (LR)",
        "Bx voxel prostate sextant (AP)",
        "Bx voxel prostate sextant (SI)",
    ),
) -> pd.DataFrame:
    """
    Robust summary of voxel level bias deltas (nominal minus aggregate)
    stratified by voxel prostate double sextant.

    nominal_deltas_df_with_abs has MultiIndex or tuple-valued columns like:
        ('Dose (Gy) deltas', 'nominal_minus_mean'),
        ('Dose (Gy) deltas', 'nominal_minus_mode'),
        ('Dose (Gy) deltas', 'nominal_minus_q50'),
        ('Dose (Gy) abs deltas', 'abs_nominal_minus_mean'),
        ('Dose (Gy) abs deltas', 'abs_nominal_minus_mode'),
        ('Dose (Gy) abs deltas', 'abs_nominal_minus_q50'),
    plus ID columns (Patient ID, Bx ID, Bx index, Voxel index).
    """
    nom_flat = _flatten_columns(nominal_deltas_df_with_abs)

    join_keys = ["Patient ID", "Bx ID", "Bx index", "Voxel index"]

    sextant_sub = sextant_df.loc[
        :,
        join_keys + list(sextant_cols),
    ].drop_duplicates()

    merged = nom_flat.merge(
        sextant_sub,
        on=join_keys,
        how="inner",
        validate="1:1",  # one row per voxel
    )

    candidate_cols = [
        "Dose (Gy) deltas nominal_minus_mean",
        "Dose (Gy) deltas nominal_minus_mode",
        "Dose (Gy) deltas nominal_minus_q50",
        "Dose (Gy) abs deltas abs_nominal_minus_mean",
        "Dose (Gy) abs deltas abs_nominal_minus_mode",
        "Dose (Gy) abs deltas abs_nominal_minus_q50",
    ]
    value_cols = [c for c in candidate_cols if c in merged.columns]
    if not value_cols:
        raise ValueError("Expected nominal delta columns not found in nominal_deltas_df_with_abs")

    group_cols = list(sextant_cols)

    # Main robust summaries for nominal bias deltas
    summary = _robust_group_summary(
        df=merged,
        group_cols=group_cols,
        value_cols=value_cols,
    )

    # ----- Add counts: n_voxels and n_datapoints (same here) -----

    group = merged.groupby(group_cols, dropna=False)
    n_voxels = group.size().rename("n_voxels").reset_index()
    n_datapoints = n_voxels.rename(columns={"n_voxels": "n_datapoints"})

    summary = (
        summary
        .merge(n_voxels, on=group_cols, how="left")
        .merge(n_datapoints, on=group_cols, how="left")
    )

    return summary




