from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

import deltas_bias_analysis
import helper_funcs
import summary_statistics
from load_data_shared import CommonLoadedData


@dataclass
class CohortQAOutputs:
    nominal_deltas_df_with_abs: pd.DataFrame
    nominal_gradient_deltas_df_with_abs: pd.DataFrame
    delta_design_df: pd.DataFrame
    delta_long_design: pd.DataFrame
    delta_corr_log1p_df: pd.DataFrame
    delta_corr_signed_df: pd.DataFrame
    delta_corr_abs_df: pd.DataFrame
    effect_size_dfs: dict[str, pd.DataFrame]
    effect_size_grad_dfs: dict[str, pd.DataFrame]
    mean_diff_stats_df: pd.DataFrame
    mean_diff_values_df: pd.DataFrame
    mean_diff_patient_pooled_df: pd.DataFrame
    mean_diff_cohort_pooled_df: pd.DataFrame
    mean_diff_grad_stats_df: pd.DataFrame
    mean_diff_grad_values_df: pd.DataFrame
    mean_diff_grad_patient_pooled_df: pd.DataFrame
    mean_diff_grad_cohort_pooled_df: pd.DataFrame
    dose_differences_df: pd.DataFrame
    dose_differences_grad_df: pd.DataFrame
    length_scale_per_biopsy_df: pd.DataFrame
    length_scale_cohort_df: pd.DataFrame
    length_scale_grad_per_biopsy_df: pd.DataFrame
    length_scale_grad_cohort_df: pd.DataFrame
    delta_log1p_stats_df: pd.DataFrame
    sextant_mc_dose_grad_df: pd.DataFrame
    sextant_mc_deltas_df: pd.DataFrame
    sextant_nominal_deltas_df: pd.DataFrame
    interdelta_corr_signed_df: pd.DataFrame
    interdelta_corr_abs_df: pd.DataFrame
    interdelta_corr_log_abs_df: pd.DataFrame
    dvh_metrics_statistics_df: pd.DataFrame


def default_delta_predictor_cols() -> list[str]:
    return [
        "Nominal dose (Gy)",
        "Nominal dose grad (Gy/mm)",
        "BX_to_prostate_centroid_distance_norm_mean_dim",
        "DIL Flatness",
    ]


def default_delta_predictor_label_map() -> dict[str, str]:
    return {
        "Nominal dose (Gy)": r"$D_{b,v}^{(0)}\ \mathrm{(Gy)}$",
        "Nominal dose grad (Gy/mm)": r"$G_{b,v}^{(0)}\ \mathrm{(Gy\ mm^{-1})}$",
        "BX_to_prostate_centroid_distance_norm_mean_dim": r"$\overline{d}^{\mathrm{norm,cen}}_{\mathrm{P},v}$",
        "DIL Flatness": r"$\mathrm{Flatness}_{\mathrm{DIL}}$",
    }


def default_delta_kind_label_map() -> dict[str, str]:
    return {
        "Δ_median": r"$j = Q_{50}$",
        "Δ_mean": r"$j = mean$",
        "Δ_mode": r"$j = mode$",
    }


def default_zero_x_predictors() -> tuple[str, ...]:
    return (
        "Nominal dose (Gy)",
        "Nominal dose grad (Gy/mm)",
    )


def _delta_correlation_exclude_cols() -> list[str]:
    return [
        "Voxel begin (Z)",
        "Voxel end (Z)",
        "Prostate Structure index",
        "Prostate DIL centroid (X, prostate frame)",
        "Prostate DIL centroid (Y, prostate frame)",
        "Prostate DIL centroid (Z, prostate frame)",
        "Prostate DIL centroid distance (prostate frame)",
        "Prostate DIL prostate sextant (LR)",
        "Prostate DIL prostate sextant (AP)",
        "Prostate DIL prostate sextant (SI)",
        "DIL PCA eigenvector major",
        "DIL PCA eigenvector minor",
        "DIL PCA eigenvector least",
        "Simulated bool",
        "Simulated type",
        "Volume (mm3)",
        "Voxel side length (mm)",
        "Relative DIL ID",
        "Relative DIL index",
        "Relative prostate ID",
        "Relative prostate index",
        "Bx position in prostate LR",
        "Bx position in prostate AP",
        "Bx position in prostate SI",
        "DIL DIL prostate sextant (LR)",
        "DIL DIL prostate sextant (AP)",
        "DIL DIL prostate sextant (SI)",
    ]


def _kind_prefix(s: str) -> str:
    if isinstance(s, str) and "Δ_mode" in s:
        return "Δ_mode"
    if isinstance(s, str) and "Δ_median" in s:
        return "Δ_median"
    if isinstance(s, str) and "Δ_mean" in s:
        return "Δ_mean"
    return str(s)


def _ols_stats(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    x_arr = x[m].to_numpy()
    y_arr = y[m].to_numpy()
    n = x_arr.size
    out: dict[str, float | int] = {
        "n": int(n),
        "slope": np.nan,
        "slope_lo": np.nan,
        "slope_hi": np.nan,
        "intercept": np.nan,
        "r2": np.nan,
        "rho": np.nan,
        "rho_p": np.nan,
    }
    if n < 3:
        return out
    X = sm.add_constant(x_arr)
    model = sm.OLS(y_arr, X).fit()
    out["intercept"] = float(model.params[0])
    out["slope"] = float(model.params[1])
    ci = np.asarray(model.conf_int(alpha=0.05))
    out["slope_lo"] = float(ci[1, 0])
    out["slope_hi"] = float(ci[1, 1])
    out["r2"] = float(model.rsquared)
    rho, p = spearmanr(x_arr, y_arr)
    out["rho"] = float(rho)
    out["rho_p"] = float(p)
    return out


def _compute_delta_predictor_regression_stats(
    long_df: pd.DataFrame,
    *,
    predictor_cols: Sequence[str],
    y_col: str,
    delta_kind_label: Sequence[str],
    delta_kind_col: str = "Delta kind",
) -> pd.DataFrame:
    df = long_df.copy()
    if delta_kind_col not in df.columns:
        raise KeyError(f"Expected a '{delta_kind_col}' column in long_df.")
    if y_col not in df.columns:
        raise KeyError(f"Expected a '{y_col}' column in long_df.")

    df["__kind__"] = df[delta_kind_col].map(_kind_prefix)
    df = df[df["__kind__"].isin(list(delta_kind_label))].copy()
    if df.empty:
        raise ValueError(f"No rows left after filtering to delta kinds {list(delta_kind_label)!r}.")

    rows: list[dict[str, object]] = []
    for col in predictor_cols:
        if col not in df.columns:
            raise KeyError(f"Predictor column not found in long_df: {col}")
        pane = df[[col, y_col, "__kind__"]].copy()
        pane = pane.rename(columns={col: "X", y_col: "Y", "__kind__": "Group"}).dropna()
        if pane.empty:
            continue
        for gname, sub in pane.groupby("Group", observed=True, sort=False):
            rows.append(
                {
                    "predictor_col": col,
                    "y_col": y_col,
                    "delta_kind": gname,
                    **_ols_stats(sub["X"], sub["Y"]),
                }
            )
    return pd.DataFrame(rows)


def _compute_dvh_metrics_statistics(cohort_global_dosimetry_dvh_metrics_df: pd.DataFrame) -> pd.DataFrame:
    exclude = [
        "Patient ID",
        "Bx ID",
        "Struct type",
        "Simulated bool",
        "Simulated type",
        "Struct index",
    ]
    value_cols = [c for c in cohort_global_dosimetry_dvh_metrics_df.columns if c not in exclude + ["Metric"]]
    wide = (
        cohort_global_dosimetry_dvh_metrics_df
        .set_index(exclude + ["Metric"])[value_cols]
        .unstack(level="Metric")
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1, level=0)
    )
    return summary_statistics.summarize_columns_with_argmax(
        wide,
        col_pairs=None,
        exclude_columns=exclude,
    )


def build_cohort_qa_outputs(
    common: CommonLoadedData,
    cohort_global_dosimetry_dvh_metrics_df: pd.DataFrame | None = None,
) -> CohortQAOutputs:
    nominal_deltas_df_with_abs = summary_statistics.compute_biopsy_nominal_deltas_with_abs(
        common.cohort_global_dosimetry_by_voxel_df
    )
    nominal_gradient_deltas_df_with_abs = summary_statistics.compute_biopsy_nominal_deltas_with_abs(
        common.cohort_global_dosimetry_by_voxel_df,
        zero_level_index_str="Dose grad (Gy/mm)",
    )

    effect_size_dfs: dict[str, pd.DataFrame] = {}
    effect_size_grad_dfs: dict[str, pd.DataFrame] = {}
    for eff_size in ("cohen", "hedges", "mean_diff"):
        effect_size_dfs[eff_size] = helper_funcs.create_eff_size_dataframe(
            common.all_voxel_wise_dose_df,
            "Patient ID",
            "Bx index",
            "Bx ID",
            "Voxel index",
            "Dose (Gy)",
            eff_size=eff_size,
            paired_bool=True,
        )
        effect_size_grad_dfs[eff_size] = helper_funcs.create_eff_size_dataframe(
            common.all_voxel_wise_dose_df,
            "Patient ID",
            "Bx index",
            "Bx ID",
            "Voxel index",
            "Dose grad (Gy/mm)",
            eff_size=eff_size,
            paired_bool=True,
        )

    (
        mean_diff_stats_df,
        mean_diff_values_df,
        mean_diff_patient_pooled_df,
        mean_diff_cohort_pooled_df,
    ) = helper_funcs.create_diff_stats_dataframe(
        common.all_voxel_wise_dose_df,
        "Patient ID",
        "Bx index",
        "Bx ID",
        "Voxel index",
        "Dose (Gy)",
    )
    (
        mean_diff_grad_stats_df,
        mean_diff_grad_values_df,
        mean_diff_grad_patient_pooled_df,
        mean_diff_grad_cohort_pooled_df,
    ) = helper_funcs.create_diff_stats_dataframe(
        common.all_voxel_wise_dose_df,
        "Patient ID",
        "Bx index",
        "Bx ID",
        "Voxel index",
        "Dose grad (Gy/mm)",
    )

    dose_differences_df = helper_funcs.compute_dose_differences_vectorized(
        common.all_voxel_wise_dose_df,
        column_name="Dose (Gy)",
    )
    dose_differences_grad_df = helper_funcs.compute_dose_differences_vectorized(
        common.all_voxel_wise_dose_df,
        column_name="Dose grad (Gy/mm)",
    )

    length_scale_per_biopsy_df = summary_statistics.compute_summary(
        dose_differences_df,
        ["Patient ID", "Bx index", "length_scale"],
        ["dose_diff", "dose_diff_abs"],
    )
    length_scale_cohort_df = summary_statistics.compute_summary(
        dose_differences_df,
        ["length_scale"],
        ["dose_diff", "dose_diff_abs"],
    )
    length_scale_grad_per_biopsy_df = summary_statistics.compute_summary(
        dose_differences_grad_df,
        ["Patient ID", "Bx index", "length_scale"],
        ["dose_diff", "dose_diff_abs"],
    )
    length_scale_grad_cohort_df = summary_statistics.compute_summary(
        dose_differences_grad_df,
        ["length_scale"],
        ["dose_diff", "dose_diff_abs"],
    )

    delta_design_df = helper_funcs.build_deltas_with_spatial_radiomics_and_distances_v2(
        nominal_deltas_df_with_abs=nominal_deltas_df_with_abs,
        biopsy_basic_df=common.cohort_biopsy_basic_spatial_features_df,
        radiomics_df=common.cohort_3d_radiomic_features_all_oar_dil_df,
        distances_df=common.cohort_voxel_level_distances_statistics_filtered_df,
        all_voxel_wise_dose_df=common.all_voxel_wise_dose_df,
        radiomics_feature_cols=None,
        bx_voxel_sextant_df=common.cohort_voxel_level_double_sextant_positions_filtered_df,
    )
    delta_long_design = deltas_bias_analysis.make_long_delta_design_from_delta_design(delta_design_df)

    corr_kwargs = {
        "design_df": delta_long_design,
        "key_cols": ("Patient ID", "Bx ID", "Bx index", "Voxel index"),
        "group_col": "Delta kind",
        "min_n": 30,
        "rank_by": "pearson",
        "exclude_cols": _delta_correlation_exclude_cols(),
    }
    delta_corr_log1p_df = deltas_bias_analysis.compute_delta_vs_predictor_correlation_generalized(
        delta_col="log1p|Delta|",
        **corr_kwargs,
    )
    delta_corr_signed_df = deltas_bias_analysis.compute_delta_vs_predictor_correlation_generalized(
        delta_col="Delta (Gy)",
        **corr_kwargs,
    )
    delta_corr_abs_df = deltas_bias_analysis.compute_delta_vs_predictor_correlation_generalized(
        delta_col="|Delta|",
        **corr_kwargs,
    )

    delta_log1p_stats_df = _compute_delta_predictor_regression_stats(
        delta_long_design,
        predictor_cols=default_delta_predictor_cols(),
        y_col="log1p|Delta|",
        delta_kind_label=("Δ_median",),
    )

    mc_deltas_df = summary_statistics.compute_mc_trial_deltas_with_abs(
        common.all_voxel_wise_dose_df
    )
    sextant_mc_dose_grad_df = deltas_bias_analysis.summarize_mc_dose_grad_by_sextant(
        all_voxel_wise_dose_df=common.all_voxel_wise_dose_df,
        sextant_df=common.cohort_voxel_level_double_sextant_positions_filtered_df,
    )
    sextant_mc_deltas_df = deltas_bias_analysis.summarize_mc_deltas_by_sextant(
        mc_deltas=mc_deltas_df,
        sextant_df=common.cohort_voxel_level_double_sextant_positions_filtered_df,
    )
    sextant_nominal_deltas_df = deltas_bias_analysis.summarize_nominal_deltas_by_sextant(
        nominal_deltas_df_with_abs=nominal_deltas_df_with_abs,
        sextant_df=common.cohort_voxel_level_double_sextant_positions_filtered_df,
    )

    combined_wide_deltas_vs_gradient, _ = helper_funcs.build_deltas_vs_gradient_df_with_abs(
        nominal_deltas_df=nominal_deltas_df_with_abs,
        cohort_by_voxel_df=common.cohort_global_dosimetry_by_voxel_df,
        zero_level_index_str="Dose (Gy)",
        gradient_top="Dose grad (Gy/mm)",
        gradient_stats=("nominal", "median", "mean", "mode"),
        gradient_stat=None,
        meta_keep=(
            "Voxel begin (Z)",
            "Voxel end (Z)",
            "Voxel index",
            "Patient ID",
            "Bx ID",
            "Bx index",
            "Simulated bool",
            "Simulated type",
        ),
        add_abs=True,
        add_log1p=True,
        return_long=True,
        require_precomputed_abs=True,
        fallback_recompute_abs=False,
    )
    interdelta_corr_signed_df = deltas_bias_analysis.compute_interdelta_correlations(
        combined_wide_deltas_vs_gradient,
        delta_cols=["Δ_mode (Gy)", "Δ_median (Gy)", "Δ_mean (Gy)"],
    )
    interdelta_corr_abs_df = deltas_bias_analysis.compute_interdelta_correlations(
        combined_wide_deltas_vs_gradient,
        delta_cols=["|Δ_mode| (Gy)", "|Δ_median| (Gy)", "|Δ_mean| (Gy)"],
    )
    interdelta_corr_log_abs_df = deltas_bias_analysis.compute_interdelta_correlations(
        combined_wide_deltas_vs_gradient,
        delta_cols=[
            "log1p|Δ_mode| (Gy)",
            "log1p|Δ_median| (Gy)",
            "log1p|Δ_mean| (Gy)",
        ],
    )

    if cohort_global_dosimetry_dvh_metrics_df is not None:
        dvh_metrics_statistics_df = _compute_dvh_metrics_statistics(cohort_global_dosimetry_dvh_metrics_df)
    else:
        dvh_metrics_statistics_df = pd.DataFrame()

    return CohortQAOutputs(
        nominal_deltas_df_with_abs=nominal_deltas_df_with_abs,
        nominal_gradient_deltas_df_with_abs=nominal_gradient_deltas_df_with_abs,
        delta_design_df=delta_design_df,
        delta_long_design=delta_long_design,
        delta_corr_log1p_df=delta_corr_log1p_df,
        delta_corr_signed_df=delta_corr_signed_df,
        delta_corr_abs_df=delta_corr_abs_df,
        effect_size_dfs=effect_size_dfs,
        effect_size_grad_dfs=effect_size_grad_dfs,
        mean_diff_stats_df=mean_diff_stats_df,
        mean_diff_values_df=mean_diff_values_df,
        mean_diff_patient_pooled_df=mean_diff_patient_pooled_df,
        mean_diff_cohort_pooled_df=mean_diff_cohort_pooled_df,
        mean_diff_grad_stats_df=mean_diff_grad_stats_df,
        mean_diff_grad_values_df=mean_diff_grad_values_df,
        mean_diff_grad_patient_pooled_df=mean_diff_grad_patient_pooled_df,
        mean_diff_grad_cohort_pooled_df=mean_diff_grad_cohort_pooled_df,
        dose_differences_df=dose_differences_df,
        dose_differences_grad_df=dose_differences_grad_df,
        length_scale_per_biopsy_df=length_scale_per_biopsy_df,
        length_scale_cohort_df=length_scale_cohort_df,
        length_scale_grad_per_biopsy_df=length_scale_grad_per_biopsy_df,
        length_scale_grad_cohort_df=length_scale_grad_cohort_df,
        delta_log1p_stats_df=delta_log1p_stats_df,
        sextant_mc_dose_grad_df=sextant_mc_dose_grad_df,
        sextant_mc_deltas_df=sextant_mc_deltas_df,
        sextant_nominal_deltas_df=sextant_nominal_deltas_df,
        interdelta_corr_signed_df=interdelta_corr_signed_df,
        interdelta_corr_abs_df=interdelta_corr_abs_df,
        interdelta_corr_log_abs_df=interdelta_corr_log_abs_df,
        dvh_metrics_statistics_df=dvh_metrics_statistics_df,
    )
