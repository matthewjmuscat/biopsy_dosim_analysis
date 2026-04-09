from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import helper_funcs
from load_data_shared import CommonLoadedData
from qa_path1_thresholds import (
    ThresholdConfig,
    attach_core_nominal_predictors,
    attach_margin_z_scores_from_trials,
    compare_path1_logit_models,
    compute_biopsy_threshold_probabilities,
    compute_nominal_core_averages_from_voxels,
    drop_influential_d2_biopsy_by_cooks,
    fit_path1_logit_per_threshold,
    summarize_path1_by_threshold_v2,
)


@dataclass
class Path1QAOutputs:
    path1_results_df: pd.DataFrame
    path1_enriched_df: pd.DataFrame
    threshold_summary_df: pd.DataFrame
    coef_margin_df: pd.DataFrame
    pred_margin_df: pd.DataFrame
    coef_gradient_df: pd.DataFrame
    pred_gradient_df: pd.DataFrame
    coef_secondary_df: pd.DataFrame
    pred_secondary_df: pd.DataFrame
    model_compare_secondary_df: pd.DataFrame
    best_secondary_df: pd.DataFrame


def default_threshold_configs() -> list[ThresholdConfig]:
    return [
        ThresholdConfig(
            metric_col="D_98% (Gy)",
            threshold=20.0,
            comparison="ge",
            label="D98 ≥ 20 Gy",
        ),
        ThresholdConfig(
            metric_col="D_50% (Gy)",
            threshold=27.0,
            comparison="ge",
            label="D50 ≥ 27 Gy",
        ),
        ThresholdConfig(
            metric_col="D_2% (Gy)",
            threshold=32.0,
            comparison="ge",
            label="D2 ≥ 32 Gy",
        ),
        ThresholdConfig(
            metric_col="V_150% (%)",
            threshold=50.0,
            comparison="ge",
            label="V150 ≥ 50%",
        ),
    ]


def default_secondary_predictors() -> list[str]:
    return [
        "nominal_core_mean_grad_gy_per_mm",
        "BX_to_prostate_centroid_distance_norm_mean_dim",
        "DIL centroid dist mean",
        "Prostate centroid dist mean",
        "Rectum centroid dist mean",
        "Urethra centroid dist mean",
        "DIL NN dist mean",
        "Prostate NN dist mean",
        "Rectum NN dist mean",
        "Urethra NN dist mean",
        "Length (mm)",
        "DIL Volume",
        "DIL Surface area",
        "DIL Surface area to volume ratio",
        "DIL Sphericity",
        "DIL Compactness 1",
        "DIL Compactness 2",
        "DIL Spherical disproportion",
        "DIL Maximum 3D diameter",
        "DIL PCA major",
        "DIL PCA minor",
        "DIL PCA least",
        "Prostate Volume",
        "Prostate Surface area",
        "Prostate Surface area to volume ratio",
        "Prostate Sphericity",
        "Prostate Compactness 1",
        "Prostate Compactness 2",
        "Prostate Spherical disproportion",
        "Prostate Maximum 3D diameter",
        "Prostate PCA major",
        "Prostate PCA minor",
        "Prostate PCA least",
    ]


def build_path1_qa_outputs(
    common: CommonLoadedData,
    calculated_dvh_metrics_per_trial_df: pd.DataFrame,
) -> Path1QAOutputs:
    path1_results_df = compute_biopsy_threshold_probabilities(
        calculated_dvh_metrics_per_trial_df,
        configs=default_threshold_configs(),
        prob_pass_cutoffs=(0.05, 0.95),
    )
    path1_results_df, _, _, _ = drop_influential_d2_biopsy_by_cooks(path1_results_df)
    path1_results_df = attach_margin_z_scores_from_trials(
        path1_results_df,
        calculated_dvh_metrics_per_trial_df,
    )

    core_nominal_df = compute_nominal_core_averages_from_voxels(common.all_voxel_wise_dose_df)
    path1_enriched_df = attach_core_nominal_predictors(path1_results_df, core_nominal_df)
    threshold_summary_df = summarize_path1_by_threshold_v2(path1_enriched_df)

    coef_margin_df, pred_margin_df = fit_path1_logit_per_threshold(
        path1_enriched_df,
        predictors=("distance_from_threshold_nominal",),
    )
    coef_gradient_df, pred_gradient_df = fit_path1_logit_per_threshold(
        path1_enriched_df,
        predictors=("distance_from_threshold_nominal", "nominal_core_mean_grad_gy_per_mm"),
    )

    design_for_models = helper_funcs.build_path1_margin_with_spatial_radiomics_and_distances(
        path1_enriched_df=path1_enriched_df,
        biopsy_basic_df=common.cohort_biopsy_basic_spatial_features_df,
        radiomics_df=common.cohort_3d_radiomic_features_all_oar_dil_df,
        distances_df=common.cohort_biopsy_level_distances_statistics_filtered_df,
    )

    secondary_predictors = default_secondary_predictors()
    base_margin_col = "distance_from_threshold_nominal"
    required_cols = [
        base_margin_col,
        "metric",
        "threshold",
        "label",
        "p_pass",
        "n_pass",
        "n_trials",
        "qa_class",
        *secondary_predictors,
    ]
    design_cc = design_for_models.dropna(subset=required_cols).copy()
    coef_margin_cc_df, _ = fit_path1_logit_per_threshold(
        design_cc,
        predictors=(base_margin_col,),
    )

    coef_rows: list[pd.DataFrame] = []
    pred_rows: list[pd.DataFrame] = []
    compare_rows: list[pd.DataFrame] = []
    for sec_col in secondary_predictors:
        coef_df, pred_df = fit_path1_logit_per_threshold(
            design_cc,
            predictors=(base_margin_col, sec_col),
        )
        coef_df["secondary_predictor"] = sec_col
        pred_df["secondary_predictor"] = sec_col
        coef_rows.append(coef_df)
        pred_rows.append(pred_df)
        cmp_df = compare_path1_logit_models(coef_margin_cc_df, coef_df)
        cmp_df["secondary_predictor"] = sec_col
        compare_rows.append(cmp_df)

    if coef_rows:
        coef_secondary_df = pd.concat(coef_rows, ignore_index=True)
        pred_secondary_df = pd.concat(pred_rows, ignore_index=True)
        model_compare_secondary_df = pd.concat(compare_rows, ignore_index=True)
        sorted_cmp = model_compare_secondary_df.sort_values(
            by=["metric", "threshold", "label", "delta_aic", "delta_brier_w", "lr_pvalue"],
            ascending=[True, True, True, True, True, True],
        )
        best_secondary_df = (
            sorted_cmp.groupby(["metric", "threshold", "label"], as_index=False).first().copy()
        )
    else:
        coef_secondary_df = pd.DataFrame()
        pred_secondary_df = pd.DataFrame()
        model_compare_secondary_df = pd.DataFrame()
        best_secondary_df = pd.DataFrame()

    return Path1QAOutputs(
        path1_results_df=path1_results_df,
        path1_enriched_df=path1_enriched_df,
        threshold_summary_df=threshold_summary_df,
        coef_margin_df=coef_margin_df,
        pred_margin_df=pred_margin_df,
        coef_gradient_df=coef_gradient_df,
        pred_gradient_df=pred_gradient_df,
        coef_secondary_df=coef_secondary_df,
        pred_secondary_df=pred_secondary_df,
        model_compare_secondary_df=model_compare_secondary_df,
        best_secondary_df=best_secondary_df,
    )
