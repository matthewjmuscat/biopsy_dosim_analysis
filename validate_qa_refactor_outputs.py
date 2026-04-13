from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


REPO_ROOT = Path(__file__).resolve().parent
LEGACY_ROOT = REPO_ROOT / "output_data"
REFACTOR_ROOT = REPO_ROOT / "output_data_QA"
VALIDATION_DIR = REPO_ROOT / "output_data_validation"


@dataclass(frozen=True)
class FilePair:
    category: str
    legacy_rel: str
    refactor_rel: str


CSV_PAIRS: list[FilePair] = [
    FilePair("dvh", "dvh_metrics/Cohort: DVH metrics per trial.csv", "csv/dvh_metrics/Cohort: DVH metrics per trial.csv"),
    FilePair("dvh", "dvh_metrics/Cohort_DVH_metrics_stats_per_biopsy.csv", "csv/dvh_metrics/Cohort_DVH_metrics_stats_per_biopsy.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_biopsy_threshold_probabilities_with_z.csv", "csv/qa_path1/Cohort_QA_Path1_biopsy_threshold_probabilities_with_z.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_threshold_probabilities_plus_predictors.csv", "csv/qa_path1/Cohort_QA_Path1_threshold_probabilities_plus_predictors.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_threshold_summary.csv", "csv/qa_path1/Cohort_QA_Path1_threshold_summary.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_coef_margin_only.csv", "csv/qa_path1/Cohort_QA_Path1_logit_coef_margin_only.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_predictions_margin_only.csv", "csv/qa_path1/Cohort_QA_Path1_logit_predictions_margin_only.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_coef_margin_plus_grad.csv", "csv/qa_path1/Cohort_QA_Path1_logit_coef_margin_plus_grad.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_predictions_margin_plus_grad.csv", "csv/qa_path1/Cohort_QA_Path1_logit_predictions_margin_plus_grad.csv"),
    FilePair("path1", "qa_path1/path1_logit_model_compare.csv", "csv/qa_path1/path1_logit_model_compare.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_margin_with_geometry_and_radiomics.csv", "csv/qa_path1/Cohort_QA_Path1_margin_with_geometry_and_radiomics.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_threshold_plus_spatial_radiomics_distances.csv", "csv/qa_path1/Cohort_QA_Path1_threshold_plus_spatial_radiomics_distances.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_margin_predictor_correlations_by_threshold.csv", "csv/qa_path1/Cohort_QA_Path1_margin_predictor_correlations_by_threshold.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_margin_categorical_summaries.csv", "csv/qa_path1/Cohort_QA_Path1_margin_categorical_summaries.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_coef_margin_plus_secondary_scan.csv", "csv/qa_path1/Cohort_QA_Path1_logit_coef_margin_plus_secondary_scan.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_predictions_margin_plus_secondary_scan.csv", "csv/qa_path1/Cohort_QA_Path1_logit_predictions_margin_plus_secondary_scan.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_model_compare_margin_plus_secondary_scan.csv", "csv/qa_path1/Cohort_QA_Path1_logit_model_compare_margin_plus_secondary_scan.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_model_compare_margin_plus_secondary_scan_sorted.csv", "csv/qa_path1/Cohort_QA_Path1_logit_model_compare_margin_plus_secondary_scan_sorted.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_model_compare_best_secondary_per_threshold.csv", "csv/qa_path1/Cohort_QA_Path1_logit_model_compare_best_secondary_per_threshold.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_model_compare_secondary_ranking.csv", "csv/qa_path1/Cohort_QA_Path1_logit_model_compare_secondary_ranking.csv"),
    FilePair("path1", "qa_path1/Cohort_QA_Path1_logit_delta95_vs_best_secondary_per_threshold.csv", "csv/qa_path1/Cohort_QA_Path1_logit_delta95_vs_best_secondary_per_threshold.csv"),
    FilePair("path1", "qa_path1/1_core/p1_core_01_biopsy_mc_probs_z.csv", "csv/qa_path1/1_core/p1_core_01_biopsy_mc_probs_z.csv"),
    FilePair("path1", "qa_path1/1_core/p1_core_02_biopsy_probs_plus_nominal_predictors.csv", "csv/qa_path1/1_core/p1_core_02_biopsy_probs_plus_nominal_predictors.csv"),
    FilePair("path1", "qa_path1/1_core/p1_core_03_threshold_summary_by_rule.csv", "csv/qa_path1/1_core/p1_core_03_threshold_summary_by_rule.csv"),
    FilePair("path1", "qa_path1/2_logit_margin/p1_logit_margin_01_coef.csv", "csv/qa_path1/2_logit_margin/p1_logit_margin_01_coef.csv"),
    FilePair("path1", "qa_path1/2_logit_margin/p1_logit_margin_02_predictions.csv", "csv/qa_path1/2_logit_margin/p1_logit_margin_02_predictions.csv"),
    FilePair("path1", "qa_path1/3_logit_grad/p1_logit_grad_01_coef.csv", "csv/qa_path1/3_logit_grad/p1_logit_grad_01_coef.csv"),
    FilePair("path1", "qa_path1/3_logit_grad/p1_logit_grad_02_predictions.csv", "csv/qa_path1/3_logit_grad/p1_logit_grad_02_predictions.csv"),
    FilePair("path1", "qa_path1/3_logit_grad/p1_logit_grad_03_model_compare_1d_vs_2d.csv", "csv/qa_path1/3_logit_grad/p1_logit_grad_03_model_compare_1d_vs_2d.csv"),
    FilePair("path1", "qa_path1/4_design/p1_design_01_margin_geom_radiomics_basic.csv", "csv/qa_path1/4_design/p1_design_01_margin_geom_radiomics_basic.csv"),
    FilePair("path1", "qa_path1/4_design/p1_design_02_margin_spatial_radiomics_distances.csv", "csv/qa_path1/4_design/p1_design_02_margin_spatial_radiomics_distances.csv"),
    FilePair("path1", "qa_path1/5_correlations/p1_corr_01_margin_vs_predictors_by_threshold.csv", "csv/qa_path1/5_correlations/p1_corr_01_margin_vs_predictors_by_threshold.csv"),
    FilePair("path1", "qa_path1/5_correlations/p1_corr_02_margin_categorical_summaries.csv", "csv/qa_path1/5_correlations/p1_corr_02_margin_categorical_summaries.csv"),
    FilePair("path1", "qa_path1/6_secondary_scan/p1_secscan_01_coef_margin_plus_all_secondaries.csv", "csv/qa_path1/6_secondary_scan/p1_secscan_01_coef_margin_plus_all_secondaries.csv"),
    FilePair("path1", "qa_path1/6_secondary_scan/p1_secscan_02_predictions_margin_plus_all_secondaries.csv", "csv/qa_path1/6_secondary_scan/p1_secscan_02_predictions_margin_plus_all_secondaries.csv"),
    FilePair("path1", "qa_path1/6_secondary_scan/p1_secscan_03_model_compare_all_vs_margin_raw.csv", "csv/qa_path1/6_secondary_scan/p1_secscan_03_model_compare_all_vs_margin_raw.csv"),
    FilePair("path1", "qa_path1/6_secondary_scan/p1_secscan_04_model_compare_all_vs_margin_sorted.csv", "csv/qa_path1/6_secondary_scan/p1_secscan_04_model_compare_all_vs_margin_sorted.csv"),
    FilePair("path1", "qa_path1/6_secondary_scan/p1_secscan_05_best_secondary_per_threshold.csv", "csv/qa_path1/6_secondary_scan/p1_secscan_05_best_secondary_per_threshold.csv"),
    FilePair("path1", "qa_path1/6_secondary_scan/p1_secscan_06_secondary_ranking_overall.csv", "csv/qa_path1/6_secondary_scan/p1_secscan_06_secondary_ranking_overall.csv"),
    FilePair("path1", "qa_path1/6_secondary_scan/p1_secscan_07_delta95_vs_best_secondary_per_threshold.csv", "csv/qa_path1/6_secondary_scan/p1_secscan_07_delta95_vs_best_secondary_per_threshold.csv"),
    FilePair("delta_corr", "deltas_bias_correlations/deltas_00_per_voxel_deltas_and_predictors.csv", "csv/deltas_bias_correlations/deltas_00_per_voxel_deltas_and_predictors.csv"),
    FilePair("delta_corr", "deltas_bias_correlations/deltas_01_long_per_voxel_deltas_and_predictors.csv", "csv/deltas_bias_correlations/deltas_01_long_per_voxel_deltas_and_predictors.csv"),
    FilePair("delta_corr", "deltas_bias_correlations/deltas_02a_correlations_log1pDelta_by_delta_kind_and_predictor.csv", "csv/deltas_bias_correlations/deltas_02a_correlations_log1pDelta_by_delta_kind_and_predictor.csv"),
    FilePair("delta_corr", "deltas_bias_correlations/deltas_02b_correlations_signedDelta_by_delta_kind_and_predictor.csv", "csv/deltas_bias_correlations/deltas_02b_correlations_signedDelta_by_delta_kind_and_predictor.csv"),
    FilePair("delta_corr", "deltas_bias_correlations/deltas_02c_correlations_abs_Delta_by_delta_kind_and_predictor.csv", "csv/deltas_bias_correlations/deltas_02c_correlations_abs_Delta_by_delta_kind_and_predictor.csv"),
    FilePair("delta_corr", "deltas_bias_correlations/03c_abs_medianDelta_vs_top4_predictors__stats.csv", "deltas_bias_correlations/03c_abs_medianDelta_vs_top4_predictors__stats.csv"),
    FilePair("experimental", "dvh_metrics/dvh_metrics_statistics_all_patients.csv", "csv/experimental/dvh_metrics/dvh_metrics_statistics_all_patients.csv"),
    FilePair("experimental", "deltas_bias_correlations/03c_log1p_medianDelta_vs_top4_predictors__stats.csv", "csv/experimental/deltas_bias_correlations/03c_log1p_medianDelta_vs_top4_predictors__stats.csv"),
    FilePair("experimental", "deltas_bias_correlations/deltas_04a_deltas_biased_correlations_signed.csv", "csv/experimental/deltas_bias_correlations/deltas_04a_deltas_biased_correlations_signed.csv"),
    FilePair("experimental", "deltas_bias_correlations/deltas_04b_deltas_biased_correlations_abs.csv", "csv/experimental/deltas_bias_correlations/deltas_04b_deltas_biased_correlations_abs.csv"),
    FilePair("experimental", "deltas_bias_correlations/deltas_04c_deltas_biased_correlations_log_abs.csv", "csv/experimental/deltas_bias_correlations/deltas_04c_deltas_biased_correlations_log_abs.csv"),
    FilePair("experimental", "deltas_bias_correlations/sextant_summaries/sextant_mc_dose_grad_summary.csv", "csv/experimental/deltas_bias_correlations/sextant_summaries/sextant_mc_dose_grad_summary.csv"),
    FilePair("experimental", "deltas_bias_correlations/sextant_summaries/sextant_mc_trial_deltas_summary.csv", "csv/experimental/deltas_bias_correlations/sextant_summaries/sextant_mc_trial_deltas_summary.csv"),
    FilePair("experimental", "deltas_bias_correlations/sextant_summaries/sextant_nominal_bias_deltas_summary.csv", "csv/experimental/deltas_bias_correlations/sextant_summaries/sextant_nominal_bias_deltas_summary.csv"),
    FilePair("effect_sizes", "effect_sizes_analysis/effect_sizes_statistics_all_patients.csv_cohen.csv", "csv/effect_sizes_analysis/effect_sizes_statistics_all_patients.csv_cohen.csv"),
    FilePair("effect_sizes", "effect_sizes_analysis/effect_sizes_statistics_all_patients.csv_hedges.csv", "csv/effect_sizes_analysis/effect_sizes_statistics_all_patients.csv_hedges.csv"),
    FilePair("effect_sizes", "effect_sizes_analysis/effect_sizes_statistics_all_patients.csv_mean_diff.csv", "csv/effect_sizes_analysis/effect_sizes_statistics_all_patients.csv_mean_diff.csv"),
    FilePair("effect_sizes", "effect_sizes_analysis/effect_sizes_dose_gradient_statistics_all_patients.csv_cohen.csv", "csv/effect_sizes_analysis/effect_sizes_dose_gradient_statistics_all_patients.csv_cohen.csv"),
    FilePair("effect_sizes", "effect_sizes_analysis/effect_sizes_dose_gradient_statistics_all_patients.csv_hedges.csv", "csv/effect_sizes_analysis/effect_sizes_dose_gradient_statistics_all_patients.csv_hedges.csv"),
    FilePair("effect_sizes", "effect_sizes_analysis/effect_sizes_dose_gradient_statistics_all_patients.csv_mean_diff.csv", "csv/effect_sizes_analysis/effect_sizes_dose_gradient_statistics_all_patients.csv_mean_diff.csv"),
    FilePair("effect_sizes", "effect_sizes_analysis/mean_diff_statistics_all_patients.csv", "csv/effect_sizes_analysis/mean_diff_statistics_all_patients.csv"),
    FilePair("effect_sizes", "effect_sizes_analysis/mean_diff_values_all_patients.csv", "csv/effect_sizes_analysis/mean_diff_values_all_patients.csv"),
    FilePair("effect_sizes", "effect_sizes_analysis/mean_diff_dose_gradient_statistics_all_patients.csv", "csv/effect_sizes_analysis/mean_diff_dose_gradient_statistics_all_patients.csv"),
    FilePair("effect_sizes", "effect_sizes_analysis/mean_diff_dose_gradient_values_all_patients.csv", "csv/effect_sizes_analysis/mean_diff_dose_gradient_values_all_patients.csv"),
    FilePair("length_scales", "length_scales_dosimetry/length_scales_dosimetry_statistics_cohort.csv", "csv/length_scales_dosimetry/length_scales_dosimetry_statistics_cohort.csv"),
    FilePair("length_scales", "length_scales_dosimetry/length_scales_dosimetry_statistics_per_biopsy.csv", "csv/length_scales_dosimetry/length_scales_dosimetry_statistics_per_biopsy.csv"),
    FilePair("length_scales", "length_scales_dosimetry/length_scales_dose_gradient_statistics_cohort.csv", "csv/length_scales_dosimetry/length_scales_dose_gradient_statistics_cohort.csv"),
    FilePair("length_scales", "length_scales_dosimetry/length_scales_dose_gradient_statistics_per_biopsy.csv", "csv/length_scales_dosimetry/length_scales_dose_gradient_statistics_per_biopsy.csv"),
]


FIGURE_PAIRS: list[FilePair] = [
    FilePair("figure", "qa_path1/figures/Fig_Path1_threshold_QA_summary_v2.pdf", "figures/qa_path1/Fig_Path1_threshold_QA_summary_v2.pdf"),
    FilePair("figure", "qa_path1/figures/Fig_Path1_p_pass_vs_margin_by_metric.pdf", "figures/qa_path1/Fig_Path1_p_pass_vs_margin_by_metric.pdf"),
    FilePair("figure", "qa_path1/figures/Fig_Path1_logit_margin_plus_grad_families.pdf", "figures/qa_path1/Fig_Path1_logit_margin_plus_grad_families.pdf"),
    FilePair("figure", "qa_path1/figures/Fig_Path1_logit_margin_plus_best_secondary_families.pdf", "figures/qa_path1/Fig_Path1_logit_margin_plus_best_secondary_families.pdf"),
    FilePair("figure", "figures/cohort_output_figures/histogram_fit_all_voxels_dose.pdf", "figures/cohort_output_figures/histogram_fit_all_voxels_dose.pdf"),
    FilePair("figure", "figures/cohort_output_figures/histogram_fit_all_voxels_dose_gradient.pdf", "figures/cohort_output_figures/histogram_fit_all_voxels_dose_gradient.pdf"),
    FilePair("figure", "figures/cohort_output_figures/dvh_boxplot_d_x.pdf", "figures/cohort_output_figures/dvh_boxplot_d_x.pdf"),
    FilePair("figure", "figures/cohort_output_figures/dvh_boxplot_v_x.pdf", "figures/cohort_output_figures/dvh_boxplot_v_x.pdf"),
    FilePair("figure", "deltas_bias_correlations/03c_abs_medianDelta_vs_top4_predictors.pdf", "deltas_bias_correlations/03c_abs_medianDelta_vs_top4_predictors.pdf"),
    FilePair("figure", "figures/cohort_output_figures/cohort_dose_abs_box_with_all_biopsy_mean_curves_v2.pdf", "figures/cohort_output_figures/cohort_dose_abs_box_with_all_biopsy_mean_curves_v2.pdf"),
    FilePair("figure", "figures/cohort_output_figures/cohort_grad_abs_box_with_all_biopsy_mean_curves_v2.pdf", "figures/cohort_output_figures/cohort_grad_abs_box_with_all_biopsy_mean_curves_v2.pdf"),
    FilePair(
        "figure",
        "figures/cohort_output_figures/effect_sizes_heatmaps/cohort_dualtri_dose_upper_dosegrad_lower_absolute_pooledstats_no_std_v2.pdf",
        "figures/cohort_output_figures/effect_sizes_heatmaps/cohort_dualtri_dose_upper_dosegrad_lower_absolute_pooledstats_no_std_v2.pdf",
    ),
]


LEGACY_QA_CSV_DIRS = [
    LEGACY_ROOT / "qa_path1",
    LEGACY_ROOT / "dvh_metrics",
    LEGACY_ROOT / "effect_sizes_analysis",
    LEGACY_ROOT / "length_scales_dosimetry",
    LEGACY_ROOT / "deltas_bias_correlations",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compare_csv_pair(pair: FilePair) -> dict[str, object]:
    legacy_path = LEGACY_ROOT / pair.legacy_rel
    refactor_path = REFACTOR_ROOT / pair.refactor_rel
    row: dict[str, object] = {
        "kind": "csv",
        "category": pair.category,
        "legacy_rel": pair.legacy_rel,
        "refactor_rel": pair.refactor_rel,
        "legacy_exists": legacy_path.exists(),
        "refactor_exists": refactor_path.exists(),
        "status": "missing",
        "legacy_rows": None,
        "refactor_rows": None,
        "legacy_cols": None,
        "refactor_cols": None,
        "detail": "",
    }
    if not legacy_path.exists() or not refactor_path.exists():
        row["detail"] = "legacy or refactor file missing"
        return row

    legacy_df = _read_csv(legacy_path)
    refactor_df = _read_csv(refactor_path)
    row["legacy_rows"] = len(legacy_df)
    row["refactor_rows"] = len(refactor_df)
    row["legacy_cols"] = len(legacy_df.columns)
    row["refactor_cols"] = len(refactor_df.columns)
    try:
        assert_frame_equal(
            legacy_df,
            refactor_df,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        row["status"] = "match"
    except AssertionError as exc:
        row["status"] = "different"
        row["detail"] = str(exc).splitlines()[0][:500]
    return row


def compare_figure_pair(pair: FilePair) -> dict[str, object]:
    legacy_path = LEGACY_ROOT / pair.legacy_rel
    refactor_path = REFACTOR_ROOT / pair.refactor_rel
    status = "present" if legacy_path.exists() and refactor_path.exists() else "missing"
    return {
        "kind": "figure",
        "category": pair.category,
        "legacy_rel": pair.legacy_rel,
        "refactor_rel": pair.refactor_rel,
        "legacy_exists": legacy_path.exists(),
        "refactor_exists": refactor_path.exists(),
        "status": status,
        "legacy_rows": None,
        "refactor_rows": None,
        "legacy_cols": None,
        "refactor_cols": None,
        "detail": "",
    }


def find_unmapped_legacy_csvs() -> list[str]:
    mapped = {pair.legacy_rel for pair in CSV_PAIRS}
    discovered: set[str] = set()
    for root in LEGACY_QA_CSV_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            rel = path.relative_to(LEGACY_ROOT).as_posix()
            if rel not in mapped:
                discovered.add(rel)
    return sorted(discovered)


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    csv_rows = [compare_csv_pair(pair) for pair in CSV_PAIRS]
    figure_rows = [compare_figure_pair(pair) for pair in FIGURE_PAIRS]
    all_rows = csv_rows + figure_rows
    summary_df = pd.DataFrame(all_rows)
    summary_csv = VALIDATION_DIR / "qa_refactor_validation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    csv_matches = sum(row["status"] == "match" for row in csv_rows)
    csv_different = sum(row["status"] == "different" for row in csv_rows)
    csv_missing = sum(row["status"] == "missing" for row in csv_rows)
    fig_present = sum(row["status"] == "present" for row in figure_rows)
    fig_missing = sum(row["status"] == "missing" for row in figure_rows)
    unmapped_legacy_csvs = find_unmapped_legacy_csvs()

    lines = [
        "# QA Refactor Validation",
        "",
        f"- CSV pairs checked: {len(csv_rows)}",
        f"- CSV exact/dataframe matches: {csv_matches}",
        f"- CSV differing pairs: {csv_different}",
        f"- CSV missing pairs: {csv_missing}",
        f"- Figure targets checked: {len(figure_rows)}",
        f"- Figure targets present: {fig_present}",
        f"- Figure targets missing: {fig_missing}",
        f"- Legacy QA CSVs still unmapped: {len(unmapped_legacy_csvs)}",
        "",
        "## Differing CSV Pairs",
    ]

    differing = [row for row in csv_rows if row["status"] == "different"]
    if differing:
        for row in differing:
            lines.append(
                f"- `{row['legacy_rel']}` vs `{row['refactor_rel']}`: {row['detail']}"
            )
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Missing CSV Pairs")
    missing_csv = [row for row in csv_rows if row["status"] == "missing"]
    if missing_csv:
        for row in missing_csv:
            lines.append(f"- `{row['legacy_rel']}` -> `{row['refactor_rel']}`")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Missing Figure Targets")
    missing_fig = [row for row in figure_rows if row["status"] == "missing"]
    if missing_fig:
        for row in missing_fig:
            lines.append(f"- `{row['legacy_rel']}` -> `{row['refactor_rel']}`")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Unmapped Legacy QA CSVs")
    if unmapped_legacy_csvs:
        for rel in unmapped_legacy_csvs:
            lines.append(f"- `{rel}`")
    else:
        lines.append("- None")

    summary_md = VALIDATION_DIR / "qa_refactor_validation_summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
