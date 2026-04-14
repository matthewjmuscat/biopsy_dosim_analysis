from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

from load_data_QA import load_qa_data
from load_data_shared import build_dataframe_inventory
from pipeline_shared_config import FigureExportConfig, QAOutputConfig, SharedPipelineConfig
from production_plots_QA import (
    plot_cohort_abs_heatmap,
    plot_cohort_delta_vs_predictors,
    plot_cohort_dvh_boxplots,
    plot_cohort_histogram,
    plot_cohort_length_scale_summary,
    plot_path1_best_secondary_families,
    plot_path1_p_pass_vs_margin,
    plot_path1_threshold_qa_summary,
)
from qa_cohort_pipeline import CohortQAOutputs, build_cohort_qa_outputs
from qa_path1_pipeline import Path1QAOutputs, build_path1_qa_outputs
from uncertainty_summary import build_uncertainty_summary_outputs, write_uncertainty_summary_outputs


def _ensure_dirs(output_config: QAOutputConfig) -> dict[str, Path]:
    dirs = {
        "root": output_config.output_root,
        "csv": output_config.output_root / output_config.csv_subdir,
        "manifests": output_config.output_root / output_config.manifest_subdir,
        "figures": output_config.output_root / output_config.figures_subdir,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_csv(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def _generate_path1_qa_figures(figures_dir: Path, path1_outputs: Path1QAOutputs) -> list[Path]:
    export_config = FigureExportConfig()
    figure_paths: list[Path] = []

    figure_paths.extend(
        plot_path1_threshold_qa_summary(
            path1_outputs.path1_results_df,
            save_dir=figures_dir,
            export_config=export_config,
            file_stem="Fig_Path1_threshold_QA_summary_v2",
        )
    )
    figure_paths.extend(
        plot_path1_p_pass_vs_margin(
            path1_outputs.path1_results_df,
            save_dir=figures_dir,
            export_config=export_config,
            file_stem="Fig_Path1_p_pass_vs_margin_by_metric",
            coef_df=path1_outputs.coef_margin_df,
            show_required_margin_line=True,
            required_prob=0.95,
        )
    )
    figure_paths.extend(
        plot_path1_best_secondary_families(
            path1_outputs.pred_gradient_df,
            path1_outputs.coef_gradient_df,
            save_dir=figures_dir,
            export_config=export_config,
            file_stem="Fig_Path1_logit_margin_plus_grad_families",
            comparison_df=path1_outputs.model_compare_gradient_df,
            overlay_1d_model=True,
            coef1_df=path1_outputs.coef_margin_df,
        )
    )
    figure_paths.extend(
        plot_path1_best_secondary_families(
            path1_outputs.pred_secondary_df,
            path1_outputs.coef_secondary_df,
            save_dir=figures_dir,
            export_config=export_config,
            file_stem="Fig_Path1_logit_margin_plus_best_secondary_families",
            comparison_df=path1_outputs.best_secondary_df,
            overlay_1d_model=True,
            coef1_df=path1_outputs.coef_margin_df,
        )
    )
    return figure_paths


def _generate_cohort_qa_figures(figures_root: Path, qa_data, cohort_outputs: CohortQAOutputs) -> list[Path]:
    export_config = FigureExportConfig()
    cohort_figures_dir = figures_root / "cohort_output_figures"
    heatmap_dir = cohort_figures_dir / "effect_sizes_heatmaps"
    delta_corr_fig_dir = figures_root.parent / "deltas_bias_correlations"
    cohort_figures_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    delta_corr_fig_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: list[Path] = []
    figure_paths.extend(
        plot_cohort_histogram(
            qa_data.common.all_voxel_wise_dose_df,
            save_dir=cohort_figures_dir,
            export_config=export_config,
            dose_col="Dose (Gy)",
            file_stem="histogram_fit_all_voxels_dose",
            xrange=(0, 60),
            quantity_tex=r"D_{b,v}^{(t)}",
            quantity_unit_tex="Gy",
        )
    )
    figure_paths.extend(
        plot_cohort_histogram(
            qa_data.common.all_voxel_wise_dose_df,
            save_dir=cohort_figures_dir,
            export_config=export_config,
            dose_col="Dose grad (Gy/mm)",
            file_stem="histogram_fit_all_voxels_dose_gradient",
            xrange=(0, 10),
            quantity_tex=r"G_{b,v}^{(t)}",
            quantity_unit_tex=r"Gy mm$^{-1}$",
        )
    )
    figure_paths.extend(
        plot_cohort_dvh_boxplots(
            qa_data.cohort_global_dosimetry_dvh_metrics_df,
            save_dir=cohort_figures_dir,
            export_config=export_config,
            file_stem="dvh_boxplot",
        )
    )

    _, _, _, delta_stats_csv, _ = plot_cohort_delta_vs_predictors(
        cohort_outputs.delta_long_design,
        save_dir=delta_corr_fig_dir,
        export_config=export_config,
        file_stem="03c_abs_medianDelta_vs_top4_predictors",
    )
    if delta_stats_csv is not None:
        pass
    figure_paths.extend(
        [
            delta_corr_fig_dir / "03c_abs_medianDelta_vs_top4_predictors.pdf",
            delta_corr_fig_dir / "03c_abs_medianDelta_vs_top4_predictors.svg",
        ]
    )

    figure_paths.extend(
        plot_cohort_length_scale_summary(
            cohort_outputs.dose_differences_df,
            save_dir=cohort_figures_dir,
            export_config=export_config,
            file_stem="cohort_dose_abs_box_with_all_biopsy_mean_curves_v2",
            metric_family="dose",
            y_max_fixed=40.0,
        )
    )
    figure_paths.extend(
        plot_cohort_length_scale_summary(
            cohort_outputs.dose_differences_grad_df,
            save_dir=cohort_figures_dir,
            export_config=export_config,
            file_stem="cohort_grad_abs_box_with_all_biopsy_mean_curves_v2",
            metric_family="grad",
            y_max_fixed=25.0,
        )
    )
    figure_paths.extend(
        plot_cohort_abs_heatmap(
            cohort_outputs.mean_diff_cohort_pooled_df,
            cohort_outputs.mean_diff_grad_cohort_pooled_df,
            save_dir=heatmap_dir,
            export_config=export_config,
            file_stem="cohort_dualtri_dose_upper_dosegrad_lower_absolute_pooledstats_no_std_v2",
        )
    )
    return figure_paths


def _write_path1_tables(csv_dir: Path, path1_outputs: Path1QAOutputs) -> None:
    path1_root = csv_dir / "qa_path1"
    core_dir = path1_root / "1_core"
    margin_dir = path1_root / "2_logit_margin"
    grad_dir = path1_root / "3_logit_grad"
    design_dir = path1_root / "4_design"
    corr_dir = path1_root / "5_correlations"
    secondary_dir = path1_root / "6_secondary_scan"
    for path in [path1_root, core_dir, margin_dir, grad_dir, design_dir, corr_dir, secondary_dir]:
        path.mkdir(parents=True, exist_ok=True)

    _write_csv(path1_root / "Cohort_QA_Path1_biopsy_threshold_probabilities_with_z.csv", path1_outputs.path1_results_df)
    _write_csv(core_dir / "p1_core_01_biopsy_mc_probs_z.csv", path1_outputs.path1_results_df)

    _write_csv(path1_root / "Cohort_QA_Path1_threshold_probabilities_plus_predictors.csv", path1_outputs.path1_enriched_df)
    _write_csv(core_dir / "p1_core_02_biopsy_probs_plus_nominal_predictors.csv", path1_outputs.path1_enriched_df)

    _write_csv(path1_root / "Cohort_QA_Path1_threshold_summary.csv", path1_outputs.threshold_summary_df)
    _write_csv(core_dir / "p1_core_03_threshold_summary_by_rule.csv", path1_outputs.threshold_summary_df)

    _write_csv(path1_root / "Cohort_QA_Path1_logit_coef_margin_only.csv", path1_outputs.coef_margin_df)
    _write_csv(margin_dir / "p1_logit_margin_01_coef.csv", path1_outputs.coef_margin_df)
    _write_csv(path1_root / "Cohort_QA_Path1_logit_predictions_margin_only.csv", path1_outputs.pred_margin_df)
    _write_csv(margin_dir / "p1_logit_margin_02_predictions.csv", path1_outputs.pred_margin_df)

    _write_csv(path1_root / "Cohort_QA_Path1_logit_coef_margin_plus_grad.csv", path1_outputs.coef_gradient_df)
    _write_csv(grad_dir / "p1_logit_grad_01_coef.csv", path1_outputs.coef_gradient_df)
    _write_csv(path1_root / "Cohort_QA_Path1_logit_predictions_margin_plus_grad.csv", path1_outputs.pred_gradient_df)
    _write_csv(grad_dir / "p1_logit_grad_02_predictions.csv", path1_outputs.pred_gradient_df)
    _write_csv(path1_root / "path1_logit_model_compare.csv", path1_outputs.model_compare_gradient_df)
    _write_csv(grad_dir / "p1_logit_grad_03_model_compare_1d_vs_2d.csv", path1_outputs.model_compare_gradient_df)

    _write_csv(path1_root / "Cohort_QA_Path1_margin_with_geometry_and_radiomics.csv", path1_outputs.design_basic_df)
    _write_csv(design_dir / "p1_design_01_margin_geom_radiomics_basic.csv", path1_outputs.design_basic_df)
    _write_csv(
        path1_root / "Cohort_QA_Path1_threshold_plus_spatial_radiomics_distances.csv",
        path1_outputs.design_spatial_radiomics_distances_df,
    )
    _write_csv(
        design_dir / "p1_design_02_margin_spatial_radiomics_distances.csv",
        path1_outputs.design_spatial_radiomics_distances_df,
    )

    _write_csv(
        path1_root / "Cohort_QA_Path1_margin_predictor_correlations_by_threshold.csv",
        path1_outputs.margin_correlations_df,
    )
    _write_csv(corr_dir / "p1_corr_01_margin_vs_predictors_by_threshold.csv", path1_outputs.margin_correlations_df)
    _write_csv(path1_root / "Cohort_QA_Path1_margin_categorical_summaries.csv", path1_outputs.margin_categorical_summary_df)
    _write_csv(corr_dir / "p1_corr_02_margin_categorical_summaries.csv", path1_outputs.margin_categorical_summary_df)

    _write_csv(path1_root / "Cohort_QA_Path1_logit_coef_margin_plus_secondary_scan.csv", path1_outputs.coef_secondary_df)
    _write_csv(secondary_dir / "p1_secscan_01_coef_margin_plus_all_secondaries.csv", path1_outputs.coef_secondary_df)
    _write_csv(path1_root / "Cohort_QA_Path1_logit_predictions_margin_plus_secondary_scan.csv", path1_outputs.pred_secondary_df)
    _write_csv(secondary_dir / "p1_secscan_02_predictions_margin_plus_all_secondaries.csv", path1_outputs.pred_secondary_df)
    _write_csv(
        path1_root / "Cohort_QA_Path1_logit_model_compare_margin_plus_secondary_scan.csv",
        path1_outputs.model_compare_secondary_df,
    )
    _write_csv(secondary_dir / "p1_secscan_03_model_compare_all_vs_margin_raw.csv", path1_outputs.model_compare_secondary_df)
    _write_csv(
        path1_root / "Cohort_QA_Path1_logit_model_compare_margin_plus_secondary_scan_sorted.csv",
        path1_outputs.model_compare_secondary_sorted_df,
    )
    _write_csv(
        secondary_dir / "p1_secscan_04_model_compare_all_vs_margin_sorted.csv",
        path1_outputs.model_compare_secondary_sorted_df,
    )
    _write_csv(
        path1_root / "Cohort_QA_Path1_logit_model_compare_best_secondary_per_threshold.csv",
        path1_outputs.best_secondary_df,
    )
    _write_csv(secondary_dir / "p1_secscan_05_best_secondary_per_threshold.csv", path1_outputs.best_secondary_df)
    _write_csv(path1_root / "Cohort_QA_Path1_logit_model_compare_secondary_ranking.csv", path1_outputs.secondary_ranking_df)
    _write_csv(secondary_dir / "p1_secscan_06_secondary_ranking_overall.csv", path1_outputs.secondary_ranking_df)
    _write_csv(
        path1_root / "Cohort_QA_Path1_logit_delta95_vs_best_secondary_per_threshold.csv",
        path1_outputs.delta95_secondary_df,
    )
    _write_csv(secondary_dir / "p1_secscan_07_delta95_vs_best_secondary_per_threshold.csv", path1_outputs.delta95_secondary_df)


def _write_cohort_tables(csv_dir: Path, qa_data, cohort_outputs: CohortQAOutputs) -> None:
    dvh_dir = csv_dir / "dvh_metrics"
    effect_sizes_dir = csv_dir / "effect_sizes_analysis"
    length_scales_dir = csv_dir / "length_scales_dosimetry"
    delta_corr_dir = csv_dir / "deltas_bias_correlations"
    for path in [dvh_dir, effect_sizes_dir, length_scales_dir, delta_corr_dir]:
        path.mkdir(parents=True, exist_ok=True)

    _write_csv(csv_dir / "qa_dvh_metrics_per_trial.csv", qa_data.calculated_dvh_metrics_per_trial_df)
    _write_csv(csv_dir / "qa_dvh_metrics_per_biopsy_summary.csv", qa_data.cohort_global_dosimetry_dvh_metrics_df)
    _write_csv(dvh_dir / "Cohort: DVH metrics per trial.csv", qa_data.calculated_dvh_metrics_per_trial_df)
    _write_csv(dvh_dir / "Cohort_DVH_metrics_stats_per_biopsy.csv", qa_data.cohort_global_dosimetry_dvh_metrics_df)

    for eff_size, df in cohort_outputs.effect_size_dfs.items():
        _write_csv(effect_sizes_dir / f"effect_sizes_statistics_all_patients.csv_{eff_size}.csv", df)
    for eff_size, df in cohort_outputs.effect_size_grad_dfs.items():
        _write_csv(effect_sizes_dir / f"effect_sizes_dose_gradient_statistics_all_patients.csv_{eff_size}.csv", df)

    _write_csv(effect_sizes_dir / "mean_diff_statistics_all_patients.csv", cohort_outputs.mean_diff_stats_df)
    _write_csv(effect_sizes_dir / "mean_diff_values_all_patients.csv", cohort_outputs.mean_diff_values_df)
    _write_csv(effect_sizes_dir / "mean_diff_dose_gradient_statistics_all_patients.csv", cohort_outputs.mean_diff_grad_stats_df)
    _write_csv(effect_sizes_dir / "mean_diff_dose_gradient_values_all_patients.csv", cohort_outputs.mean_diff_grad_values_df)

    _write_csv(length_scales_dir / "length_scales_dosimetry_statistics_per_biopsy.csv", cohort_outputs.length_scale_per_biopsy_df)
    _write_csv(length_scales_dir / "length_scales_dosimetry_statistics_cohort.csv", cohort_outputs.length_scale_cohort_df)
    _write_csv(
        length_scales_dir / "length_scales_dose_gradient_statistics_per_biopsy.csv",
        cohort_outputs.length_scale_grad_per_biopsy_df,
    )
    _write_csv(
        length_scales_dir / "length_scales_dose_gradient_statistics_cohort.csv",
        cohort_outputs.length_scale_grad_cohort_df,
    )

    _write_csv(delta_corr_dir / "deltas_00_per_voxel_deltas_and_predictors.csv", cohort_outputs.delta_design_df)
    _write_csv(delta_corr_dir / "deltas_01_long_per_voxel_deltas_and_predictors.csv", cohort_outputs.delta_long_design)
    _write_csv(
        delta_corr_dir / "deltas_02a_correlations_log1pDelta_by_delta_kind_and_predictor.csv",
        cohort_outputs.delta_corr_log1p_df,
    )
    _write_csv(
        delta_corr_dir / "deltas_02b_correlations_signedDelta_by_delta_kind_and_predictor.csv",
        cohort_outputs.delta_corr_signed_df,
    )
    _write_csv(
        delta_corr_dir / "deltas_02c_correlations_abs_Delta_by_delta_kind_and_predictor.csv",
        cohort_outputs.delta_corr_abs_df,
    )


def _write_cohort_experimental_tables(csv_dir: Path, cohort_outputs: CohortQAOutputs) -> None:
    experimental_root = csv_dir / "experimental"
    dvh_dir = experimental_root / "dvh_metrics"
    delta_corr_dir = experimental_root / "deltas_bias_correlations"
    sextant_dir = delta_corr_dir / "sextant_summaries"
    for path in [experimental_root, dvh_dir, delta_corr_dir, sextant_dir]:
        path.mkdir(parents=True, exist_ok=True)

    if not cohort_outputs.dvh_metrics_statistics_df.empty:
        _write_csv(
            dvh_dir / "dvh_metrics_statistics_all_patients.csv",
            cohort_outputs.dvh_metrics_statistics_df,
            index=True,
        )

    _write_csv(
        delta_corr_dir / "03c_log1p_medianDelta_vs_top4_predictors__stats.csv",
        cohort_outputs.delta_log1p_stats_df,
    )
    _write_csv(
        delta_corr_dir / "deltas_04a_deltas_biased_correlations_signed.csv",
        cohort_outputs.interdelta_corr_signed_df,
    )
    _write_csv(
        delta_corr_dir / "deltas_04b_deltas_biased_correlations_abs.csv",
        cohort_outputs.interdelta_corr_abs_df,
    )
    _write_csv(
        delta_corr_dir / "deltas_04c_deltas_biased_correlations_log_abs.csv",
        cohort_outputs.interdelta_corr_log_abs_df,
    )
    _write_csv(
        sextant_dir / "sextant_mc_dose_grad_summary.csv",
        cohort_outputs.sextant_mc_dose_grad_df,
    )
    _write_csv(
        sextant_dir / "sextant_mc_trial_deltas_summary.csv",
        cohort_outputs.sextant_mc_deltas_df,
    )
    _write_csv(
        sextant_dir / "sextant_nominal_bias_deltas_summary.csv",
        cohort_outputs.sextant_nominal_deltas_df,
    )


def _write_uncertainty_tables(csv_dir: Path, manifest_dir: Path, qa_data) -> dict[str, Path]:
    outputs = build_uncertainty_summary_outputs(qa_data.common)
    return write_uncertainty_summary_outputs(
        csv_root=csv_dir / "uncertainty_sources",
        manifest_root=manifest_dir,
        outputs=outputs,
    )


def main() -> None:
    pipeline_config = SharedPipelineConfig(
        output_root=Path(__file__).resolve().parent / "output_data_QA",
    )
    output_config = QAOutputConfig(output_root=pipeline_config.output_root)

    write_inventory_csv = True
    write_dvh_csvs = True
    write_path1_csvs = True
    write_cohort_csvs = True
    write_uncertainty_csvs = True
    generate_path1_qa_figures = True
    generate_cohort_qa_figures = True

    print("[main_QA] loading common and QA-specific data")
    qa_data = load_qa_data(pipeline_config)
    dirs = _ensure_dirs(output_config)

    if write_dvh_csvs:
        per_trial_path = dirs["csv"] / "qa_dvh_metrics_per_trial.csv"
        per_biopsy_path = dirs["csv"] / "qa_dvh_metrics_per_biopsy_summary.csv"
        qa_data.calculated_dvh_metrics_per_trial_df.to_csv(per_trial_path, index=False)
        qa_data.cohort_global_dosimetry_dvh_metrics_df.to_csv(per_biopsy_path, index=False)
        print(f"[main_QA] wrote {per_trial_path}")
        print(f"[main_QA] wrote {per_biopsy_path}")

    if write_inventory_csv:
        inventory_df = build_dataframe_inventory(
            qa_data.common,
            extra_frames={
                "calculated_dvh_metrics_per_trial_df": qa_data.calculated_dvh_metrics_per_trial_df,
                "cohort_global_dosimetry_dvh_metrics_df": qa_data.cohort_global_dosimetry_dvh_metrics_df,
            },
        )
        inventory_path = dirs["manifests"] / "qa_table_inventory.csv"
        inventory_df.to_csv(inventory_path, index=False)
        print(f"[main_QA] wrote {inventory_path}")

    if write_uncertainty_csvs:
        uncertainty_paths = _write_uncertainty_tables(dirs["csv"], dirs["manifests"], qa_data)
        print(f"[main_QA] wrote uncertainty summaries under {uncertainty_paths['configured_biopsy'].parent}")

    path1_outputs = None
    if write_path1_csvs or generate_path1_qa_figures:
        print("[main_QA] computing Path-1 QA tables from loaded data")
        path1_outputs = build_path1_qa_outputs(
            qa_data.common,
            qa_data.calculated_dvh_metrics_per_trial_df,
        )
        if write_path1_csvs:
            _write_path1_tables(dirs["csv"], path1_outputs)
            print(f"[main_QA] wrote Path-1 QA tables under {dirs['csv'] / 'qa_path1'}")

    cohort_outputs = None
    if write_cohort_csvs or generate_cohort_qa_figures:
        print("[main_QA] computing cohort QA outputs from loaded data")
        cohort_outputs = build_cohort_qa_outputs(
            qa_data.common,
            qa_data.cohort_global_dosimetry_dvh_metrics_df,
        )
        if write_cohort_csvs:
            _write_cohort_tables(dirs["csv"], qa_data, cohort_outputs)
            _write_cohort_experimental_tables(dirs["csv"], cohort_outputs)
            print(f"[main_QA] wrote cohort QA tables under {dirs['csv']}")

    if generate_path1_qa_figures:
        figures_dir = dirs["figures"] / "qa_path1"
        figures_dir.mkdir(parents=True, exist_ok=True)
        if path1_outputs is None:
            raise RuntimeError("Path-1 outputs were not computed before figure generation.")
        figure_paths = _generate_path1_qa_figures(figures_dir, path1_outputs)
        print(f"[main_QA] wrote {len(figure_paths)} Path-1 figure files to {figures_dir}")

    if generate_cohort_qa_figures:
        if cohort_outputs is None:
            raise RuntimeError("Cohort QA outputs were not computed before figure generation.")
        figure_paths = _generate_cohort_qa_figures(dirs["figures"], qa_data, cohort_outputs)
        print(f"[main_QA] wrote {len(figure_paths)} cohort QA figure files under {dirs['figures']}")

    summary_lines = [
        f"n_biopsies={len(qa_data.common.valid_bx_keys)}",
        f"n_patients_all={len(qa_data.common.unique_patient_ids_all)}",
        f"n_patients_f2={len(qa_data.common.unique_patient_ids_f2)}",
        f"n_voxel_rows={len(qa_data.common.all_voxel_wise_dose_df)}",
        f"n_point_rows={len(qa_data.common.all_point_wise_dose_df)}",
        f"n_dvh_metric_rows={len(qa_data.cohort_global_dosimetry_dvh_metrics_df)}",
    ]
    summary_path = dirs["manifests"] / "qa_run_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[main_QA] wrote {summary_path}")
    print("[main_QA] complete")


if __name__ == "__main__":
    main()
