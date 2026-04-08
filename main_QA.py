from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

from load_data_QA import load_qa_data
from load_data_shared import build_dataframe_inventory
from pipeline_shared_config import FigureExportConfig, QAOutputConfig, SharedPipelineConfig
from production_plots_QA import (
    plot_path1_best_secondary_families,
    plot_path1_p_pass_vs_margin,
    plot_path1_threshold_qa_summary,
)
from qa_path1_pipeline import Path1QAOutputs, build_path1_qa_outputs


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


def _write_path1_tables(csv_dir: Path, path1_outputs: Path1QAOutputs) -> None:
    path1_root = csv_dir / "qa_path1"
    core_dir = path1_root / "1_core"
    margin_dir = path1_root / "2_logit_margin"
    grad_dir = path1_root / "3_logit_grad"
    secondary_dir = path1_root / "6_secondary_scan"
    for path in [path1_root, core_dir, margin_dir, grad_dir, secondary_dir]:
        path.mkdir(parents=True, exist_ok=True)

    path1_outputs.path1_results_df.to_csv(
        path1_root / "Cohort_QA_Path1_biopsy_threshold_probabilities_with_z.csv",
        index=False,
    )
    path1_outputs.path1_results_df.to_csv(
        core_dir / "p1_core_01_biopsy_mc_probs_z.csv",
        index=False,
    )
    path1_outputs.path1_enriched_df.to_csv(
        core_dir / "p1_core_02_biopsy_probs_plus_nominal_predictors.csv",
        index=False,
    )
    path1_outputs.threshold_summary_df.to_csv(
        core_dir / "p1_core_03_threshold_summary_by_rule.csv",
        index=False,
    )
    path1_outputs.coef_margin_df.to_csv(
        margin_dir / "p1_logit_margin_01_coef.csv",
        index=False,
    )
    path1_outputs.pred_margin_df.to_csv(
        margin_dir / "p1_logit_margin_02_predictions.csv",
        index=False,
    )
    path1_outputs.coef_gradient_df.to_csv(
        grad_dir / "p1_logit_grad_01_coef.csv",
        index=False,
    )
    path1_outputs.pred_gradient_df.to_csv(
        grad_dir / "p1_logit_grad_02_predictions.csv",
        index=False,
    )
    path1_outputs.coef_secondary_df.to_csv(
        secondary_dir / "p1_secscan_01_coef_margin_plus_all_secondaries.csv",
        index=False,
    )
    path1_outputs.pred_secondary_df.to_csv(
        secondary_dir / "p1_secscan_02_predictions_margin_plus_all_secondaries.csv",
        index=False,
    )
    path1_outputs.model_compare_secondary_df.to_csv(
        secondary_dir / "p1_secscan_03_model_compare_all_vs_margin_raw.csv",
        index=False,
    )
    path1_outputs.best_secondary_df.to_csv(
        secondary_dir / "p1_secscan_05_best_secondary_per_threshold.csv",
        index=False,
    )


def main() -> None:
    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------
    pipeline_config = SharedPipelineConfig(
        output_root=Path(__file__).resolve().parent / "output_data_QA",
    )
    output_config = QAOutputConfig(output_root=pipeline_config.output_root)

    write_inventory_csv = True
    write_dvh_csvs = True
    write_path1_csvs = True
    generate_path1_qa_figures = True

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

    if generate_path1_qa_figures:
        figures_dir = dirs["figures"] / "qa_path1"
        figures_dir.mkdir(parents=True, exist_ok=True)
        if path1_outputs is None:
            raise RuntimeError("Path-1 outputs were not computed before figure generation.")
        figure_paths = _generate_path1_qa_figures(figures_dir, path1_outputs)
        print(f"[main_QA] wrote {len(figure_paths)} figure files to {figures_dir}")

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
