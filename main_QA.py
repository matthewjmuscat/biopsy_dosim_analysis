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


def _load_existing_path1_inputs(base_dir: Path) -> dict[str, pd.DataFrame]:
    file_map = {
        "path1_results_df": base_dir / "Cohort_QA_Path1_biopsy_threshold_probabilities_with_z.csv",
        "coef_margin_df": base_dir / "2_logit_margin" / "p1_logit_margin_01_coef.csv",
        "coef_secondary_df": base_dir / "6_secondary_scan" / "p1_secscan_01_coef_margin_plus_all_secondaries.csv",
        "pred_secondary_df": base_dir / "6_secondary_scan" / "p1_secscan_02_predictions_margin_plus_all_secondaries.csv",
        "best_secondary_df": base_dir / "6_secondary_scan" / "p1_secscan_05_best_secondary_per_threshold.csv",
    }
    missing = [str(path) for path in file_map.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required legacy Path-1 QA CSV inputs for the new QA plot lane: "
            + ", ".join(missing)
        )
    return {name: pd.read_csv(path) for name, path in file_map.items()}


def _generate_path1_qa_figures(figures_dir: Path) -> list[Path]:
    qa_seed_dir = Path(__file__).resolve().parent / "output_data" / "qa_path1"
    inputs = _load_existing_path1_inputs(qa_seed_dir)
    export_config = FigureExportConfig()

    figure_paths: list[Path] = []
    figure_paths.extend(
        plot_path1_threshold_qa_summary(
            inputs["path1_results_df"],
            save_dir=figures_dir,
            export_config=export_config,
            file_stem="Fig_Path1_threshold_QA_summary_v2",
        )
    )
    figure_paths.extend(
        plot_path1_p_pass_vs_margin(
            inputs["path1_results_df"],
            save_dir=figures_dir,
            export_config=export_config,
            file_stem="Fig_Path1_p_pass_vs_margin_by_metric",
            coef_df=inputs["coef_margin_df"],
            show_required_margin_line=True,
            required_prob=0.95,
        )
    )
    figure_paths.extend(
        plot_path1_best_secondary_families(
            inputs["pred_secondary_df"],
            inputs["coef_secondary_df"],
            save_dir=figures_dir,
            export_config=export_config,
            file_stem="Fig_Path1_logit_margin_plus_best_secondary_families",
            comparison_df=inputs["best_secondary_df"],
            overlay_1d_model=True,
            coef1_df=inputs["coef_margin_df"],
        )
    )
    return figure_paths


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

    if generate_path1_qa_figures:
        figures_dir = dirs["figures"] / "qa_path1"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figure_paths = _generate_path1_qa_figures(figures_dir)
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
