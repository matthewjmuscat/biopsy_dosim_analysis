from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import helper_funcs
import summary_statistics
from load_data_exemplars import load_exemplar_data
from load_data_shared import build_cumulative_dvh_table, build_dataframe_inventory
from pipeline_shared_config import (
    ExemplarSelectionConfig,
    ExemplarsOutputConfig,
    FigureExportConfig,
    SharedPipelineConfig,
)
from production_plots_exemplars import (
    build_biopsy_heading_map,
    plot_exemplar_axial_profile_pair,
    plot_exemplar_cumulative_dvh_pair,
    plot_exemplar_delta_lines,
    plot_exemplar_length_scale_boxes,
    plot_exemplar_voxel_dualboxes,
    plot_exemplar_voxel_pair_heatmap,
)


def _ensure_dirs(output_config: ExemplarsOutputConfig) -> dict[str, Path]:
    dirs = {
        "root": output_config.output_root,
        "csv": output_config.output_root / output_config.csv_subdir,
        "manifests": output_config.output_root / output_config.manifest_subdir,
        "figures": output_config.output_root / output_config.figures_subdir,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _selected_exemplar_summary(data) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    voxel_df = data.common.all_voxel_wise_dose_df
    point_df = data.common.all_point_wise_dose_df
    biopsy_basic_df = data.common.cohort_biopsy_basic_spatial_features_df
    for exemplar in data.selected_exemplars:
        voxel_sub = voxel_df[
            (voxel_df["Patient ID"] == exemplar.patient_id)
            & (voxel_df["Bx index"].astype(int) == exemplar.bx_index)
        ]
        point_sub = point_df[
            (point_df["Patient ID"] == exemplar.patient_id)
            & (point_df["Bx index"].astype(int) == exemplar.bx_index)
        ]
        biopsy_basic_sub = biopsy_basic_df[
            (biopsy_basic_df["Patient ID"] == exemplar.patient_id)
            & (biopsy_basic_df["Bx index"].astype(int) == exemplar.bx_index)
        ]
        length_mm = float(biopsy_basic_sub.iloc[0]["Length (mm)"]) if not biopsy_basic_sub.empty else float("nan")
        rows.append(
            {
                "Patient ID": exemplar.patient_id,
                "Bx index": exemplar.bx_index,
                "Bx ID": exemplar.bx_id,
                "Display label": exemplar.display_label,
                "Length (mm)": length_mm,
                "n_voxels": int(voxel_sub["Voxel index"].nunique()),
                "n_trials": int(voxel_sub["MC trial num"].nunique()),
                "n_voxel_rows": int(len(voxel_sub)),
                "n_point_rows": int(len(point_sub)),
            }
        )
    return pd.DataFrame(rows)


def _filter_df_to_pairs(df: pd.DataFrame, pairs: list[tuple[str, int]]) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        patient_col = ("Patient ID", "")
        bx_index_col = ("Bx index", "")
    else:
        patient_col = "Patient ID"
        bx_index_col = "Bx index"

    pair_index = pd.MultiIndex.from_tuples(
        [(str(patient_id), int(bx_index)) for patient_id, bx_index in pairs],
        names=["Patient ID", "Bx index"],
    )
    row_index = pd.MultiIndex.from_arrays(
        [
            df[patient_col].astype(str).to_numpy(),
            pd.to_numeric(df[bx_index_col], errors="coerce").astype("Int64").to_numpy(),
        ],
        names=["Patient ID", "Bx index"],
    )
    return df.loc[row_index.isin(pair_index)].copy()


def _replace_bx_ids_with_labels(
    df: pd.DataFrame,
    biopsy_label_map: dict[tuple[str, int], str],
) -> pd.DataFrame:
    out = df.copy()
    out["Bx ID"] = [
        biopsy_label_map.get((str(patient_id), int(bx_index)), str(bx_id))
        for patient_id, bx_index, bx_id in zip(out["Patient ID"], out["Bx index"], out["Bx ID"])
    ]
    return out


def _generate_selected_exemplar_figures(
    data,
    figures_dir: Path,
    selection_config: ExemplarSelectionConfig,
) -> list[Path]:
    selected_pairs = [(item.patient_id, item.bx_index) for item in data.selected_exemplars]
    biopsy_label_map = build_biopsy_heading_map(
        selected_pairs,
        explicit_map=selection_config.display_label_map,
    )
    export_config = FigureExportConfig()

    selected_voxel_df = _filter_df_to_pairs(data.common.all_voxel_wise_dose_df, selected_pairs)
    selected_point_df = _filter_df_to_pairs(data.common.all_point_wise_dose_df, selected_pairs)
    selected_global_by_voxel_df = _filter_df_to_pairs(
        data.common.cohort_global_dosimetry_by_voxel_df,
        selected_pairs,
    )
    selected_shifts_df = data.common.all_mc_structure_transformation_df[
        data.common.all_mc_structure_transformation_df["Patient ID"].astype(str).isin(
            [str(patient_id) for patient_id, _ in selected_pairs]
        )
    ].copy()

    print(
        "[main_exemplars] building selected-biopsy figure inputs "
        f"from {len(selected_voxel_df)} voxel-wise rows"
    )
    mc_deltas = summary_statistics.compute_mc_trial_deltas_with_abs(selected_voxel_df)
    nominal_dose_deltas_df = summary_statistics.compute_biopsy_nominal_deltas_with_abs(
        selected_global_by_voxel_df,
        zero_level_index_str="Dose (Gy)",
    )
    nominal_gradient_deltas_df = summary_statistics.compute_biopsy_nominal_deltas_with_abs(
        selected_global_by_voxel_df,
        zero_level_index_str="Dose grad (Gy/mm)",
    )

    print("[main_exemplars] computing selected-biopsy voxel pairing statistics")
    dose_pair_stats_df, _, _, _ = helper_funcs.create_diff_stats_dataframe(
        selected_voxel_df,
        patient_id_col="Patient ID",
        bx_index_col="Bx index",
        bx_id_col="Bx ID",
        voxel_index_col="Voxel index",
        dose_col="Dose (Gy)",
    )
    grad_pair_stats_df, _, _, _ = helper_funcs.create_diff_stats_dataframe(
        selected_voxel_df,
        patient_id_col="Patient ID",
        bx_index_col="Bx index",
        bx_id_col="Bx ID",
        voxel_index_col="Voxel index",
        dose_col="Dose grad (Gy/mm)",
    )
    dose_pair_stats_df = _replace_bx_ids_with_labels(dose_pair_stats_df, biopsy_label_map)
    grad_pair_stats_df = _replace_bx_ids_with_labels(grad_pair_stats_df, biopsy_label_map)

    print("[main_exemplars] computing selected-biopsy length-scale summaries")
    dose_length_scale_df = helper_funcs.compute_dose_differences_vectorized(
        selected_voxel_df,
        column_name="Dose (Gy)",
    )
    print("[main_exemplars] building selected-biopsy cumulative DVH curves")
    selected_cumulative_dvh_df = build_cumulative_dvh_table(selected_voxel_df)

    figure_paths: list[Path] = []
    figure_paths.extend(
        plot_exemplar_axial_profile_pair(
            selected_point_df,
            selected_shifts_df,
            biopsies=selected_pairs,
            save_dir=figures_dir,
            file_stem="Fig_exemplars_axial_dose_pair",
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            value_col="Dose (Gy)",
            y_label=r"Dose along core $D_b(z)$ (Gy)",
            num_trials_to_show=3,
        )
    )
    figure_paths.extend(
        plot_exemplar_axial_profile_pair(
            selected_point_df,
            selected_shifts_df,
            biopsies=selected_pairs,
            save_dir=figures_dir,
            file_stem="Fig_exemplars_axial_gradient_pair",
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            value_col="Dose grad (Gy/mm)",
            y_label=r"Dose-gradient magnitude $G_b(z)$ (Gy mm$^{-1}$)",
            num_trials_to_show=3,
        )
    )
    figure_paths.extend(
        plot_exemplar_cumulative_dvh_pair(
            selected_cumulative_dvh_df,
            selected_shifts_df,
            biopsies=selected_pairs,
            save_dir=figures_dir,
            file_stem="Fig_exemplars_cumulative_dvh_pair",
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            num_trials_to_show=3,
        )
    )
    figure_paths.extend(
        plot_exemplar_delta_lines(
            nominal_dose_deltas_df,
            biopsies=selected_pairs,
            save_dir=figures_dir,
            fig_name="Fig_exemplars_dose_delta_overlay_with_abs",
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            zero_level_index_str="Dose (Gy)",
            x_axis="Voxel index",
            linewidth_signed=2.0,
            linewidth_abs=3.2,
            show_markers=True,
            include_abs=True,
            legend_fontsize=export_config.legend_fontsize,
        )
    )
    figure_paths.extend(
        plot_exemplar_delta_lines(
            nominal_gradient_deltas_df,
            biopsies=selected_pairs,
            save_dir=figures_dir,
            fig_name="Fig_exemplars_gradient_delta_overlay_with_abs",
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            zero_level_index_str="Dose grad (Gy/mm)",
            x_axis="Voxel index",
            linewidth_signed=2.0,
            linewidth_abs=3.2,
            show_markers=True,
            include_abs=True,
            legend_fontsize=export_config.legend_fontsize,
        )
    )
    figure_paths.extend(
        plot_exemplar_voxel_dualboxes(
            mc_deltas,
            biopsies=selected_pairs,
            output_dir=figures_dir,
            plot_name_base="Fig_exemplars_voxel_dualboxes_dose",
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            metric="Dose (Gy)",
            x_axis="Voxel index",
            lane_gap=2.0,
            box_width=0.32,
            pair_gap=0.10,
            biopsy_gap=0.22,
            show_points=False,
            whisker_mode="q05q95",
            showfliers=False,
        )
    )
    figure_paths.extend(
        plot_exemplar_voxel_dualboxes(
            mc_deltas,
            biopsies=selected_pairs,
            output_dir=figures_dir,
            plot_name_base="Fig_exemplars_voxel_dualboxes_gradient",
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            metric="Dose grad (Gy/mm)",
            x_axis="Voxel index",
            lane_gap=2.0,
            box_width=0.32,
            pair_gap=0.10,
            biopsy_gap=0.22,
            show_points=False,
            whisker_mode="q05q95",
            showfliers=False,
        )
    )
    figure_paths.extend(
        plot_exemplar_length_scale_boxes(
            dose_length_scale_df,
            save_dir=figures_dir,
            file_name="Fig_exemplars_length_scale_dose_abs",
            export_config=export_config,
            title=None,
            figsize=(10, 6),
            show_points=False,
            violin_or_box="box",
            trend_lines=["mean"],
            annotate_counts=True,
            annotation_box=False,
            y_trim=True,
            y_min_fixed=0,
            xlabel=None,
            ylabel=None,
            multi_pairs=selected_pairs,
            metric_family="dose",
        )
    )
    figure_paths.extend(
        plot_exemplar_voxel_pair_heatmap(
            upper_df=dose_pair_stats_df,
            lower_df=grad_pair_stats_df,
            save_dir=figures_dir,
            save_name_base="Fig_exemplars_voxel_pair_heatmap_abs",
            export_config=export_config,
            upper_mean_col="mean_abs_diff",
            upper_std_col=None,
            lower_mean_col="mean_abs_diff",
            lower_std_col=None,
            vmin_upper=0.0,
            vmin_lower=0.0,
            cmap="Reds",
            cbar_label_upper=r"$\overline{|M_{b,ij}^{D}|}$ (Gy, upper)",
            cbar_label_lower=r"$\overline{|M_{b,ij}^{G}|}$ (Gy mm$^{-1}$, lower)",
            show_title=False,
            show_annotation_box=True,
            cell_annot_fontsize=export_config.annotation_fontsize,
            cell_value_decimals=1,
        )
    )
    return figure_paths


def main() -> None:
    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------
    pipeline_config = SharedPipelineConfig(
        output_root=Path(__file__).resolve().parent / "output_data_exemplars",
    )
    selection_config = ExemplarSelectionConfig()
    output_config = ExemplarsOutputConfig(output_root=pipeline_config.output_root)

    write_inventory_csv = True
    write_selection_manifest = True
    write_dvh_metric_csvs = True
    generate_selected_exemplar_figures = True

    print("[main_exemplars] loading common and exemplar-specific data")
    exemplar_data = load_exemplar_data(
        pipeline_config,
        selection_config,
        build_supporting_dvh_tables=write_dvh_metric_csvs,
        build_cumulative_dvh_table_from_voxels=False,
    )
    dirs = _ensure_dirs(output_config)

    if (
        write_dvh_metric_csvs
        and exemplar_data.calculated_dvh_metrics_per_trial_df is not None
        and exemplar_data.cohort_global_dosimetry_dvh_metrics_df is not None
    ):
        per_trial_path = dirs["csv"] / "exemplar_dvh_metrics_per_trial.csv"
        per_biopsy_path = dirs["csv"] / "exemplar_dvh_metrics_per_biopsy_summary.csv"
        exemplar_data.calculated_dvh_metrics_per_trial_df.to_csv(per_trial_path, index=False)
        exemplar_data.cohort_global_dosimetry_dvh_metrics_df.to_csv(per_biopsy_path, index=False)
        print(f"[main_exemplars] wrote {per_trial_path}")
        print(f"[main_exemplars] wrote {per_biopsy_path}")

    if write_selection_manifest:
        heading_map = build_biopsy_heading_map(
            [(item.patient_id, item.bx_index) for item in exemplar_data.selected_exemplars],
            explicit_map=selection_config.display_label_map,
        )
        selection_df = _selected_exemplar_summary(exemplar_data)
        selection_df["Generated heading"] = selection_df.apply(
            lambda row: heading_map[(str(row["Patient ID"]), int(row["Bx index"]))],
            axis=1,
        )
        selection_path = dirs["manifests"] / "selected_exemplars.csv"
        selection_df.to_csv(selection_path, index=False)
        print(f"[main_exemplars] wrote {selection_path}")

    if write_inventory_csv:
        extra_frames: dict[str, pd.DataFrame] = {}
        if exemplar_data.calculated_dvh_metrics_per_trial_df is not None:
            extra_frames["calculated_dvh_metrics_per_trial_df"] = exemplar_data.calculated_dvh_metrics_per_trial_df
        if exemplar_data.cohort_global_dosimetry_dvh_metrics_df is not None:
            extra_frames["cohort_global_dosimetry_dvh_metrics_df"] = exemplar_data.cohort_global_dosimetry_dvh_metrics_df
        if exemplar_data.all_cumulative_dvh_by_mc_trial_number_df is not None:
            extra_frames["all_cumulative_dvh_by_mc_trial_number_df"] = exemplar_data.all_cumulative_dvh_by_mc_trial_number_df
        inventory_df = build_dataframe_inventory(
            exemplar_data.common,
            extra_frames=extra_frames,
        )
        inventory_path = dirs["manifests"] / "exemplars_table_inventory.csv"
        inventory_df.to_csv(inventory_path, index=False)
        print(f"[main_exemplars] wrote {inventory_path}")

    if generate_selected_exemplar_figures:
        figures_dir = dirs["figures"] / "selected_exemplars"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figure_paths = _generate_selected_exemplar_figures(
            exemplar_data,
            figures_dir,
            selection_config,
        )
        print(f"[main_exemplars] wrote {len(figure_paths)} figure files to {figures_dir}")

    summary_lines = [
        f"n_selected_exemplars={len(exemplar_data.selected_exemplars)}",
        f"n_patients_all={len(exemplar_data.common.unique_patient_ids_all)}",
        f"n_patients_f2={len(exemplar_data.common.unique_patient_ids_f2)}",
        f"n_voxel_rows={len(exemplar_data.common.all_voxel_wise_dose_df)}",
        f"n_point_rows={len(exemplar_data.common.all_point_wise_dose_df)}",
        "n_cumulative_dvh_rows="
        + (
            str(len(exemplar_data.all_cumulative_dvh_by_mc_trial_number_df))
            if exemplar_data.all_cumulative_dvh_by_mc_trial_number_df is not None
            else "not_built"
        ),
    ]
    summary_path = dirs["manifests"] / "exemplars_run_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[main_exemplars] wrote {summary_path}")
    print("[main_exemplars] complete")


if __name__ == "__main__":
    main()
