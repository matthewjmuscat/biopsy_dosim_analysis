from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd

import GPR_analysis_helpers as gpr_helpers
import GPR_production_plots as gpr_plots
import GPR_calibration


def run_kernel_sensitivity(
    all_voxel_wise_dose_df: pd.DataFrame,
    semivariogram_df: pd.DataFrame,
    output_dir: Path,
    figs_dir: Path,
    csv_dir: Path,
    target_stat: str = "median",
    kernel_specs: Iterable[Tuple[str, float | None, str]] | None = None,
    file_types: tuple[str, ...] = ("pdf", "svg"),
    position_mode: str = "center",
    kernel_color_map: dict[str, str] | None = None,
):
    """
    Run the GP+metrics pipeline for a list of kernels and aggregate results.

    Parameters
    ----------
    all_voxel_wise_dose_df : pd.DataFrame
        Trial-wise voxel table (already filtered by Simulated type).
    semivariogram_df : pd.DataFrame
        Empirical semivariograms for the same filtered cohort.
    output_dir : Path
        Base output directory (kernel-specific CSVs and figures are written here).
    target_stat : str
        "median" (default) or "mean" voxel statistic for GP targets.
    kernel_specs : iterable of (kernel_name, param, label)
        Example default (when None):
            [
                ("matern", 1.5, "matern_nu_1_5"),
                ("matern", 2.5, "matern_nu_2_5"),
            ]
        Additional kernels like ("rbf", None, "rbf") supported if enabled.
    file_types : tuple[str,...]
        File extensions to save plots.
    """

    if kernel_specs is None:
        kernel_specs = [
            ("matern", 1.5, "matern_nu_1_5"),
            ("matern", 2.5, "matern_nu_2_5"),
            ("rbf", None, "rbf"),
            ("exp", None, "exp"),
        ]

    kernel_specs = list(kernel_specs)

    all_metrics = []
    all_calib = []

    for kernel_name, kernel_param, kernel_label in kernel_specs:
        print(f"Running kernel sensitivity for {kernel_label} ...")
        results, metrics_df, cohort_summary_df, by_patient = gpr_helpers.run_gp_and_collect_metrics(
            all_voxel_wise_dose_df=all_voxel_wise_dose_df,
            semivariogram_df=semivariogram_df,
            output_dir=output_dir,
            target_stat=target_stat,
            nu=kernel_param if kernel_name == "matern" else None,
            kernel_spec=(kernel_name, kernel_param),
            kernel_label=kernel_label,
            position_mode=position_mode,
            save_csv=False,
        )

        # Save kernel-specific metrics
        metrics_path = csv_dir / f"metrics_kernel_{kernel_label}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved kernel metrics to {metrics_path}")

        by_patient_path = csv_dir / f"patient_rollup_{kernel_label}.csv"
        by_patient.to_csv(by_patient_path, index=False)

        cohort_summary_path = csv_dir / f"cohort_summary_{kernel_label}.csv"
        cohort_summary_df.to_csv(cohort_summary_path, index=False)

        # Calibration metrics and figures per kernel
        calib_df = GPR_calibration.build_calibration_metrics(
            results,
            mean_bounds=(-1.0, 1.0),
            sd_bounds=(0.5, 1.5),
        )
        calib_df["kernel_label"] = kernel_label
        calib_csv = csv_dir / f"calibration_metrics_{kernel_label}.csv"
        calib_df.to_csv(calib_csv, index=False)
        calib_fig_dir = figs_dir / f"calibration_{kernel_label}"
        calib_fig_dir.mkdir(parents=True, exist_ok=True)
        gpr_plots.calibration_plots_production(
            calib_df=calib_df,
            save_dir=calib_fig_dir,
            save_formats=file_types,
            mean_bounds=(-1.0, 1.0),
            sd_bounds=(0.5, 1.5),
            modes_list=[("histogram",), ("histogram", "kde"), ("kde",)],
            kernel_color_map=kernel_color_map,
        )
        print(f"Saved calibration metrics to {calib_csv} and figures to {calib_fig_dir}")

        all_metrics.append(metrics_df)
        all_calib.append(calib_df)

    if not all_metrics:
        print("No kernel specs provided; nothing to do.")
        return None

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_path = csv_dir / "metrics_kernel_all.csv"
    combined_metrics.to_csv(combined_path, index=False)
    print(f"Saved combined kernel metrics to {combined_path}")

    # Combined calibration overlays across kernels
    try:
        if all_calib:
            combined_calib = pd.concat(all_calib, ignore_index=True)
            calib_all_dir = figs_dir / "calibration_all"
            calib_all_dir.mkdir(parents=True, exist_ok=True)
            gpr_plots.calibration_plots_production(
                calib_df=combined_calib,
                save_dir=calib_all_dir,
                save_formats=file_types,
                mean_bounds=(-1.0, 1.0),
                sd_bounds=(0.5, 1.5),
                modes_list=[("histogram",), ("histogram", "kde"), ("kde",)],
                hue_col="kernel_label",
                kernel_color_map=kernel_color_map,
            )
            combined_calib_path = csv_dir / "calibration_metrics_all.csv"
            combined_calib.to_csv(combined_calib_path, index=False)
            print(f"Saved combined calibration metrics to {combined_calib_path} and overlay figures to {calib_all_dir}")
    except Exception as e:
        print(f"Warning: could not generate combined calibration overlays: {e}")

    # Plots
    mode_list = [("histogram",), ("histogram", "kde"), ("kde",)]

    try:
        for plot_type in mode_list:
            suffix = "_".join(plot_type)
            gpr_plots.plot_kernel_sensitivity_histogram(
                combined_metrics,
                value_col="ell",
                y_label=r"$\ell$ (mm)",
                save_dir=figs_dir,
                file_name_base=f"kernel_sensitivity_ell_{suffix}",
                file_types=file_types,
                show_title=False,
                modes=plot_type,
                kde_bw_scale=None, # it will use Scott's rule by default
                legend_fontsize =12,
                kernel_color_map=kernel_color_map,
            )
        print("Plotted kernel sensitivity ell hist/KDE. Save figures to", figs_dir)
    except Exception as e:
        print(f"Warning: could not plot ell histogram: {e}")

    try:
        for plot_type in mode_list:
            suffix = "_".join(plot_type)
            gpr_plots.plot_kernel_sensitivity_histogram(
                combined_metrics,
                value_col="mean_ratio",
                y_label="Mean uncertainty reduction ratio",
                save_dir=figs_dir,
                file_name_base=f"kernel_sensitivity_mean_ratio_{suffix}",
                file_types=file_types,
                show_title=False,
                modes=plot_type,
                kde_bw_scale=None, # it will use Scott's rule by default
                legend_fontsize =12,     
                kernel_color_map=kernel_color_map,
            )
        print("Plotted kernel sensitivity mean_ratio hist/KDE. Save figures to", figs_dir)
    except Exception as e:
        print(f"Warning: could not plot mean_ratio histogram: {e}")

    try:
        gpr_plots.plot_kernel_sensitivity_scatter(
            combined_metrics,
            x_col="mean_ratio",
            y_col="integ_ratio",
            x_label="Mean voxelwise ratio",
            y_label="Integrated SD ratio",
            save_dir=figs_dir,
            file_name_base="kernel_sensitivity_ratio_scatter",
            file_types=file_types,
            show_title=False,
            kernel_color_map=kernel_color_map,
        )
        print("Plotted kernel sensitivity ratio scatter plot. Save figures to", figs_dir)
    except Exception as e:
        print(f"Warning: could not plot scatter: {e}")

    try:
        gpr_plots.plot_kernel_sensitivity_mean_sd_with_fits(
            combined_metrics,
            save_dir=figs_dir,
            file_name_base="kernel_sensitivity_mean_sd_scatter_with_fits",
            file_types=file_types,
            kernel_color_map=kernel_color_map,
        )
        print("Plotted kernel sensitivity mean/SD fits. Save figures to", figs_dir)
    except Exception as e:
        print(f"Warning: could not plot kernel sensitivity mean/SD fits: {e}")

    try:
        for plot_type in mode_list:
            suffix = "_".join(plot_type)
            gpr_plots.plot_kernel_sensitivity_histogram(
                combined_metrics,
                value_col="sv_rmse",
                y_label=r"Semivariogram RMSE",
                save_dir=figs_dir,
                file_name_base=f"kernel_sensitivity_sv_rmse_{suffix}",
                file_types=file_types,
                show_title=False,
                modes=plot_type,
                kde_bw_scale=None, # it will use Scott's rule by default
                legend_fontsize =12,
            )
        print("Plotted kernel sensitivity sv_rmse hist/KDE. Save figures to", figs_dir)
    except Exception as e:
        print(f"Warning: could not plot sv_rmse histogram: {e}")

    return combined_metrics
