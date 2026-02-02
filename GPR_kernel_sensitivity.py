from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd

import GPR_analysis_helpers as gpr_helpers
import GPR_production_plots as gpr_plots


def run_kernel_sensitivity(
    all_voxel_wise_dose_df: pd.DataFrame,
    semivariogram_df: pd.DataFrame,
    output_dir: Path,
    target_stat: str = "median",
    kernel_specs: Iterable[Tuple[str, float | None, str]] | None = None,
    file_types: tuple[str, ...] = ("pdf", "svg"),
    position_mode: str = "center",
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
        ]

    kernel_specs = list(kernel_specs)

    all_metrics = []
    output_dir = Path(output_dir)
    figs_dir = output_dir / "figures"
    csv_dir = output_dir / "csv"
    figs_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    for kernel_name, kernel_param, kernel_label in kernel_specs:
        print(f"Running kernel sensitivity for {kernel_label} ...")
        _, metrics_df, cohort_summary_df, by_patient = gpr_helpers.run_gp_and_collect_metrics(
            all_voxel_wise_dose_df=all_voxel_wise_dose_df,
            semivariogram_df=semivariogram_df,
            output_dir=output_dir,
            target_stat=target_stat,
            nu=kernel_param if kernel_name == "matern" else 1.5,
            kernel_spec=(kernel_name, kernel_param),
            kernel_label=kernel_label,
            position_mode=position_mode,
        )

        # Save kernel-specific metrics
        metrics_path = csv_dir / f"metrics_kernel_{kernel_label}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved kernel metrics to {metrics_path}")

        by_patient_path = csv_dir / f"patient_rollup_{kernel_label}.csv"
        by_patient.to_csv(by_patient_path, index=False)

        cohort_summary_path = csv_dir / f"cohort_summary_{kernel_label}.csv"
        cohort_summary_df.to_csv(cohort_summary_path, index=False)

        all_metrics.append(metrics_df)

    if not all_metrics:
        print("No kernel specs provided; nothing to do.")
        return None

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_path = csv_dir / "metrics_kernel_all.csv"
    combined_metrics.to_csv(combined_path, index=False)
    print(f"Saved combined kernel metrics to {combined_path}")

    # Plots
    try:
        gpr_plots.plot_kernel_sensitivity_boxplot(
            combined_metrics,
            value_col="ell",
            y_label=r"$\ell$ (mm)",
            save_dir=figs_dir,
            file_name_base="kernel_sensitivity_ell",
            file_types=file_types,
            show_title=False,
        )
    except Exception as e:
        print(f"Warning: could not plot ell boxplot: {e}")

    try:
        gpr_plots.plot_kernel_sensitivity_boxplot(
            combined_metrics,
            value_col="mean_ratio",
            y_label="Mean uncertainty reduction ratio",
            save_dir=figs_dir,
            file_name_base="kernel_sensitivity_mean_ratio",
            file_types=file_types,
            show_title=False,
        )
    except Exception as e:
        print(f"Warning: could not plot mean_ratio boxplot: {e}")

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
        )
    except Exception as e:
        print(f"Warning: could not plot scatter: {e}")

    return combined_metrics
