from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl

import production_plots
from pipeline_shared_config import FigureExportConfig
from qa_cohort_pipeline import (
    default_delta_kind_label_map,
    default_delta_predictor_cols,
    default_delta_predictor_label_map,
    default_zero_x_predictors,
)


def _default_path1_best_secondary_plot_metadata():
    return {
        "per_label_legend_title": {
            "D2 ≥ 32 Gy": r"Levels of $\hat{g}_1$",
            "D50 ≥ 27 Gy": r"Levels of $\hat{g}_2$",
            "D98 ≥ 20 Gy": r"Levels of $\hat{g}_3$",
            "V150 ≥ 50%": r"Levels of $\hat{g}_4$",
        },
        "per_label_grad_label_template": {
            "D2 ≥ 32 Gy": r"$\hat{{g}}_1 = {value:.2f}\ {unit}$",
            "D50 ≥ 27 Gy": r"$\hat{{g}}_2 = {value:.2f}$",
            "D98 ≥ 20 Gy": r"$\hat{{g}}_3 = {value:.2f}\ {unit}$",
            "V150 ≥ 50%": r"$\hat{{g}}_4 = {value:.2f}\ {unit}$",
        },
        "per_label_secondary_unit": {
            "D2 ≥ 32 Gy": r"\mathrm{Gy\ mm^{-1}}",
            "D50 ≥ 27 Gy": "",
            "D98 ≥ 20 Gy": r"\mathrm{Gy\ mm^{-1}}",
            "V150 ≥ 50%": r"\mathrm{mm}",
        },
        "per_label_secondary_annotation": {
            "D2 ≥ 32 Gy": (
                r"Secondary predictor: "
                r"$\hat{g}_1 = \overline{G}^{(0)}_b$" "\n"
                r"(core nominal dose gradient, $\mathrm{Gy\ mm^{-1}}$)"
            ),
            "D50 ≥ 27 Gy": (
                r"Secondary predictor: "
                r"$\hat{g}_2 = \mathrm{SphDisp}_{\mathrm{DIL}}$" "\n"
                r"(DIL spherical disproportion, dimensionless)"
            ),
            "D98 ≥ 20 Gy": (
                r"Secondary predictor: "
                r"$\hat{g}_3 = \overline{G}^{(0)}_b$" "\n"
                r"(core nominal dose gradient, $\mathrm{Gy\ mm^{-1}}$)"
            ),
            "V150 ≥ 50%": (
                r"Secondary predictor: "
                r"$\hat{g}_4 = \overline{d}^{\mathrm{NN}}_{\mathrm{R}}$" "\n"
                r"(rectum mean NN distance, $\mathrm{mm}$)"
            ),
        },
        "per_label_stats_box_xy": {
            "V150 ≥ 50%": (0.02, 0.60),
        },
    }


def _default_file_name(file_stem: str, export_config: FigureExportConfig) -> str:
    primary_ext = str(export_config.save_formats[0]).lstrip(".")
    return f"{file_stem}.{primary_ext}"


def _expected_export_paths(save_dir: str | Path, file_stem: str, export_config: FigureExportConfig) -> list[Path]:
    save_dir = Path(save_dir)
    return [save_dir / f"{file_stem}.{fmt.lstrip('.')}" for fmt in export_config.save_formats]


@contextmanager
def _font_rc(export_config: FigureExportConfig):
    with mpl.rc_context(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "axes.unicode_minus": True,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelsize": export_config.axes_label_fontsize,
            "xtick.labelsize": export_config.tick_label_fontsize,
            "ytick.labelsize": export_config.tick_label_fontsize,
            "legend.fontsize": export_config.legend_fontsize,
            "axes.titlesize": export_config.title_fontsize,
        }
    ):
        yield


def plot_path1_threshold_qa_summary(
    df,
    save_dir: str | Path,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    file_stem: str = "Fig_Path1_threshold_QA_summary",
    file_name: str | None = None,
    **kwargs: Any,
):
    file_name = file_name or _default_file_name(file_stem, export_config)
    kwargs.setdefault("show_title", False)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("annotate_percents", True)
    kwargs.setdefault("dpi", export_config.dpi)
    kwargs.setdefault("axis_label_fontsize", export_config.axes_label_fontsize + 1)
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize + 1)
    kwargs.setdefault("legend_fontsize", export_config.legend_fontsize + 1)
    kwargs.setdefault("annotation_fontsize", export_config.annotation_fontsize + 2)
    kwargs.setdefault("panel_letter_fontsize", export_config.title_fontsize + 1)
    with _font_rc(export_config):
        return production_plots.production_plot_path1_threshold_qa_summary_v2(
            df,
            save_dir=save_dir,
            file_name=file_name,
            save_formats=export_config.save_formats,
            **kwargs,
        )


def plot_path1_p_pass_vs_margin(
    df,
    save_dir: str | Path,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    file_stem: str = "Fig_Path1_p_pass_vs_margin_by_metric",
    file_name: str | None = None,
    **kwargs: Any,
):
    file_name = file_name or _default_file_name(file_stem, export_config)
    kwargs.setdefault("show_panel_titles", False)
    kwargs.setdefault("annotate_fit_stats", True)
    kwargs.setdefault("fit_stats", ("n", "mcfadden_r2", "wrmse"))
    kwargs.setdefault("dpi", export_config.dpi)
    kwargs.setdefault("axis_label_fontsize", export_config.axes_label_fontsize)
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("panel_title_fontsize", export_config.axes_label_fontsize + 1)
    kwargs.setdefault("legend_fontsize", export_config.legend_fontsize + 1)
    kwargs.setdefault("panel_letter_fontsize", export_config.title_fontsize + 1)
    kwargs.setdefault("panel_letter_y", 1.12)
    kwargs.setdefault("xlabel_rule_second_line", True)
    kwargs.setdefault("fit_stats_fontsize", export_config.annotation_fontsize)
    kwargs.setdefault("label_required_margin_line", True)
    kwargs.setdefault("required_margin_label_fontsize", export_config.annotation_fontsize)
    kwargs.setdefault("required_margin_label_y", 0.50)
    kwargs.setdefault("include_required_margin_in_fit_box", False)
    with _font_rc(export_config):
        return production_plots.production_plot_path1_p_pass_vs_margin_by_metric(
            df,
            save_dir=save_dir,
            file_name=file_name,
            save_formats=export_config.save_formats,
            **kwargs,
        )


def plot_path1_best_secondary_families(
    pred_df,
    coef_df,
    save_dir: str | Path,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    file_stem: str = "Fig_Path1_logit_margin_plus_best_secondary_families",
    file_name: str | None = None,
    **kwargs: Any,
):
    file_name = file_name or _default_file_name(file_stem, export_config)
    comparison_df = kwargs.get("comparison_df")
    is_best_secondary_plot = comparison_df is not None and hasattr(comparison_df, "columns") and "secondary_predictor" in comparison_df.columns

    kwargs.setdefault("dpi", export_config.dpi)
    kwargs.setdefault("axis_label_fontsize", export_config.axes_label_fontsize - 1)
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault(
        "panel_title_fontsize",
        export_config.annotation_fontsize + 1 if is_best_secondary_plot else export_config.axes_label_fontsize + 1,
    )
    kwargs.setdefault("global_legend_fontsize", export_config.legend_fontsize + 1)
    kwargs.setdefault("global_legend_title_fontsize", export_config.legend_fontsize + 1)
    kwargs.setdefault("panel_legend_fontsize", 9)
    kwargs.setdefault("panel_legend_title_fontsize", 9)
    kwargs.setdefault("fit_stats_fontsize", 9)
    kwargs.setdefault("panel_letter_fontsize", export_config.title_fontsize + 1)
    kwargs.setdefault("panel_letter_y", 1.14)
    kwargs.setdefault("show_panel_secondary_legend", True)
    kwargs.setdefault("xlabel_rule_second_line", True)
    kwargs.setdefault("grad_quantiles", (0.10, 0.37, 0.63, 0.90))
    kwargs.setdefault("annotate_fit_stats", True)
    if is_best_secondary_plot:
        meta = _default_path1_best_secondary_plot_metadata()
        kwargs.setdefault("show_panel_titles", True)
        kwargs.setdefault("include_criterion_in_panel_title", False)
        kwargs.setdefault("fit_stats", ("n", "delta_aic", "delta_rmse", "lr_p"))
        kwargs.setdefault("per_label_legend_title", meta["per_label_legend_title"])
        kwargs.setdefault("per_label_grad_label_template", meta["per_label_grad_label_template"])
        kwargs.setdefault("per_label_secondary_unit", meta["per_label_secondary_unit"])
        kwargs.setdefault("per_label_secondary_annotation", meta["per_label_secondary_annotation"])
        kwargs.setdefault("per_label_panel_title", meta["per_label_secondary_annotation"])
        kwargs.setdefault("per_label_stats_box_xy", meta["per_label_stats_box_xy"])
    else:
        kwargs.setdefault("show_panel_titles", False)
        kwargs.setdefault("fit_stats", ("n", "delta_aic", "delta_brier", "lr_p"))
    with _font_rc(export_config):
        return production_plots.production_plot_path1_logit_margin_plus_grad_families_generalized(
            pred_df=pred_df,
            coef_df=coef_df,
            save_dir=save_dir,
            file_name=file_name,
            save_formats=export_config.save_formats,
            **kwargs,
        )


def plot_cohort_histogram(
    df,
    save_dir: str | Path,
    *,
    dose_col: str,
    file_stem: str,
    export_config: FigureExportConfig = FigureExportConfig(),
    **kwargs: Any,
) -> list[Path]:
    kwargs.setdefault("dists_to_try", ["lognorm"])
    kwargs.setdefault("bin_size", 1)
    kwargs.setdefault("vertical_gridlines", True)
    kwargs.setdefault("horizontal_gridlines", True)
    kwargs.setdefault("show_minor_ticks", True)
    kwargs.setdefault("vertical_minor_gridlines", False)
    kwargs.setdefault("title", None)
    kwargs.setdefault("include_min_max", True)
    kwargs.setdefault("axis_label_fontsize", export_config.axes_label_fontsize + 2)
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize + 1)
    kwargs.setdefault("legend_fontsize", export_config.legend_fontsize)
    kwargs.setdefault("stats_box_fontsize", export_config.annotation_fontsize)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with _font_rc(export_config):
        production_plots.histogram_and_fit_v2(
            df,
            dose_col=dose_col,
            save_path=save_dir,
            custom_name=file_stem,
            **kwargs,
        )
    return _expected_export_paths(save_dir, file_stem, export_config)


def plot_cohort_dvh_boxplots(
    cohort_bx_dvh_metrics_df,
    save_dir: str | Path,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    file_stem: str = "dvh_boxplot",
    **kwargs: Any,
) -> list[Path]:
    kwargs.setdefault("title", None)
    kwargs.setdefault("axis_label_font_size", export_config.axes_label_fontsize)
    kwargs.setdefault("tick_label_font_size", export_config.tick_label_fontsize)
    kwargs.setdefault("metric_order_D", ["D_2", "D_50", "D_98"])
    kwargs.setdefault("metric_order_V", ["V_100", "V_125", "V_150", "V_175", "V_200", "V_300"])
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with _font_rc(export_config):
        production_plots.dvh_boxplot_pretty(
            cohort_bx_dvh_metrics_df,
            save_path=save_dir,
            custom_name=file_stem,
            **kwargs,
        )
    return [
        save_dir / f"{file_stem}_d_x.pdf",
        save_dir / f"{file_stem}_d_x.svg",
        save_dir / f"{file_stem}_d_x.png",
        save_dir / f"{file_stem}_v_x.pdf",
        save_dir / f"{file_stem}_v_x.svg",
        save_dir / f"{file_stem}_v_x.png",
    ]


def plot_cohort_delta_vs_predictors(
    delta_long_design,
    save_dir: str | Path,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    file_stem: str = "03c_abs_medianDelta_vs_top4_predictors",
    **kwargs: Any,
):
    kwargs.setdefault("predictor_cols", default_delta_predictor_cols())
    kwargs.setdefault("predictor_label_map", default_delta_predictor_label_map())
    kwargs.setdefault("y_col", "|Delta|")
    kwargs.setdefault("delta_kind_label", ["Δ_median", "Δ_mean", "Δ_mode"])
    kwargs.setdefault("delta_kind_col", "Delta kind")
    kwargs.setdefault("axes_label_fontsize", export_config.axes_label_fontsize)
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("legend_fontsize", export_config.legend_fontsize)
    kwargs.setdefault("height", 3.0)
    kwargs.setdefault("aspect", 1.4)
    kwargs.setdefault("facet_cols", 2)
    kwargs.setdefault("label_style", "latex")
    kwargs.setdefault("idx_sub", ("b", "v"))
    kwargs.setdefault("j_symbol", "(j)")
    kwargs.setdefault("scatter", True)
    kwargs.setdefault("scatter_sample", 20000)
    kwargs.setdefault("scatter_alpha", 0.3)
    kwargs.setdefault("scatter_size", 10.0)
    kwargs.setdefault("annotate_stats", False)
    kwargs.setdefault("write_stats_csv", True)
    kwargs.setdefault("title", None)
    kwargs.setdefault("delta_kind_label_map", default_delta_kind_label_map())
    kwargs.setdefault("zero_x_predictors", default_zero_x_predictors())
    kwargs.setdefault("show_minor_grid", False)
    kwargs.setdefault("save_formats", export_config.save_formats)
    kwargs.setdefault("shared_legend", True)
    kwargs.setdefault("shared_legend_title", None)
    kwargs.setdefault("shared_legend_ncol", 4)
    kwargs.setdefault("shared_legend_y", 0.995)
    kwargs.setdefault("shared_legend_fontsize", export_config.legend_fontsize)
    kwargs.setdefault("shared_legend_ci_label", "Shaded band: 95% CI of OLS fit")
    kwargs.setdefault("show_panel_letters", True)
    kwargs.setdefault("panel_letter_fontsize", export_config.title_fontsize)
    kwargs.setdefault("panel_letter_y", 1.15)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with _font_rc(export_config):
        return production_plots.plot_delta_vs_predictors_pkg_generalized(
            delta_long_design,
            save_dir=save_dir,
            file_prefix=file_stem,
            **kwargs,
        )


def plot_cohort_length_scale_summary(
    df,
    save_dir: str | Path,
    *,
    file_stem: str,
    export_config: FigureExportConfig = FigureExportConfig(),
    metric_family: str,
    y_max_fixed: float | None = None,
    **kwargs: Any,
) -> list[Path]:
    kwargs.setdefault("title", None)
    kwargs.setdefault("violin_or_box", "box")
    kwargs.setdefault("trend_lines", ("mean",))
    kwargs.setdefault("annotate_counts", True)
    kwargs.setdefault("annotation_box", False)
    kwargs.setdefault("y_trim", True)
    kwargs.setdefault("y_min_fixed", 0)
    if y_max_fixed is not None:
        kwargs.setdefault("y_max_fixed", y_max_fixed)
    kwargs.setdefault("include_pair_curves_in_ylim", False)
    kwargs.setdefault("metric_family", metric_family)
    kwargs.setdefault("show_pair_mean_curves", True)
    kwargs.setdefault("show_pair_legend", False)
    kwargs.setdefault("pair_line_alpha", 0.5)
    kwargs.setdefault("pair_line_width", 0.9)
    kwargs.setdefault("box_color", "#D8D8D8")
    kwargs.setdefault("disable_x_minor_ticks", True)
    kwargs.setdefault("axis_label_font_size", export_config.axes_label_fontsize)
    kwargs.setdefault("tick_label_font_size", export_config.tick_label_fontsize)
    kwargs.setdefault("annotation_label_font_size", export_config.annotation_fontsize)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with _font_rc(export_config):
        production_plots.plot_dose_vs_length_with_summary_cohort_v2(
            df,
            x_col="length_scale",
            y_col="dose_diff_abs",
            save_dir=str(save_dir),
            file_name=file_stem,
            **kwargs,
        )
    return [
        save_dir / f"{file_stem}.pdf",
        save_dir / f"{file_stem}.svg",
        save_dir / f"{file_stem}.png",
    ]


def plot_cohort_abs_heatmap(
    mean_diff_cohort_pooled_df,
    mean_diff_grad_cohort_pooled_df,
    save_dir: str | Path,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    file_stem: str = "cohort_dualtri_dose_upper_dosegrad_lower_absolute_pooledstats_no_std_v2",
    **kwargs: Any,
) -> list[Path]:
    normalized_stem = file_stem
    if normalized_stem.startswith("cohort_dualtri_"):
        normalized_stem = normalized_stem[len("cohort_dualtri_") :]
    if normalized_stem.endswith("_v2"):
        normalized_stem = normalized_stem[: -len("_v2")]
    kwargs.setdefault("upper_mean_col", "mean_abs_diff")
    kwargs.setdefault("upper_std_col", None)
    kwargs.setdefault("lower_mean_col", "mean_abs_diff")
    kwargs.setdefault("lower_std_col", None)
    kwargs.setdefault("n_col", "n_biopsies")
    kwargs.setdefault("n_label_fontsize", 10)
    kwargs.setdefault("cell_annot_fontsize", 8)
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("axis_label_fontsize", export_config.axes_label_fontsize)
    kwargs.setdefault("cbar_tick_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("cbar_label_fontsize", export_config.axes_label_fontsize)
    kwargs.setdefault("cbar_label_upper", r"$\overline{|M_{ij}^{D}|}$ ($\mathrm{Gy}$, Upper triangle)")
    kwargs.setdefault("cbar_label_lower", r"$\overline{|M_{ij}^{G}|}$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)")
    kwargs.setdefault("show_title", False)
    kwargs.setdefault("show_annotation_box", False)
    kwargs.setdefault("vmin_upper", 0.0)
    kwargs.setdefault("vmin_lower", 0.0)
    kwargs.setdefault("cmap", "Reds")
    kwargs.setdefault("cbar_pad", 0.6)
    kwargs.setdefault("cbar_label_pad", 8.0)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with _font_rc(export_config):
        production_plots.plot_cohort_eff_size_dualtri_mean_std_with_pooled_dfs_v2(
            mean_diff_cohort_pooled_df,
            mean_diff_grad_cohort_pooled_df,
            save_path_base=save_dir,
            save_name_base=normalized_stem,
            **kwargs,
        )
    return [
        save_dir / f"{file_stem}.pdf",
        save_dir / f"{file_stem}.svg",
        save_dir / f"{file_stem}.png",
    ]
