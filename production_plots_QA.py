from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl

import production_plots
from pipeline_shared_config import FigureExportConfig


def _default_file_name(file_stem: str, export_config: FigureExportConfig) -> str:
    primary_ext = str(export_config.save_formats[0]).lstrip(".")
    return f"{file_stem}.{primary_ext}"


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
    kwargs.setdefault("show_panel_titles", True)
    kwargs.setdefault("dpi", export_config.dpi)
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
    kwargs.setdefault("dpi", export_config.dpi)
    with _font_rc(export_config):
        return production_plots.production_plot_path1_logit_margin_plus_grad_families_generalized(
            pred_df=pred_df,
            coef_df=coef_df,
            save_dir=save_dir,
            file_name=file_name,
            save_formats=export_config.save_formats,
            **kwargs,
        )
