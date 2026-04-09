from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, NullLocator, StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kernel_regression import KernelReg

import production_plots
from pipeline_shared_config import FigureExportConfig


def _alpha_code(index: int) -> str:
    if index < 0:
        raise ValueError("index must be >= 0")
    chars: list[str] = []
    n = int(index)
    while True:
        n, rem = divmod(n, 26)
        chars.append(chr(ord("A") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(chars))


def build_biopsy_heading_map(
    biopsy_pairs: Sequence[tuple[str, int]],
    *,
    explicit_map: Mapping[tuple[str, int], str] | None = None,
    prefix: str = "Biopsy",
) -> dict[tuple[str, int], str]:
    out = {k: str(v) for k, v in dict(explicit_map or {}).items()}
    used = set(out.values())
    next_idx = 0
    for pair in biopsy_pairs:
        if pair in out:
            continue
        while True:
            candidate = f"{prefix} {_alpha_code(next_idx)}"
            next_idx += 1
            if candidate not in used:
                break
        out[pair] = candidate
        used.add(candidate)
    return out


@contextmanager
def _font_rc(export_config: FigureExportConfig):
    with mpl.rc_context(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "axes.labelsize": export_config.axes_label_fontsize,
            "xtick.labelsize": export_config.tick_label_fontsize,
            "ytick.labelsize": export_config.tick_label_fontsize,
            "legend.fontsize": export_config.legend_fontsize,
            "axes.titlesize": export_config.title_fontsize,
        }
    ):
        yield


def _existing_export_paths(
    save_dir: str | Path,
    file_stem: str,
    export_config: FigureExportConfig,
    *,
    fallback_formats: Sequence[str] = (),
) -> list[Path]:
    candidate_formats = list(export_config.save_formats) + [fmt for fmt in fallback_formats if fmt not in export_config.save_formats]
    paths: list[Path] = []
    for fmt in candidate_formats:
        path = Path(save_dir) / f"{file_stem}.{str(fmt).lstrip('.')}"
        if path.exists():
            paths.append(path)
    return paths


def _matching_export_paths(
    save_dir: str | Path,
    file_stem_fragment: str,
    export_config: FigureExportConfig,
) -> list[Path]:
    paths: list[Path] = []
    for fmt in export_config.save_formats:
        pattern = f"*{file_stem_fragment}.{str(fmt).lstrip('.')}"
        paths.extend(sorted(Path(save_dir).glob(pattern)))
    return paths


ANNOT_BBOX = dict(facecolor="white", edgecolor="black", alpha=1.0, linewidth=0.6, boxstyle="round,pad=0.25")
PROFILE_Q05_Q25_COLOR = "#62d2a2"
PROFILE_Q25_Q75_COLOR = "#7da8de"
PROFILE_Q75_Q95_COLOR = "#62d2a2"
PROFILE_MEDIAN_COLOR = "black"
PROFILE_NOMINAL_COLOR = "#c33d3d"
PROFILE_MODE_COLOR = "#a01d88"
PROFILE_MEAN_COLOR = "#cc7a00"
BIOPSY_PALETTE = ["#0b3b8a", "#c75000", "#2a9d8f", "#7a5195"]

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def _save_figure_multi(fig, save_dir: str | Path, file_stem: str, export_config: FigureExportConfig) -> list[Path]:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for fmt in export_config.save_formats:
        path = out_dir / f"{file_stem}.{str(fmt).lstrip('.')}"
        fig.savefig(path, bbox_inches="tight", dpi=export_config.dpi)
        out_paths.append(path)
    return out_paths


def _add_panel_label(ax, label: str, export_config: FigureExportConfig) -> None:
    ax.text(
        0.00,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=export_config.title_fontsize + 3,
    )


def _add_axes_group_heading(
    fig,
    axes: Sequence[Any],
    label: str,
    export_config: FigureExportConfig,
    *,
    pad: float = 0.012,
    ha: str = "center",
    fontweight: str | None = None,
) -> None:
    positions = [ax.get_position() for ax in axes if ax is not None]
    if not positions:
        return
    x0 = min(pos.x0 for pos in positions)
    x1 = max(pos.x1 for pos in positions)
    y1 = max(pos.y1 for pos in positions)
    x = x0 if ha == "left" else 0.5 * (x0 + x1)
    fig.text(
        x,
        y1 + pad,
        str(label),
        ha=ha,
        va="bottom",
        fontsize=export_config.title_fontsize + 3,
        fontweight=fontweight,
    )


def _add_heatmap_group_heading(
    fig,
    ax,
    aux_axes: Sequence[Any],
    label: str,
    export_config: FigureExportConfig,
    *,
    pad: float = 0.026,
) -> None:
    main_pos = ax.get_position()
    aux_positions = [aux_ax.get_position() for aux_ax in aux_axes if aux_ax is not None]
    top_y = max([main_pos.y1] + [pos.y1 for pos in aux_positions])
    heading_x = max(0.02, main_pos.x0 - 0.060)
    fig.text(
        heading_x,
        top_y + pad,
        str(label),
        ha="left",
        va="bottom",
        fontsize=export_config.title_fontsize + 3,
        fontweight="normal",
    )


def _add_shared_direction_arrow(
    fig,
    *,
    label: str,
    export_config: FigureExportConfig,
    y_text: float = 0.058,
    text_x: float = 0.50,
    arrow_x: float = 0.67,
    text_ha: str = "center",
) -> None:
    arrow_token = r"$\rightarrow$"
    display_arrow = "→"
    label_text = str(label)
    if arrow_token in label_text:
        base_text = label_text.replace(arrow_token, "").rstrip()
        fig.text(
            text_x,
            y_text,
            base_text,
            ha=text_ha,
            va="center",
            fontsize=export_config.axes_label_fontsize + 1,
            fontweight="semibold",
        )
        fig.text(
            arrow_x,
            y_text + 0.001,
            display_arrow,
            ha="left",
            va="center",
            fontsize=export_config.axes_label_fontsize + 16,
            fontweight="semibold",
        )
        return
    fig.text(
        0.5,
        y_text,
        label_text,
        ha="center",
        va="center",
        fontsize=export_config.axes_label_fontsize + 1,
        fontweight="semibold",
    )


def _biopsy_display_label(
    pair: tuple[str, int],
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
) -> str:
    if biopsy_label_map and pair in biopsy_label_map:
        return str(biopsy_label_map[pair])
    return f"{pair[0]}, Bx {pair[1]}"


def _sanitize_file_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(label))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_").lower()


def _build_trial_annotation_lines(trial_curves: Sequence[dict[str, object]]) -> str | None:
    lines = [str(curve["annotation"]) for curve in trial_curves if curve.get("annotation")]
    if not lines:
        return None
    return "\n".join(lines)


def _add_outside_annotation_box(
    ax,
    text: str | None,
    export_config: FigureExportConfig,
    *,
    x: float = 1.0,
    y: float = 1.115,
) -> None:
    if not text:
        return
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=export_config.annotation_fontsize,
        bbox=ANNOT_BBOX,
        clip_on=False,
    )


def _kernel_fit_line(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, *, bandwidth_mm: float = 1.0) -> np.ndarray:
    if len(x) == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    if len(x) == 1 or np.nanstd(y) < 1e-12:
        return np.full_like(x_grid, float(np.nanmean(y)), dtype=float)
    kr = KernelReg(endog=y, exog=x, var_type="c", bw=[bandwidth_mm])
    y_fit, _ = kr.fit(x_grid)
    return np.asarray(y_fit, dtype=float)


def _build_profile_curves(
    df: pd.DataFrame,
    *,
    value_col: str,
    x_col: str = "Z (Bx frame)",
    bandwidth_mm: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    work = df[[x_col, value_col, "MC trial num"]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[x_col, value_col])
    x_grid = np.linspace(work[x_col].min(), work[x_col].max(), 600)

    quantile_curves: dict[float, np.ndarray] = {}
    for q in (0.05, 0.25, 0.50, 0.75, 0.95):
        q_df = work.groupby(x_col)[value_col].quantile(q).reset_index()
        quantile_curves[q] = _kernel_fit_line(
            q_df[x_col].to_numpy(dtype=float),
            q_df[value_col].to_numpy(dtype=float),
            x_grid,
            bandwidth_mm=bandwidth_mm,
        )

    nominal_df = work[work["MC trial num"] == 0]
    nominal_curve = _kernel_fit_line(
        nominal_df[x_col].to_numpy(dtype=float),
        nominal_df[value_col].to_numpy(dtype=float),
        x_grid,
        bandwidth_mm=bandwidth_mm,
    )

    mode_rows: list[tuple[float, float]] = []
    mean_rows: list[tuple[float, float]] = []
    for z_val, g in work.groupby(x_col, sort=True):
        vals = g[value_col].to_numpy(dtype=float)
        if len(vals) < 2:
            mode_val = float(vals[0])
        else:
            try:
                kde = gaussian_kde(vals)
                val_grid = np.linspace(vals.min(), vals.max(), 400)
                mode_val = float(val_grid[np.argmax(kde(val_grid))])
            except Exception:
                mode_val = float(np.median(vals))
        mode_rows.append((float(z_val), mode_val))
        mean_rows.append((float(z_val), float(np.mean(vals))))

    mode_df = pd.DataFrame(mode_rows, columns=[x_col, value_col])
    mean_df = pd.DataFrame(mean_rows, columns=[x_col, value_col])

    return x_grid, {
        "q05": quantile_curves[0.05],
        "q25": quantile_curves[0.25],
        "q50": quantile_curves[0.50],
        "q75": quantile_curves[0.75],
        "q95": quantile_curves[0.95],
        "nominal": nominal_curve,
        "mode": _kernel_fit_line(
            mode_df[x_col].to_numpy(dtype=float),
            mode_df[value_col].to_numpy(dtype=float),
            x_grid,
            bandwidth_mm=bandwidth_mm,
        ),
        "mean": _kernel_fit_line(
            mean_df[x_col].to_numpy(dtype=float),
            mean_df[value_col].to_numpy(dtype=float),
            x_grid,
            bandwidth_mm=bandwidth_mm,
        ),
    }


def _selected_trial_curves(
    df: pd.DataFrame,
    *,
    value_col: str,
    shifts_df: pd.DataFrame,
    bx_index: int,
    bx_ref: str,
    x_col: str = "Z (Bx frame)",
    bandwidth_mm: float = 1.0,
    max_trials: int = 3,
) -> list[dict[str, object]]:
    work = df[[x_col, value_col, "MC trial num"]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[x_col, value_col])
    x_grid = np.linspace(work[x_col].min(), work[x_col].max(), 600)

    trial_ids = sorted(int(t) for t in work["MC trial num"].unique() if int(t) != 0)[:max_trials]
    shifts_idx = (
        shifts_df[
            (shifts_df["Structure type"] == bx_ref)
            & (pd.to_numeric(shifts_df["Structure index"], errors="coerce") == int(bx_index))
        ]
        .set_index("Trial")[["Shift (X)", "Shift (Y)", "Shift (Z)"]]
    )

    out: list[dict[str, object]] = []
    for trial in trial_ids:
        t_df = work[work["MC trial num"].astype(int) == int(trial)]
        y_curve = _kernel_fit_line(
            t_df[x_col].to_numpy(dtype=float),
            t_df[value_col].to_numpy(dtype=float),
            x_grid,
            bandwidth_mm=bandwidth_mm,
        )
        ann_text = None
        if trial in shifts_idx.index:
            sx, sy, sz = shifts_idx.loc[trial].astype(float).tolist()
            d_tot = float(np.sqrt(sx * sx + sy * sy + sz * sz))
            ann_text = f"{trial}: ({sx:.1f}, {sy:.1f}, {sz:.1f}), d = {d_tot:.1f} mm"
        out.append({"trial": trial, "x_grid": x_grid, "y_curve": y_curve, "annotation": ann_text})
    return out


def _add_right_edge_trial_labels(
    ax,
    trial_curves: Sequence[dict[str, object]],
    *,
    color: str = "black",
    x_pad_fraction: float = 0.07,
) -> None:
    if not trial_curves:
        return
    x_max = max(float(curve["x_grid"][-1]) for curve in trial_curves)
    x_min = min(float(curve["x_grid"][0]) for curve in trial_curves)
    x_span = max(x_max - x_min, 1.0)
    y_vals = np.array([float(np.asarray(curve["y_curve"])[-1]) for curve in trial_curves], dtype=float)
    order = np.argsort(y_vals)
    ymin, ymax = ax.get_ylim()
    yrange = max(ymax - ymin, 1.0)
    min_gap = 0.06 * yrange
    adjusted = y_vals.copy()
    for idx in order[1:]:
        prev = order[np.where(order == idx)[0][0] - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev] + min_gap)
    adjusted = np.clip(adjusted, ymin + 0.03 * yrange, ymax - 0.03 * yrange)
    x_text = x_max + x_pad_fraction * x_span
    ax.set_xlim(right=x_text + 0.12 * x_span)
    for i, curve in enumerate(trial_curves):
        y_end = float(np.asarray(curve["y_curve"])[-1])
        ax.annotate(
            str(curve["trial"]),
            xy=(x_max, y_end),
            xytext=(x_text, adjusted[i]),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=12,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=1.1, shrinkA=0, shrinkB=0),
            clip_on=False,
        )


def _add_monotone_trial_labels(
    ax,
    trial_curves: Sequence[dict[str, object]],
    *,
    target_y: float = 50.0,
    y_spacing: float = 4.0,
    x_offset_fraction: float = 0.09,
    color: str = "black",
    outside_axes: bool = False,
    x_axes_fraction: float = 1.03,
) -> None:
    if not trial_curves:
        return

    placements: list[tuple[float, float, int]] = []
    for curve in trial_curves:
        x_grid = np.asarray(curve["x_grid"], dtype=float)
        y_curve = np.asarray(curve["y_curve"], dtype=float)
        idx = int(np.nanargmin(np.abs(y_curve - target_y)))
        placements.append((float(x_grid[idx]), float(y_curve[idx]), int(curve["trial"])))

    placements.sort(key=lambda item: item[0])
    y_center = float(np.mean([item[1] for item in placements]))
    y_targets = y_center + (np.arange(len(placements)) - 0.5 * (len(placements) - 1)) * y_spacing
    ymin, ymax = ax.get_ylim()
    y_targets = np.clip(y_targets, ymin + 0.04 * (ymax - ymin), ymax - 0.04 * (ymax - ymin))
    x_min, x_max = ax.get_xlim()
    x_span = max(x_max - x_min, 1.0)
    if outside_axes:
        text_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        x_text = x_axes_fraction
    else:
        text_transform = ax.transData
        x_text = x_max + x_offset_fraction * x_span
        ax.set_xlim(right=x_text + 0.16 * x_span)

    for (x_src, y_src, trial), y_text in zip(placements, y_targets):
        ax.annotate(
            str(trial),
            xy=(x_src, y_src),
            xytext=(x_text, float(y_text)),
            textcoords=text_transform,
            ha="left",
            va="center",
            fontsize=12,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=1.1, shrinkA=0, shrinkB=0),
            clip_on=False,
        )


def _apply_publication_axis_style(
    ax,
    export_config: FigureExportConfig,
    *,
    show_minor_x: bool = True,
    show_minor_y: bool = True,
) -> None:
    ax.grid(True, which="major", color="#b8b8b8", linewidth=0.6, alpha=0.28)
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
    if show_minor_x or show_minor_y:
        ax.minorticks_on()
    if not show_minor_x:
        ax.xaxis.set_minor_locator(NullLocator())
    if not show_minor_y:
        ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", which="major", labelsize=export_config.tick_label_fontsize, length=5, width=0.9, direction="out", top=False, right=False)
    ax.tick_params(axis="both", which="minor", length=3, width=0.6, direction="out", top=False, right=False)


def _draw_profile_axis(
    ax,
    *,
    point_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    pair: tuple[str, int],
    value_col: str,
    y_label: str | None,
    export_config: FigureExportConfig,
    biopsy_label_map: Mapping[tuple[str, int], str] | None,
    bx_ref: str,
    num_trials_to_show: int,
    annotation_y: float = 1.115,
) -> tuple[list[Any], list[str]]:
    patient_id, bx_index = pair
    label = _biopsy_display_label(pair, biopsy_label_map)
    bx_df = point_df[
        (point_df["Patient ID"].astype(str) == str(patient_id))
        & (pd.to_numeric(point_df["Bx index"], errors="coerce") == int(bx_index))
    ].copy()
    x_grid, curves = _build_profile_curves(bx_df, value_col=value_col)
    trial_curves = _selected_trial_curves(
        bx_df,
        value_col=value_col,
        shifts_df=shifts_df[shifts_df["Patient ID"].astype(str) == str(patient_id)].copy(),
        bx_index=int(bx_index),
        bx_ref=bx_ref,
        max_trials=num_trials_to_show,
    )

    fill_1 = ax.fill_between(x_grid, curves["q05"], curves["q25"], color=PROFILE_Q05_Q25_COLOR, alpha=0.65)
    fill_2 = ax.fill_between(x_grid, curves["q25"], curves["q75"], color=PROFILE_Q25_Q75_COLOR, alpha=0.55)
    fill_3 = ax.fill_between(x_grid, curves["q75"], curves["q95"], color=PROFILE_Q75_Q95_COLOR, alpha=0.65)
    ax.plot(x_grid, curves["q05"], linestyle=":", linewidth=1.1, color="black")
    ax.plot(x_grid, curves["q25"], linestyle=":", linewidth=1.1, color="black")
    ax.plot(x_grid, curves["q75"], linestyle=":", linewidth=1.1, color="black")
    ax.plot(x_grid, curves["q95"], linestyle=":", linewidth=1.1, color="black")
    median_line, = ax.plot(x_grid, curves["q50"], color=PROFILE_MEDIAN_COLOR, linewidth=2.0)
    nominal_line, = ax.plot(x_grid, curves["nominal"], color=PROFILE_NOMINAL_COLOR, linewidth=2.0)
    mode_line, = ax.plot(x_grid, curves["mode"], color=PROFILE_MODE_COLOR, linewidth=1.8)
    mean_line, = ax.plot(x_grid, curves["mean"], color=PROFILE_MEAN_COLOR, linewidth=1.8)

    for trial_curve in trial_curves:
        ax.plot(
            trial_curve["x_grid"],
            trial_curve["y_curve"],
            color="black",
            linewidth=1.05,
            linestyle="--",
            alpha=0.82,
        )
    _add_right_edge_trial_labels(ax, trial_curves, color="black")
    _add_outside_annotation_box(
        ax,
        _build_trial_annotation_lines(trial_curves),
        export_config,
        y=annotation_y,
    )

    ax.set_xlabel(r"Axial position along biopsy $z$ (mm)", fontsize=export_config.axes_label_fontsize)
    ax.set_ylabel(y_label or "", fontsize=export_config.axes_label_fontsize)
    _apply_publication_axis_style(ax, export_config)
    _add_panel_label(ax, label, export_config)

    legend_handles = [
        fill_1,
        fill_2,
        fill_3,
        median_line,
        nominal_line,
        mode_line,
        mean_line,
    ]
    legend_labels = [
        "5th-25th Q",
        "25th-75th Q",
        "75th-95th Q",
        "Median (Q50)",
        "Nominal",
        "Mode",
        "Mean",
    ]
    return legend_handles, legend_labels


def plot_exemplar_axial_profile_pair(
    point_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    biopsies: Sequence[tuple[str, int]],
    save_dir: str | Path,
    file_stem: str,
    *,
    value_col: str,
    y_label: str,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    bx_ref: str = "Bx ref",
    num_trials_to_show: int = 3,
    shared_arrow_label: str = r"To biopsy needle tip / patient superior $\rightarrow$",
) -> list[Path]:
    with _font_rc(export_config):
        fig, axes = plt.subplots(
            1,
            len(biopsies),
            figsize=(6.8 * len(biopsies), 5.35),
            dpi=export_config.dpi,
            sharey=False,
        )
        if len(biopsies) == 1:
            axes = [axes]

        legend_handles = None
        legend_labels = None

        for idx, (ax, pair) in enumerate(zip(axes, biopsies)):
            handles, labels = _draw_profile_axis(
                ax,
                point_df=point_df,
                shifts_df=shifts_df,
                pair=pair,
                value_col=value_col,
                y_label=y_label if idx == 0 else None,
                export_config=export_config,
                biopsy_label_map=biopsy_label_map,
                bx_ref=bx_ref,
                num_trials_to_show=num_trials_to_show,
            )
            if legend_handles is None:
                legend_handles = handles
                legend_labels = labels

        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=4,
                frameon=True,
                fancybox=True,
                facecolor="white",
                edgecolor="black",
                framealpha=0.95,
                fontsize=export_config.legend_fontsize + 1,
                bbox_to_anchor=(0.5, 1.10),
            )
        _add_shared_direction_arrow(
            fig,
            label=shared_arrow_label,
            export_config=export_config,
            text_x=0.595,
            arrow_x=0.603,
            text_ha="right",
        )
        fig.subplots_adjust(top=0.77, bottom=0.20, wspace=0.18)
        out_paths = _save_figure_multi(fig, save_dir, file_stem, export_config)
        plt.close(fig)
        return out_paths


def plot_exemplar_axial_profile_quad(
    point_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    biopsies: Sequence[tuple[str, int]],
    save_dir: str | Path,
    file_stem: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    bx_ref: str = "Bx ref",
    num_trials_to_show: int = 3,
    shared_arrow_label: str = r"To biopsy needle tip / patient superior $\rightarrow$",
) -> list[Path]:
    with _font_rc(export_config):
        fig, axes = plt.subplots(
            2,
            len(biopsies),
            figsize=(6.8 * len(biopsies), 9.1),
            dpi=export_config.dpi,
            sharex=False,
            sharey=False,
        )
        if len(biopsies) == 1:
            axes = np.asarray([[axes[0]], [axes[1]]], dtype=object)

        legend_handles = None
        legend_labels = None
        row_specs = [
            ("Dose (Gy)", r"Dose along core $D_b(z)$ (Gy)"),
            ("Dose grad (Gy/mm)", "Dose-gradient magnitude\nalong core $G_b(z)$ (Gy mm$^{-1}$)"),
        ]

        for row_idx, (value_col, y_label) in enumerate(row_specs):
            for col_idx, pair in enumerate(biopsies):
                handles, labels = _draw_profile_axis(
                    axes[row_idx, col_idx],
                    point_df=point_df,
                    shifts_df=shifts_df,
                    pair=pair,
                    value_col=value_col,
                    y_label=y_label if col_idx == 0 else None,
                    export_config=export_config,
                    biopsy_label_map=biopsy_label_map,
                    bx_ref=bx_ref,
                    num_trials_to_show=num_trials_to_show,
                    annotation_y=1.07 if row_idx == 0 else 1.05,
                )
                if legend_handles is None:
                    legend_handles = handles
                    legend_labels = labels

        fig.text(0.040, 0.955, "a)", ha="left", va="top", fontsize=export_config.title_fontsize + 2)
        fig.text(0.040, 0.49, "b)", ha="left", va="top", fontsize=export_config.title_fontsize + 2)
        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=4,
                frameon=True,
                fancybox=True,
                facecolor="white",
                edgecolor="black",
                framealpha=0.95,
                fontsize=export_config.legend_fontsize + 1,
                bbox_to_anchor=(0.5, 0.995),
            )
        _add_shared_direction_arrow(fig, label=shared_arrow_label, export_config=export_config, y_text=0.026)
        fig.subplots_adjust(top=0.84, bottom=0.14, hspace=0.72, wspace=0.18)
        out_paths = _save_figure_multi(fig, save_dir, file_stem, export_config)
        plt.close(fig)
        return out_paths


def _interp_curve_linear(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    y = np.minimum.accumulate(y)
    y_grid = np.interp(x_grid, x, y, left=100.0, right=0.0)
    return np.clip(y_grid, 0.0, 100.0)


def _draw_dvh_axis(
    ax,
    *,
    cumulative_dvh_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    pair: tuple[str, int],
    export_config: FigureExportConfig,
    biopsy_label_map: Mapping[tuple[str, int], str] | None,
    bx_ref: str,
    num_trials_to_show: int,
    y_label: str | None,
) -> tuple[list[Any], list[str]]:
    patient_id, bx_index = pair
    label = _biopsy_display_label(pair, biopsy_label_map)
    bx_df = cumulative_dvh_df[
        (cumulative_dvh_df["Patient ID"].astype(str) == str(patient_id))
        & (pd.to_numeric(cumulative_dvh_df["Bx index"], errors="coerce") == int(bx_index))
    ].copy()
    x_grid = np.linspace(float(bx_df["Dose (Gy)"].min()), float(bx_df["Dose (Gy)"].max()), 700)
    trial_ids = sorted(int(t) for t in bx_df["MC trial"].unique())
    Y = np.empty((len(trial_ids), len(x_grid)), dtype=float)
    trial_frames = {int(t): g for t, g in bx_df.groupby("MC trial", sort=False)}
    for i, trial in enumerate(trial_ids):
        g = trial_frames[trial]
        Y[i] = _interp_curve_linear(
            g["Dose (Gy)"].to_numpy(dtype=float),
            g["Percent volume"].to_numpy(dtype=float),
            x_grid,
        )
    q05 = np.percentile(Y, 5, axis=0)
    q25 = np.percentile(Y, 25, axis=0)
    q50 = np.percentile(Y, 50, axis=0)
    q75 = np.percentile(Y, 75, axis=0)
    q95 = np.percentile(Y, 95, axis=0)

    fill_1 = ax.fill_between(x_grid, q05, q25, color=PROFILE_Q05_Q25_COLOR, alpha=0.65)
    fill_2 = ax.fill_between(x_grid, q25, q75, color=PROFILE_Q25_Q75_COLOR, alpha=0.55)
    fill_3 = ax.fill_between(x_grid, q75, q95, color=PROFILE_Q75_Q95_COLOR, alpha=0.65)
    ax.plot(x_grid, q05, linestyle=":", linewidth=1.1, color="black")
    ax.plot(x_grid, q25, linestyle=":", linewidth=1.1, color="black")
    ax.plot(x_grid, q75, linestyle=":", linewidth=1.1, color="black")
    ax.plot(x_grid, q95, linestyle=":", linewidth=1.1, color="black")
    median_line, = ax.plot(x_grid, q50, color=PROFILE_MEDIAN_COLOR, linewidth=2.0)

    if 0 in trial_frames:
        nominal_curve = _interp_curve_linear(
            trial_frames[0]["Dose (Gy)"].to_numpy(dtype=float),
            trial_frames[0]["Percent volume"].to_numpy(dtype=float),
            x_grid,
        )
        nominal_line, = ax.plot(x_grid, nominal_curve, color=PROFILE_NOMINAL_COLOR, linewidth=2.0)
    else:
        nominal_line, = ax.plot([], [], color=PROFILE_NOMINAL_COLOR, linewidth=2.0)

    trial_curves = []
    shifts_idx = (
        shifts_df[
            (shifts_df["Patient ID"].astype(str) == str(patient_id))
            & (shifts_df["Structure type"] == bx_ref)
            & (pd.to_numeric(shifts_df["Structure index"], errors="coerce") == int(bx_index))
        ]
        .set_index("Trial")[["Shift (X)", "Shift (Y)", "Shift (Z)"]]
    )
    for trial in [t for t in trial_ids if t != 0][:num_trials_to_show]:
        y_curve = _interp_curve_linear(
            trial_frames[trial]["Dose (Gy)"].to_numpy(dtype=float),
            trial_frames[trial]["Percent volume"].to_numpy(dtype=float),
            x_grid,
        )
        ax.plot(x_grid, y_curve, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
        ann_text = None
        if trial in shifts_idx.index:
            sx, sy, sz = shifts_idx.loc[trial].astype(float).tolist()
            d_tot = float(np.sqrt(sx * sx + sy * sy + sz * sz))
            ann_text = f"{trial}: ({sx:.1f}, {sy:.1f}, {sz:.1f}), d = {d_tot:.1f} mm"
        trial_curves.append({"trial": trial, "x_grid": x_grid, "y_curve": y_curve, "annotation": ann_text})
    _add_monotone_trial_labels(
        ax,
        trial_curves,
        target_y=52.0,
        y_spacing=5.0,
        outside_axes=True,
        x_axes_fraction=1.06,
        color="black",
    )
    _add_outside_annotation_box(ax, _build_trial_annotation_lines(trial_curves), export_config, y=1.14)
    ax.set_xlabel(r"Dose (Gy)", fontsize=export_config.axes_label_fontsize)
    ax.set_ylabel(y_label or "", fontsize=export_config.axes_label_fontsize)
    _apply_publication_axis_style(ax, export_config)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0.0, 101.0, 20.0))
    ax.tick_params(axis="y", which="both", left=True, labelleft=True)
    for tick in ax.get_yticklabels():
        tick.set_visible(True)
    active_percent = pd.to_numeric(bx_df["Percent volume"], errors="coerce")
    active_dose = pd.to_numeric(bx_df.loc[active_percent > 0.5, "Dose (Gy)"], errors="coerce").dropna()
    x_min = max(0.0, float(np.nanmin(x_grid)))
    if not active_dose.empty:
        active_xmax = float(np.nanquantile(active_dose.to_numpy(dtype=float), 0.995))
    else:
        active_mask = np.nanmax(Y, axis=0) > 0.5
        if np.any(active_mask):
            active_xmax = float(x_grid[np.where(active_mask)[0][-1]])
        else:
            active_xmax = float(np.nanmax(x_grid))
    x_max = active_xmax + 0.04 * max(active_xmax - x_min, 1.0)
    x_max = min(float(np.nanmax(x_grid)), x_max)
    ax.set_xlim(x_min, x_max)
    _add_panel_label(ax, label, export_config)

    return [fill_1, fill_2, fill_3, median_line, nominal_line], [
        "5th-25th Q",
        "25th-75th Q",
        "75th-95th Q",
        "Median (Q50)",
        "Nominal",
    ]


def plot_exemplar_cumulative_dvh_pair(
    cumulative_dvh_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    biopsies: Sequence[tuple[str, int]],
    save_dir: str | Path,
    file_stem: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    bx_ref: str = "Bx ref",
    num_trials_to_show: int = 3,
) -> list[Path]:
    with _font_rc(export_config):
        fig, axes = plt.subplots(1, len(biopsies), figsize=(7.2 * len(biopsies), 5.35), dpi=export_config.dpi, sharey=False)
        if len(biopsies) == 1:
            axes = [axes]

        legend_handles = None
        legend_labels = None

        for idx, (ax, pair) in enumerate(zip(axes, biopsies)):
            handles, labels = _draw_dvh_axis(
                ax,
                cumulative_dvh_df=cumulative_dvh_df,
                shifts_df=shifts_df,
                pair=pair,
                export_config=export_config,
                biopsy_label_map=biopsy_label_map,
                bx_ref=bx_ref,
                num_trials_to_show=num_trials_to_show,
                y_label=r"Percent volume (\%)",
            )
            if legend_handles is None:
                legend_handles = handles
                legend_labels = labels

        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=len(legend_handles),
                frameon=True,
                fancybox=True,
                facecolor="white",
                edgecolor="black",
                framealpha=0.95,
                fontsize=export_config.legend_fontsize + 1,
                bbox_to_anchor=(0.5, 1.09),
            )
        fig.subplots_adjust(top=0.78, bottom=0.15, wspace=0.28)
        out_paths = _save_figure_multi(fig, save_dir, file_stem, export_config)
        plt.close(fig)
        return out_paths


def _build_delta_overlay_payload(
    deltas_df: pd.DataFrame,
    *,
    biopsies: Sequence[tuple[str, int]],
    zero_level_index_str: str,
    x_axis: str,
    include_abs: bool,
    require_precomputed_abs: bool,
    fallback_recompute_abs: bool,
    median_superscript: str,
    order_kinds: Sequence[str],
    palette: str | Sequence[Any],
    biopsy_label_map: Mapping[tuple[str, int], str] | None,
) -> dict[str, Any]:
    def _mi(name: str):
        return (name, "") if isinstance(deltas_df.columns, pd.MultiIndex) and (name, "") in deltas_df.columns else name

    def _has(key):
        if isinstance(deltas_df.columns, pd.MultiIndex):
            return key in deltas_df.columns
        return key in deltas_df.columns

    def _block_col(metric: str, *, use_abs: bool, suffix: str):
        top = f"{metric} abs deltas" if use_abs else f"{metric} deltas"
        if isinstance(deltas_df.columns, pd.MultiIndex):
            key = (top, suffix)
            if key in deltas_df.columns:
                return key
        flat = f"{top}_{suffix}"
        if flat in deltas_df.columns:
            return flat
        if use_abs and fallback_recompute_abs:
            return None
        if use_abs and require_precomputed_abs:
            raise KeyError(f"Missing absolute delta column {(top, suffix)!r}")
        return None

    pid_c = _mi("Patient ID")
    bxi_c = _mi("Bx index")
    x_c = _mi("Voxel begin (Z)" if x_axis == "Voxel begin (Z)" else "Voxel index")
    tidy_frames = []
    for pair in biopsies:
        pid, bx = pair
        label = _biopsy_display_label(pair, biopsy_label_map)
        sub = deltas_df[(deltas_df[pid_c] == pid) & (pd.to_numeric(deltas_df[bxi_c], errors="coerce") == int(bx))].copy()
        if sub.empty:
            continue
        x_work_col = ("_x", "") if isinstance(sub.columns, pd.MultiIndex) else "_x"
        sub[x_work_col] = pd.to_numeric(sub[x_c], errors="coerce")
        sub = sub.dropna(subset=[x_work_col]).sort_values(x_work_col)
        signed_cols = {
            "mean": _block_col(zero_level_index_str, use_abs=False, suffix="nominal_minus_mean"),
            "mode": _block_col(zero_level_index_str, use_abs=False, suffix="nominal_minus_mode"),
            "median": _block_col(zero_level_index_str, use_abs=False, suffix="nominal_minus_q50"),
        }
        signed = sub.loc[:, [signed_cols[k] for k in order_kinds]].copy()
        signed.columns = list(order_kinds)
        signed = signed.assign(x=sub[x_work_col].to_numpy(), Biopsy=label, Kind="Signed")
        signed = signed.melt(id_vars=["x", "Biopsy", "Kind"], var_name="j", value_name="Value")
        tidy_frames.append(signed)

        if include_abs:
            abs_cols = {
                "mean": _block_col(zero_level_index_str, use_abs=True, suffix="abs_nominal_minus_mean"),
                "mode": _block_col(zero_level_index_str, use_abs=True, suffix="abs_nominal_minus_mode"),
                "median": _block_col(zero_level_index_str, use_abs=True, suffix="abs_nominal_minus_q50"),
            }
            if all(abs_cols[k] is not None for k in order_kinds):
                absolute = sub.loc[:, [abs_cols[k] for k in order_kinds]].copy()
                absolute.columns = list(order_kinds)
            else:
                absolute = signed.loc[:, ["j", "Value"]].pivot(columns="j", values="Value").abs()
                absolute = absolute.loc[:, list(order_kinds)].reset_index(drop=True)
            absolute = absolute.assign(x=sub[x_work_col].to_numpy(), Biopsy=label, Kind="Absolute")
            absolute = absolute.melt(id_vars=["x", "Biopsy", "Kind"], var_name="j", value_name="Value")
            tidy_frames.append(absolute)

    if not tidy_frames:
        raise ValueError("No delta-line rows found for requested biopsies.")

    tidy = pd.concat(tidy_frames, ignore_index=True)
    if isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(order_kinds))
    else:
        colors = list(palette)
    color_map = {order_kinds[i]: colors[i] for i in range(len(order_kinds))}

    def _latex_j(kind: str) -> str:
        delta_type_text = "G" if "grad" in zero_level_index_str.lower() else "D"
        if kind == "mean":
            sup = r"\mathrm{mean}"
        elif kind == "mode":
            sup = r"\mathrm{mode}"
        elif kind == "median":
            sup = r"\mathrm{" + median_superscript.replace("%", r"\%") + r"}"
        else:
            sup = r"\mathrm{" + str(kind) + r"}"
        return rf"$\Delta {delta_type_text}_{{b,v}}^{{{sup}}}$"

    return {
        "tidy": tidy,
        "signed_df": tidy[tidy["Kind"] == "Signed"].copy(),
        "abs_df": tidy[tidy["Kind"] == "Absolute"].copy(),
        "color_map": color_map,
        "order_kinds": list(order_kinds),
        "delta_type_text": "G" if "grad" in zero_level_index_str.lower() else "D",
        "unit_text": r"Gy mm$^{-1}$" if "grad" in zero_level_index_str.lower() else r"Gy",
        "x_axis": x_axis,
        "zero_level_index_str": zero_level_index_str,
        "biopsy_labels": list(dict.fromkeys(tidy["Biopsy"])),
        "marker_cycle": ["o", "s", "^", "D", "P", "X", "v", ">", "<"],
        "latex_j": _latex_j,
        "include_abs": include_abs,
    }


def _draw_delta_overlay_axis(
    ax,
    *,
    payload: Mapping[str, Any],
    export_config: FigureExportConfig,
    linewidth_signed: float,
    linewidth_abs: float,
    linestyle_signed: str | tuple,
    linestyle_absolute: str | tuple,
    show_markers: bool,
    marker_size: int,
    marker_edgewidth: float,
    marker_every: int | None,
    y_tick_decimals: int | None,
    include_style_handles: bool,
) -> list[Any]:
    def _dash_style(style):
        if isinstance(style, tuple):
            if len(style) == 2 and isinstance(style[1], (tuple, list)):
                return style
            return (0, tuple(style))
        if str(style).lower() in {"solid", "-"}:
            return "solid"
        return style

    signed_df = payload["signed_df"]
    abs_df = payload["abs_df"]
    color_map = payload["color_map"]
    order_kinds = payload["order_kinds"]
    tidy = payload["tidy"]
    sns.lineplot(
        data=signed_df,
        x="x",
        y="Value",
        hue="j",
        hue_order=order_kinds,
        palette=color_map,
        units="Biopsy",
        estimator=None,
        errorbar=None,
        sort=False,
        linestyle=_dash_style(linestyle_signed),
        linewidth=linewidth_signed,
        legend=False,
        ax=ax,
        zorder=2,
    )
    if payload["include_abs"] and not abs_df.empty:
        sns.lineplot(
            data=abs_df,
            x="x",
            y="Value",
            hue="j",
            hue_order=order_kinds,
            palette=color_map,
            units="Biopsy",
            estimator=None,
            errorbar=None,
            sort=False,
            linestyle=_dash_style(linestyle_absolute),
            linewidth=linewidth_abs,
            legend=False,
            ax=ax,
            zorder=3,
        )
    for line in ax.lines:
        line.set_solid_capstyle("round")
        line.set_dash_capstyle("round")
        line.set_path_effects([pe.Stroke(linewidth=line.get_linewidth() + 1.2, foreground="white", alpha=0.65), pe.Normal()])

    if show_markers:
        biopsy_to_marker = {
            label: payload["marker_cycle"][i % len(payload["marker_cycle"])]
            for i, label in enumerate(payload["biopsy_labels"])
        }
        for (jv, kv, bv), group in tidy.groupby(["j", "Kind", "Biopsy"], sort=False):
            mark = biopsy_to_marker[bv]
            sc = ax.scatter(
                group["x"],
                group["Value"],
                marker=mark,
                s=marker_size,
                facecolors="white" if kv == "Absolute" else color_map[jv],
                edgecolors=color_map[jv],
                linewidths=marker_edgewidth,
                zorder=4,
            )
            if marker_every is not None:
                offsets = sc.get_offsets()
                sc.set_offsets(offsets[:: max(1, int(marker_every))])

    x_vals = np.asarray(pd.to_numeric(tidy["x"], errors="coerce").dropna().unique(), dtype=float)
    x_vals = np.sort(x_vals)
    if x_vals.size:
        if payload["x_axis"] == "Voxel index":
            ticks = np.arange(int(np.floor(x_vals.min())), int(np.ceil(x_vals.max())) + 1, dtype=int)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(t)) for t in ticks])
            ax.set_xlim(float(ticks.min()), float(ticks.max()))
        else:
            ax.set_xticks(x_vals)
            ax.set_xticklabels([f"{val:g}" for val in x_vals])

    if payload["include_abs"]:
        y_label = rf"$\Delta {payload['delta_type_text']}_{{b,v}}^{{j}}$, $|\Delta {payload['delta_type_text']}_{{b,v}}^{{j}}|$ ({payload['unit_text']})"
    else:
        y_label = rf"$\Delta {payload['delta_type_text']}_{{b,v}}^{{j}}$ ({payload['unit_text']})"
    x_label = "Voxel index" if payload["x_axis"] == "Voxel index" else r"Axial position along biopsy $z$ (mm)"
    ax.set_xlabel(x_label, fontsize=export_config.axes_label_fontsize)
    ax.set_ylabel(y_label, fontsize=export_config.axes_label_fontsize)
    ax.axhline(0.0, linestyle=":", linewidth=1.0, alpha=0.85, color="0.25")
    _apply_publication_axis_style(ax, export_config, show_minor_x=False, show_minor_y=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="y", which="major", left=True, length=5, width=0.9)
    ax.tick_params(axis="y", which="minor", left=True, length=3, width=0.6)
    if y_tick_decimals is not None:
        ax.yaxis.set_major_formatter(StrMethodFormatter(f"{{x:.{int(y_tick_decimals)}f}}"))
        ax.get_yaxis().get_offset_text().set_visible(False)

    handles = [Line2D([0], [0], color=color_map[j], lw=linewidth_signed, label=payload["latex_j"](j)) for j in order_kinds]
    if include_style_handles:
        handles.append(Line2D([0], [0], color="black", lw=linewidth_signed, linestyle=_dash_style(linestyle_signed), label=rf"$\Delta {payload['delta_type_text']}$"))
        if payload["include_abs"]:
            handles.append(Line2D([0], [0], color="black", lw=linewidth_abs, linestyle=_dash_style(linestyle_absolute), label=rf"$|\Delta {payload['delta_type_text']}|$"))
    if show_markers:
        for idx, label in enumerate(payload["biopsy_labels"]):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=payload["marker_cycle"][idx % len(payload["marker_cycle"])],
                    color="black",
                    linestyle="None",
                    label=label,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                )
            )
    return handles


def plot_exemplar_delta_lines_pair(
    *,
    dose_deltas_df: pd.DataFrame,
    gradient_deltas_df: pd.DataFrame,
    biopsies: Sequence[tuple[str, int]],
    save_dir: str | Path,
    fig_name: str,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    include_abs: bool = False,
    legend_fontsize: int | None = None,
) -> list[Path]:
    legend_fs = export_config.legend_fontsize if legend_fontsize is None else int(legend_fontsize)
    dose_payload = _build_delta_overlay_payload(
        dose_deltas_df,
        biopsies=biopsies,
        zero_level_index_str="Dose (Gy)",
        x_axis="Voxel index",
        include_abs=include_abs,
        require_precomputed_abs=True,
        fallback_recompute_abs=False,
        median_superscript="Q50",
        order_kinds=("mean", "mode", "median"),
        palette="tab10",
        biopsy_label_map=biopsy_label_map,
    )
    grad_payload = _build_delta_overlay_payload(
        gradient_deltas_df,
        biopsies=biopsies,
        zero_level_index_str="Dose grad (Gy/mm)",
        x_axis="Voxel index",
        include_abs=include_abs,
        require_precomputed_abs=True,
        fallback_recompute_abs=False,
        median_superscript="Q50",
        order_kinds=("mean", "mode", "median"),
        palette="tab10",
        biopsy_label_map=biopsy_label_map,
    )
    with _font_rc(export_config):
        sns.set_theme(style="white", rc={"axes.facecolor": "white", "figure.facecolor": "white"})
        fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.2), dpi=export_config.dpi, sharex=False, sharey=False)
        handles = _draw_delta_overlay_axis(
            axes[0],
            payload=dose_payload,
            export_config=export_config,
            linewidth_signed=2.0,
            linewidth_abs=3.0,
            linestyle_signed="solid",
            linestyle_absolute=(0, (4, 3)),
            show_markers=True,
            marker_size=42,
            marker_edgewidth=1.0,
            marker_every=None,
            y_tick_decimals=1,
            include_style_handles=False,
        )
        _draw_delta_overlay_axis(
            axes[1],
            payload=grad_payload,
            export_config=export_config,
            linewidth_signed=2.0,
            linewidth_abs=3.0,
            linestyle_signed="solid",
            linestyle_absolute=(0, (4, 3)),
            show_markers=True,
            marker_size=42,
            marker_edgewidth=1.0,
            marker_every=None,
            y_tick_decimals=1,
            include_style_handles=False,
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=len(handles),
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.95,
            fontsize=legend_fs,
        )
        fig.subplots_adjust(top=0.79, bottom=0.15, wspace=0.28)
        out_paths = _save_figure_multi(fig, save_dir, fig_name, export_config)
        plt.close(fig)
        return out_paths


def plot_exemplar_delta_lines(
    deltas_df,
    biopsies: Sequence[tuple[str, int]],
    save_dir: str | Path,
    fig_name: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    **kwargs: Any,
):
    payload = _build_delta_overlay_payload(
        deltas_df,
        biopsies=biopsies,
        zero_level_index_str=str(kwargs.pop("zero_level_index_str", "Dose (Gy)")),
        x_axis=str(kwargs.pop("x_axis", "Voxel index")),
        include_abs=bool(kwargs.pop("include_abs", True)),
        require_precomputed_abs=bool(kwargs.pop("require_precomputed_abs", True)),
        fallback_recompute_abs=bool(kwargs.pop("fallback_recompute_abs", False)),
        median_superscript=str(kwargs.pop("median_superscript", "Q50")),
        order_kinds=tuple(kwargs.pop("order_kinds", ("mean", "mode", "median"))),
        palette=kwargs.pop("palette", "tab10"),
        biopsy_label_map=biopsy_label_map,
    )
    linewidth_signed = float(kwargs.pop("linewidth_signed", 2.0))
    linewidth_abs = float(kwargs.pop("linewidth_abs", 3.0))
    linestyle_signed = kwargs.pop("linestyle_signed", "solid")
    linestyle_absolute = kwargs.pop("linestyle_absolute", (0, (4, 3)))
    show_markers = bool(kwargs.pop("show_markers", False))
    marker_size = int(kwargs.pop("marker_size", 42))
    marker_edgewidth = float(kwargs.pop("marker_edgewidth", 1.0))
    marker_every = kwargs.pop("marker_every", None)
    y_tick_decimals = kwargs.pop("y_tick_decimals", 1)
    legend_fontsize = kwargs.pop("legend_fontsize", export_config.legend_fontsize)

    with _font_rc(export_config):
        sns.set_theme(style="white", rc={"axes.facecolor": "white", "figure.facecolor": "white"})
        fig, ax = plt.subplots(figsize=(13.4, 5.25), dpi=export_config.dpi)
        handles = _draw_delta_overlay_axis(
            ax,
            payload=payload,
            export_config=export_config,
            linewidth_signed=linewidth_signed,
            linewidth_abs=linewidth_abs,
            linestyle_signed=linestyle_signed,
            linestyle_absolute=linestyle_absolute,
            show_markers=show_markers,
            marker_size=marker_size,
            marker_edgewidth=marker_edgewidth,
            marker_every=marker_every,
            y_tick_decimals=y_tick_decimals,
            include_style_handles=False,
        )
        legend = ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.17),
            ncol=len(handles),
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.95,
            fontsize=legend_fontsize,
        )
        for txt in legend.get_texts():
            txt.set_fontsize(legend_fontsize)
        fig.subplots_adjust(top=0.72, bottom=0.15)
        out_paths = _save_figure_multi(fig, save_dir, fig_name, export_config)
        plt.close(fig)
        return out_paths


def _build_voxel_dualboxes_payload(
    deltas_df,
    *,
    biopsies: Sequence[tuple[str, int]],
    biopsy_label_map: Mapping[tuple[str, int], str] | None,
    metric: str,
    x_axis: str,
    lane_gap: float,
    box_width: float,
    pair_gap: float,
    biopsy_gap: float,
    y_label_mode: str,
) -> dict[str, Any]:
    def _mi(name: str):
        return (name, "") if isinstance(deltas_df.columns, pd.MultiIndex) and (name, "") in deltas_df.columns else name

    def _has(key):
        if isinstance(deltas_df.columns, pd.MultiIndex):
            return key in deltas_df.columns
        return key in deltas_df.columns

    def _trial_col(use_abs: bool):
        if use_abs:
            candidates = [
                (f"{metric} abs deltas", "abs_nominal_minus_trial"),
                f"{metric} abs deltas_abs_nominal_minus_trial",
            ]
        else:
            candidates = [
                (f"{metric} deltas", "nominal_minus_trial"),
                f"{metric} deltas_nominal_minus_trial",
            ]
        for cand in candidates:
            if _has(cand):
                return cand
        raise KeyError(f"Missing trial delta column for metric={metric!r}, use_abs={use_abs}.")

    def _ylabel():
        is_grad = "grad" in metric.lower()
        unit = r"(Gy mm$^{-1}$)" if is_grad else r"(Gy)"
        if is_grad:
            left, right = r"$\Delta G_{b,v,t}$", r"$|\Delta G_{b,v,t}|$"
        else:
            left, right = r"$\Delta D_{b,v,t}$", r"$|\Delta D_{b,v,t}|$"
        if y_label_mode == "and":
            return f"{left} and {right}  {unit}"
        if y_label_mode == "slash":
            return f"{left} / {right}  {unit}"
        return f"{left}, {right}  {unit}"

    pid_c = _mi("Patient ID")
    bxi_c = _mi("Bx index")
    x_c = _mi(x_axis)
    signed_key = _trial_col(False)
    abs_key = _trial_col(True)
    df = deltas_df[
        (deltas_df[pid_c].isin([pid for pid, _ in biopsies]))
        & (pd.to_numeric(deltas_df[bxi_c], errors="coerce").isin([int(bx) for _, bx in biopsies]))
    ].copy()
    if df.empty:
        raise ValueError("No voxel dualbox rows found for requested biopsies.")
    x_work_col = ("_x", "") if isinstance(df.columns, pd.MultiIndex) else "_x"
    df[x_work_col] = pd.to_numeric(df[x_c], errors="coerce")
    x_vals_sorted = np.sort(pd.to_numeric(df[x_work_col], errors="coerce").dropna().unique())
    lane_centers = np.arange(len(x_vals_sorted), dtype=float) * lane_gap
    within_pair_offset = box_width / 2.0 + pair_gap / 2.0
    biopsy_center_gap = 2.0 * within_pair_offset + biopsy_gap
    biopsy_offsets = np.linspace(
        -0.5 * (len(biopsies) - 1) * biopsy_center_gap,
        0.5 * (len(biopsies) - 1) * biopsy_center_gap,
        len(biopsies),
    )

    per_signed: dict[tuple[str, int], list[np.ndarray]] = {}
    per_abs: dict[tuple[str, int], list[np.ndarray]] = {}
    for pair in biopsies:
        pid, bx = pair
        sub = df[(df[pid_c] == pid) & (pd.to_numeric(df[bxi_c], errors="coerce") == int(bx))].copy()
        sub = sub.dropna(subset=[x_work_col]).sort_values(x_work_col)
        signed_arrays: list[np.ndarray] = []
        abs_arrays: list[np.ndarray] = []
        for x_val in x_vals_sorted:
            group = sub[sub[x_work_col] == x_val]
            signed_arrays.append(pd.to_numeric(group[signed_key], errors="coerce").dropna().to_numpy(dtype=float))
            abs_arrays.append(pd.to_numeric(group[abs_key], errors="coerce").dropna().to_numpy(dtype=float))
        per_signed[pair] = signed_arrays
        per_abs[pair] = abs_arrays

    return {
        "metric": metric,
        "x_axis": x_axis,
        "x_vals_sorted": x_vals_sorted,
        "lane_centers": lane_centers,
        "within_pair_offset": within_pair_offset,
        "biopsy_offsets": biopsy_offsets,
        "box_width": box_width,
        "per_signed": per_signed,
        "per_abs": per_abs,
        "biopsy_labels": [_biopsy_display_label(pair, biopsy_label_map) for pair in biopsies],
        "y_label": _ylabel(),
        "is_grad": "grad" in metric.lower(),
    }


def _draw_voxel_dualboxes_axis(
    ax,
    *,
    deltas_df,
    biopsies: Sequence[tuple[str, int]],
    export_config: FigureExportConfig,
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    show_xlabel: bool = True,
    **kwargs: Any,
) -> list[Any]:
    metric = str(kwargs.pop("metric", "Dose (Gy)"))
    x_axis = str(kwargs.pop("x_axis", "Voxel index"))
    lane_gap = float(kwargs.pop("lane_gap", 1.25))
    box_width = float(kwargs.pop("box_width", 0.22))
    pair_gap = float(kwargs.pop("pair_gap", 0.12))
    biopsy_gap = float(kwargs.pop("biopsy_gap", 0.28))
    whisker_mode = str(kwargs.pop("whisker_mode", "q05q95"))
    showfliers = bool(kwargs.pop("showfliers", False))
    show_points = bool(kwargs.pop("show_points", False))
    point_size = float(kwargs.pop("point_size", 7.0))
    point_alpha = float(kwargs.pop("point_alpha", 0.22))
    jitter_width = float(kwargs.pop("jitter_width", 0.10))
    y_label_mode = str(kwargs.pop("y_label_mode", "comma"))
    y_tick_decimals = kwargs.pop("y_tick_decimals", 1)
    whis = (5, 95) if whisker_mode == "q05q95" else 1.5

    def _group_low_high(arr: np.ndarray) -> tuple[float, float] | None:
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        if whisker_mode == "q05q95":
            return (float(np.nanpercentile(vals, 5)), float(np.nanpercentile(vals, 95)))
        q1, q3 = np.nanpercentile(vals, [25, 75])
        iqr = q3 - q1
        return (float(max(np.nanmin(vals), q1 - 1.5 * iqr)), float(min(np.nanmax(vals), q3 + 1.5 * iqr)))

    payload = _build_voxel_dualboxes_payload(
        deltas_df,
        biopsies=biopsies,
        biopsy_label_map=biopsy_label_map,
        metric=metric,
        x_axis=x_axis,
        lane_gap=lane_gap,
        box_width=box_width,
        pair_gap=pair_gap,
        biopsy_gap=biopsy_gap,
        y_label_mode=y_label_mode,
    )
    lane_centers = payload["lane_centers"]
    x_vals_sorted = payload["x_vals_sorted"]

    for center in lane_centers:
        ax.axvline(center, color="#d4d4d4", linewidth=0.7, alpha=0.85, zorder=0)

    y_min = np.inf
    y_max = -np.inf

    for idx, pair in enumerate(biopsies):
        color = BIOPSY_PALETTE[idx % len(BIOPSY_PALETTE)]
        pos_signed = lane_centers + payload["biopsy_offsets"][idx] - payload["within_pair_offset"]
        pos_abs = lane_centers + payload["biopsy_offsets"][idx] + payload["within_pair_offset"]
        signed_groups = payload["per_signed"][pair]
        abs_groups = payload["per_abs"][pair]

        if any(len(arr) for arr in signed_groups):
            bp = ax.boxplot(
                signed_groups,
                positions=pos_signed,
                widths=payload["box_width"],
                manage_ticks=False,
                whis=whis,
                showfliers=showfliers,
                patch_artist=True,
            )
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.33)
                box.set_edgecolor(color)
                box.set_linewidth(1.15)
            for key in ["whiskers", "caps", "medians"]:
                for line in bp[key]:
                    line.set_color(color)
                    line.set_linewidth(1.15)

        if any(len(arr) for arr in abs_groups):
            bp = ax.boxplot(
                abs_groups,
                positions=pos_abs,
                widths=payload["box_width"],
                manage_ticks=False,
                whis=whis,
                showfliers=showfliers,
                patch_artist=True,
            )
            for box in bp["boxes"]:
                box.set_facecolor("white")
                box.set_alpha(1.0)
                box.set_edgecolor(color)
                box.set_linewidth(1.15)
                box.set_linestyle("--")
            for key in ["whiskers", "caps", "medians"]:
                for line in bp[key]:
                    line.set_color(color)
                    line.set_linewidth(1.15)
                    line.set_linestyle("--")

        if show_points:
            rng = np.random.default_rng(1000 + idx)
            for x_pos, arr in zip(pos_signed, signed_groups):
                if len(arr):
                    x_jitter = x_pos + (rng.random(len(arr)) - 0.5) * jitter_width
                    ax.scatter(x_jitter, arr, s=point_size, alpha=point_alpha, color=color, marker="o", zorder=3)
            for x_pos, arr in zip(pos_abs, abs_groups):
                if len(arr):
                    x_jitter = x_pos + (rng.random(len(arr)) - 0.5) * jitter_width
                    ax.scatter(x_jitter, arr, s=point_size, alpha=point_alpha, color=color, marker="x", zorder=3)

        for arr in signed_groups:
            bounds = _group_low_high(arr)
            if bounds is not None:
                y_min = min(y_min, bounds[0])
                y_max = max(y_max, bounds[1])
        for arr in abs_groups:
            bounds = _group_low_high(arr)
            if bounds is not None:
                y_min = min(y_min, bounds[0])
                y_max = max(y_max, bounds[1])

    ax.set_xticks(lane_centers)
    if x_axis == "Voxel index":
        ax.set_xticklabels([str(int(v)) for v in x_vals_sorted])
        ax.set_xlabel("Voxel index" if show_xlabel else "", fontsize=export_config.axes_label_fontsize)
    else:
        ax.set_xticklabels([f"{v:g}" for v in x_vals_sorted])
        ax.set_xlabel(r"Axial position along biopsy $z$ (mm)" if show_xlabel else "", fontsize=export_config.axes_label_fontsize)
    ax.set_ylabel(payload["y_label"], fontsize=export_config.axes_label_fontsize)
    ax.axhline(0.0, linestyle=":", linewidth=1.0, alpha=0.85, color="0.25")
    _apply_publication_axis_style(ax, export_config, show_minor_x=False, show_minor_y=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="y", which="major", left=True, length=5, width=0.9)
    ax.tick_params(axis="y", which="minor", left=True, length=3, width=0.6)
    if y_tick_decimals is not None:
        ax.yaxis.set_major_formatter(StrMethodFormatter(f"{{x:.{int(y_tick_decimals)}f}}"))
        ax.get_yaxis().get_offset_text().set_visible(False)
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
        pad = 0.06 * max(y_max - y_min, 1e-6)
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        ax.set_ylim(-1.0, 1.0)

    handles = [
        Patch(
            facecolor=BIOPSY_PALETTE[i % len(BIOPSY_PALETTE)],
            edgecolor=BIOPSY_PALETTE[i % len(BIOPSY_PALETTE)],
            alpha=0.33,
            label=payload["biopsy_labels"][i],
        )
        for i in range(len(payload["biopsy_labels"]))
    ]
    handles.append(Line2D([0], [0], color="black", linewidth=1.2, label=r"$\Delta$"))
    handles.append(Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label=r"$|\Delta|$"))
    return handles


def plot_exemplar_voxel_dualboxes(
    deltas_df,
    biopsies: Sequence[tuple[str, int]],
    output_dir: str | Path,
    plot_name_base: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    **kwargs: Any,
):
    legend_loc = str(kwargs.pop("legend_loc", "upper center"))
    with _font_rc(export_config):
        fig, ax = plt.subplots(figsize=(11.8, 5.3), dpi=export_config.dpi)
        handles = _draw_voxel_dualboxes_axis(
            ax,
            deltas_df=deltas_df,
            biopsies=biopsies,
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            **kwargs,
        )
        ax.legend(
            handles=handles,
            loc=legend_loc,
            bbox_to_anchor=(0.5, 1.15) if legend_loc == "upper center" else None,
            ncol=max(2, len(biopsies) + 2),
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.95,
            fontsize=export_config.legend_fontsize,
        )
        fig.subplots_adjust(top=0.80, bottom=0.15)
        out_paths = _save_figure_multi(fig, output_dir, plot_name_base, export_config)
        plt.close(fig)
        return out_paths


def plot_exemplar_voxel_dualboxes_pair(
    deltas_df,
    biopsies: Sequence[tuple[str, int]],
    output_dir: str | Path,
    plot_name_base: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    **kwargs: Any,
) -> list[Path]:
    with _font_rc(export_config):
        fig, axes = plt.subplots(2, 1, figsize=(11.8, 9.2), dpi=export_config.dpi, sharex=False)
        handles = _draw_voxel_dualboxes_axis(
            axes[0],
            deltas_df=deltas_df,
            biopsies=biopsies,
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            metric="Dose (Gy)",
            show_xlabel=True,
            **kwargs,
        )
        _draw_voxel_dualboxes_axis(
            axes[1],
            deltas_df=deltas_df,
            biopsies=biopsies,
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            metric="Dose grad (Gy/mm)",
            show_xlabel=True,
            **kwargs,
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=max(2, len(biopsies) + 2),
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.95,
            fontsize=export_config.legend_fontsize,
        )
        fig.subplots_adjust(top=0.86, bottom=0.11, hspace=0.26)
        out_paths = _save_figure_multi(fig, output_dir, plot_name_base, export_config)
        plt.close(fig)
        return out_paths


def _build_length_scale_payload(
    df,
    *,
    biopsy_label_map: Mapping[tuple[str, int], str] | None,
    x_col: str,
    y_col: str,
    multi_pairs: Sequence[tuple[str, int]] | None,
) -> dict[str, Any]:
    work = df.copy()
    if multi_pairs:
        pair_set = {(str(pid), int(bx)) for pid, bx in multi_pairs}
        mask = [(str(pid), int(bx)) in pair_set for pid, bx in zip(work["Patient ID"], work["Bx index"])]
        work = work.loc[mask].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[x_col, y_col])
    work["Biopsy"] = [
        _biopsy_display_label((str(pid), int(bx)), biopsy_label_map)
        for pid, bx in zip(work["Patient ID"], work["Bx index"])
    ]
    x_vals = np.sort(work[x_col].unique())
    biopsy_labels = list(dict.fromkeys(work["Biopsy"]))
    return {"work": work, "x_vals": x_vals, "biopsy_labels": biopsy_labels}


def _draw_length_scale_boxes_axis(
    ax,
    *,
    df,
    export_config: FigureExportConfig,
    show_xlabel: bool = True,
    **kwargs: Any,
) -> list[Any]:
    biopsy_label_map = kwargs.pop("biopsy_label_map", None)
    x_col = str(kwargs.pop("x_col", "length_scale"))
    y_col = str(kwargs.pop("y_col", "dose_diff_abs"))
    show_points = bool(kwargs.pop("show_points", False))
    y_trim = bool(kwargs.pop("y_trim", True))
    y_min_fixed = kwargs.pop("y_min_fixed", 0)
    metric_family = kwargs.pop("metric_family", "dose")
    multi_pairs = kwargs.pop("multi_pairs", None)
    y_tick_decimals = kwargs.pop("y_tick_decimals", 1)

    payload = _build_length_scale_payload(
        df,
        biopsy_label_map=biopsy_label_map,
        x_col=x_col,
        y_col=y_col,
        multi_pairs=multi_pairs,
    )
    work = payload["work"]
    x_vals = payload["x_vals"]
    biopsy_labels = payload["biopsy_labels"]
    centers = np.arange(len(x_vals), dtype=float)
    box_width = 0.24
    biopsy_offsets = np.linspace(-0.18, 0.18, max(1, len(biopsy_labels)))
    whisker_highs: list[float] = []

    for center in centers:
        ax.axvline(center, color="#d4d4d4", linewidth=0.7, alpha=0.85, zorder=0)

    for idx, biopsy in enumerate(biopsy_labels):
        color = BIOPSY_PALETTE[idx % len(BIOPSY_PALETTE)]
        sub = work[work["Biopsy"] == biopsy].copy()
        groups = [sub.loc[sub[x_col] == x_val, y_col].to_numpy(dtype=float) for x_val in x_vals]
        pos = centers + biopsy_offsets[idx]
        bp = ax.boxplot(
            groups,
            positions=pos,
            widths=box_width,
            manage_ticks=False,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.33)
            box.set_edgecolor(color)
            box.set_linewidth(1.15)
        for key in ["whiskers", "caps", "medians"]:
            for line in bp[key]:
                line.set_color(color)
                line.set_linewidth(1.15)

        means = [np.nanmean(arr) if len(arr) else np.nan for arr in groups]
        ax.plot(
            pos,
            means,
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=4.8,
            zorder=4,
            path_effects=[pe.Stroke(linewidth=3.8, foreground="white", alpha=0.75), pe.Normal()],
        )
        if show_points:
            rng = np.random.default_rng(3000 + idx)
            for x_pos, arr in zip(pos, groups):
                if len(arr):
                    x_jitter = x_pos + (rng.random(len(arr)) - 0.5) * 0.10
                    ax.scatter(x_jitter, arr, s=8, alpha=0.18, color=color, zorder=2)
        for arr in groups:
            if len(arr):
                whisker_highs.append(float(np.nanpercentile(arr, 95)))

    if metric_family == "grad":
        y_label = r"$\mathcal{S}_b^{G}(\ell_k)$ (Gy mm$^{-1}$)"
    else:
        y_label = r"$\mathcal{S}_b^{D}(\ell_k)$ (Gy)"
    ax.set_xticks(centers)
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else f"{v:g}" for v in x_vals])
    ax.set_xlabel(r"$\ell_k$ (mm)" if show_xlabel else "", fontsize=export_config.axes_label_fontsize)
    ax.set_ylabel(y_label, fontsize=export_config.axes_label_fontsize)
    if y_trim and whisker_highs:
        y_upper = max(whisker_highs)
        y_lower = y_min_fixed if y_min_fixed is not None else max(0.0, work[y_col].quantile(0.05))
        ax.set_ylim(y_lower, y_upper * 1.06)
    _apply_publication_axis_style(ax, export_config, show_minor_x=False, show_minor_y=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="y", which="major", left=True, length=5, width=0.9)
    ax.tick_params(axis="y", which="minor", left=True, length=3, width=0.6)
    if y_tick_decimals is not None:
        ax.yaxis.set_major_formatter(StrMethodFormatter(f"{{x:.{int(y_tick_decimals)}f}}"))
        ax.get_yaxis().get_offset_text().set_visible(False)
    handles = [
        Patch(
            facecolor=BIOPSY_PALETTE[i % len(BIOPSY_PALETTE)],
            edgecolor=BIOPSY_PALETTE[i % len(BIOPSY_PALETTE)],
            alpha=0.33,
            label=label,
        )
        for i, label in enumerate(biopsy_labels)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.1,
            marker="o",
            markersize=5.0,
            label="Mean across MC trials",
        )
    )
    return handles


def plot_exemplar_length_scale_boxes(
    df,
    save_dir: str | Path,
    file_name: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    **kwargs: Any,
):
    with _font_rc(export_config):
        fig, ax = plt.subplots(figsize=(10.4, 5.9), dpi=export_config.dpi)
        handles = _draw_length_scale_boxes_axis(
            ax,
            df=df,
            export_config=export_config,
            **kwargs,
        )
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=max(2, len(handles)),
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.95,
            fontsize=export_config.legend_fontsize,
        )
        fig.subplots_adjust(top=0.80, bottom=0.15)
        out_paths = _save_figure_multi(fig, save_dir, file_name, export_config)
        plt.close(fig)
        return out_paths


def plot_exemplar_length_scale_boxes_pair(
    dose_df,
    gradient_df,
    save_dir: str | Path,
    file_name: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    multi_pairs: Sequence[tuple[str, int]] | None = None,
) -> list[Path]:
    with _font_rc(export_config):
        fig, axes = plt.subplots(2, 1, figsize=(10.6, 9.2), dpi=export_config.dpi, sharex=False)
        handles = _draw_length_scale_boxes_axis(
            axes[0],
            df=dose_df,
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            multi_pairs=multi_pairs,
            metric_family="dose",
            y_min_fixed=0,
            show_xlabel=True,
        )
        _draw_length_scale_boxes_axis(
            axes[1],
            df=gradient_df,
            export_config=export_config,
            biopsy_label_map=biopsy_label_map,
            multi_pairs=multi_pairs,
            metric_family="grad",
            y_min_fixed=0,
            show_xlabel=True,
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=max(2, len(handles)),
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.95,
            fontsize=export_config.legend_fontsize,
        )
        fig.subplots_adjust(top=0.86, bottom=0.11, hspace=0.26)
        out_paths = _save_figure_multi(fig, save_dir, file_name, export_config)
        plt.close(fig)
        return out_paths


def _build_voxel_pair_heatmap_panels(
    upper_df,
    lower_df,
    *,
    patient_id_col: str,
    bx_index_col: str,
    bx_id_col: str,
    biopsy_label_map: Mapping[tuple[str, int], str] | None,
    upper_mean_col: str,
    upper_std_col: str | None,
    lower_mean_col: str,
    lower_std_col: str | None,
    vmin,
    vmax,
    vmin_upper,
    vmax_upper,
    vmin_lower,
    vmax_lower,
    cell_value_decimals: int,
    requested_biopsies: Sequence[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    upper_groups = upper_df.groupby([patient_id_col, bx_index_col, bx_id_col])
    lower_groups = lower_df.groupby([patient_id_col, bx_index_col, bx_id_col])
    all_keys = sorted(set(upper_groups.groups.keys()) | set(lower_groups.groups.keys()))

    if requested_biopsies is not None:
        ordered_keys: list[tuple[Any, Any, Any]] = []
        for patient_id, bx_index in requested_biopsies:
            for key in all_keys:
                if str(key[0]) == str(patient_id) and int(key[1]) == int(bx_index):
                    ordered_keys.append(key)
                    break
        all_keys = ordered_keys

    def _pivot(grp: pd.DataFrame | None, col: str | None, voxels: list[int]) -> pd.DataFrame | None:
        if grp is None or col is None:
            return None
        return grp.pivot(index="voxel1", columns="voxel2", values=col).reindex(index=voxels, columns=voxels)

    panels: list[dict[str, Any]] = []
    fmt = f"{{:.{int(cell_value_decimals)}f}}"
    for key in all_keys:
        pid, bxi, bxid = key
        grp_upper = upper_groups.get_group(key) if key in upper_groups.groups else None
        grp_lower = lower_groups.get_group(key) if key in lower_groups.groups else None
        voxels = set()
        if grp_upper is not None:
            voxels |= set(grp_upper["voxel1"]) | set(grp_upper["voxel2"])
        if grp_lower is not None:
            voxels |= set(grp_lower["voxel1"]) | set(grp_lower["voxel2"])
        voxels = sorted(int(v) for v in voxels)
        n = len(voxels)
        if n == 0:
            continue

        UM_df = _pivot(grp_upper, upper_mean_col, voxels)
        LM_df = _pivot(grp_lower, lower_mean_col, voxels)
        if UM_df is None:
            UM_df = pd.DataFrame(index=voxels, columns=voxels, dtype=float)
        if LM_df is None:
            LM_df = pd.DataFrame(index=voxels, columns=voxels, dtype=float)
        US_df = _pivot(grp_upper, upper_std_col, voxels) if upper_std_col else None
        LS_df = _pivot(grp_lower, lower_std_col, voxels) if lower_std_col else None

        UM = UM_df.values
        LM = LM_df.values
        upper_display = np.full((n, n), np.nan, dtype=float)
        lower_display = np.full((n, n), np.nan, dtype=float)
        annot = np.full((n, n), "", dtype=object)
        US = US_df.values if US_df is not None else None
        LS = LS_df.values if LS_df is not None else None
        for i in range(n):
            for j in range(n):
                if i < j:
                    upper_display[i, j] = UM[i, j]
                    if np.isfinite(UM[i, j]):
                        sval = US[i, j] if US is not None else np.nan
                        annot[i, j] = fmt.format(UM[i, j]) if not np.isfinite(sval) else f"{fmt.format(UM[i, j])}\n±\n{fmt.format(sval)}"
                elif i > j:
                    lower_display[i, j] = LM[j, i]
                    if np.isfinite(LM[j, i]):
                        sval = LS[j, i] if LS is not None else np.nan
                        annot[i, j] = fmt.format(LM[j, i]) if not np.isfinite(sval) else f"{fmt.format(LM[j, i])}\n±\n{fmt.format(sval)}"
                else:
                    if np.isfinite(UM[i, j]):
                        upper_display[i, j] = UM[i, j]
                        sval = US[i, j] if US is not None else np.nan
                        annot[i, j] = fmt.format(UM[i, j]) if not np.isfinite(sval) else f"{fmt.format(UM[i, j])}\n±\n{fmt.format(sval)}"
                    elif np.isfinite(LM[i, j]):
                        lower_display[i, j] = LM[i, j]
                        sval = LS[i, j] if LS is not None else np.nan
                        annot[i, j] = fmt.format(LM[i, j]) if not np.isfinite(sval) else f"{fmt.format(LM[i, j])}\n±\n{fmt.format(sval)}"

        def _resolve_limits(arr: np.ndarray, local_vmin, local_vmax):
            vals = arr[np.isfinite(arr)]
            low = float(vals.min()) if vals.size else (vmin if vmin is not None else 0.0)
            high = float(vals.max()) if vals.size else (vmax if vmax is not None else 1.0)
            return (local_vmin if local_vmin is not None else low, local_vmax if local_vmax is not None else high)

        panels.append(
            {
                "pair": (str(pid), int(bxi)),
                "label": _biopsy_display_label((str(pid), int(bxi)), biopsy_label_map) if biopsy_label_map else str(bxid),
                "voxels": voxels,
                "upper_display": upper_display,
                "lower_display": lower_display,
                "annot": annot,
                "upper_limits": _resolve_limits(upper_display, vmin_upper, vmax_upper),
                "lower_limits": _resolve_limits(lower_display, vmin_lower, vmax_lower),
            }
        )
    return panels


def _draw_voxel_pair_heatmap_axis(
    ax,
    *,
    panel: Mapping[str, Any],
    export_config: FigureExportConfig,
    tick_label_fontsize: int,
    axis_label_fontsize: int,
    cbar_tick_fontsize: int,
    cbar_label_fontsize: int,
    cbar_label_upper: str,
    cbar_label_lower: str,
    show_title: bool,
    title_text: str,
    cell_annot_fontsize: int,
    y_axis_upper_tri_label: str,
    x_axis_upper_tri_label: str,
    y_axis_lower_tri_label: str,
    x_axis_lower_tri_label: str,
    color_bar_positions: str,
    cmap: str,
    cbar_pad: float,
    cbar_label_pad: float,
    show_annotation_box: bool,
    annotation_info: Mapping[str, Any] | None,
) -> list[Any]:
    upper_display = np.asarray(panel["upper_display"], dtype=float)
    lower_display = np.asarray(panel["lower_display"], dtype=float)
    annot = np.asarray(panel["annot"], dtype=object)
    voxels = list(panel["voxels"])
    upper_limits = tuple(panel["upper_limits"])
    lower_limits = tuple(panel["lower_limits"])
    n = len(voxels)

    sns.heatmap(
        lower_display,
        cmap=cmap,
        vmin=lower_limits[0],
        vmax=lower_limits[1],
        mask=~np.isfinite(lower_display),
        cbar=False,
        linewidths=0.0,
        linecolor="white",
        square=True,
        ax=ax,
    )
    sns.heatmap(
        upper_display,
        cmap=cmap,
        vmin=upper_limits[0],
        vmax=upper_limits[1],
        mask=~np.isfinite(upper_display),
        cbar=False,
        linewidths=0.0,
        linecolor="white",
        square=True,
        ax=ax,
    )
    cmap_obj = plt.get_cmap(cmap)
    for i in range(n):
        for j in range(n):
            if not annot[i, j]:
                continue
            val = upper_display[i, j] if np.isfinite(upper_display[i, j]) else lower_display[i, j]
            low, high = upper_limits if np.isfinite(upper_display[i, j]) else lower_limits
            color = production_plots.get_contrasting_color(val, low, high, cmap_obj)
            ax.text(j + 0.5, i + 0.5, annot[i, j], ha="center", va="center", fontsize=cell_annot_fontsize, color=color)

    if show_title:
        ax.set_title(title_text, fontsize=export_config.title_fontsize)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(voxels, fontsize=tick_label_fontsize)
    ax.set_yticklabels(voxels, fontsize=tick_label_fontsize)
    ax.set_xlabel(x_axis_lower_tri_label, fontsize=axis_label_fontsize, labelpad=10)
    ax.set_ylabel(y_axis_lower_tri_label, fontsize=axis_label_fontsize, labelpad=12)
    ax.minorticks_off()
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", which="major", length=3.4, width=0.7, direction="out", top=False, right=False, bottom=True, left=True)
    ax.tick_params(axis="both", which="minor", length=0, top=False, right=False, bottom=False, left=False)
    for side in ["bottom", "left", "top", "right"]:
        ax.spines[side].set_visible(False)

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(ax.get_xticks())
    top_ax.set_xticklabels(voxels, fontsize=tick_label_fontsize)
    top_ax.set_xlabel(x_axis_upper_tri_label, fontsize=axis_label_fontsize, labelpad=15)
    top_ax.minorticks_off()
    top_ax.xaxis.set_minor_locator(NullLocator())
    top_ax.tick_params(axis="x", which="major", length=4, width=0.7, direction="out", top=True, bottom=False, pad=2)
    top_ax.tick_params(axis="x", which="minor", length=0, top=False, bottom=False)
    top_ax.spines["top"].set_visible(False)

    right_ax = ax.secondary_yaxis("right")
    right_ax.set_yticks(ax.get_yticks())
    right_ax.set_yticklabels(voxels, fontsize=tick_label_fontsize)
    right_ax.set_ylabel(y_axis_upper_tri_label, fontsize=axis_label_fontsize, labelpad=12)
    right_ax.minorticks_off()
    right_ax.yaxis.set_minor_locator(NullLocator())
    right_ax.tick_params(axis="y", which="major", length=4, width=0.7, direction="out", right=True, left=False, pad=2)
    right_ax.tick_params(axis="y", which="minor", length=0, right=False, left=False)
    right_ax.spines["right"].set_visible(False)

    divider = make_axes_locatable(ax)
    if color_bar_positions == "left_right":
        cax_lower = divider.append_axes("left", size="4.8%", pad=cbar_pad)
        cax_upper = divider.append_axes("right", size="4.8%", pad=cbar_pad)
        cbar_lower = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*lower_limits), cmap=cmap), cax=cax_lower)
        cbar_upper = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*upper_limits), cmap=cmap), cax=cax_upper)
        cbar_lower.ax.yaxis.set_ticks_position("left")
        cbar_lower.ax.yaxis.set_label_position("left")
        cbar_lower.ax.yaxis.tick_left()
        cbar_upper.ax.yaxis.set_ticks_position("right")
        cbar_upper.ax.yaxis.set_label_position("right")
        cbar_upper.ax.yaxis.tick_right()
    else:
        cax_lower = divider.append_axes("bottom", size="4%", pad=cbar_pad)
        cax_upper = divider.append_axes("top", size="4%", pad=cbar_pad)
        cbar_lower = plt.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*lower_limits), cmap=cmap),
            cax=cax_lower,
            orientation="horizontal",
        )
        cbar_upper = plt.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*upper_limits), cmap=cmap),
            cax=cax_upper,
            orientation="horizontal",
        )
        cax_upper.xaxis.set_ticks_position("top")
        cax_upper.xaxis.set_label_position("top")
    cbar_lower.ax.minorticks_off()
    cbar_upper.ax.minorticks_off()
    cbar_lower.ax.tick_params(axis="y", labelsize=cbar_tick_fontsize, which="major", length=4, width=0.7, direction="out", left=True, right=False, labelleft=True, labelright=False, pad=2)
    cbar_upper.ax.tick_params(axis="y", labelsize=cbar_tick_fontsize, which="major", length=4, width=0.7, direction="out", left=False, right=True, labelleft=False, labelright=True, pad=2)
    cbar_lower.ax.spines["right"].set_visible(False)
    cbar_upper.ax.spines["left"].set_visible(False)
    cbar_lower.set_label(cbar_label_lower, fontsize=cbar_label_fontsize, labelpad=cbar_label_pad)
    cbar_upper.set_label(cbar_label_upper, fontsize=cbar_label_fontsize, labelpad=cbar_label_pad)

    if show_annotation_box:
        ann = dict(annotation_info or {})
        ann["Biopsy"] = panel["label"]
        ax.text(
            0.02,
            0.02,
            "\n".join(f"{k}: {v}" for k, v in ann.items()),
            transform=ax.transAxes,
            fontsize=export_config.annotation_fontsize,
            ha="left",
            va="bottom",
            bbox=ANNOT_BBOX,
        )

    return [cax_lower, ax, cax_upper]


def plot_exemplar_voxel_pair_heatmap(
    upper_df,
    lower_df,
    save_dir: str | Path,
    save_name_base: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    **kwargs: Any,
):
    patient_id_col = kwargs.pop("patient_id_col", "Patient ID")
    bx_index_col = kwargs.pop("bx_index_col", "Bx index")
    bx_id_col = kwargs.pop("bx_id_col", "Bx ID")
    biopsy_label_map = kwargs.pop("biopsy_label_map", None)
    upper_mean_col = kwargs.pop("upper_mean_col", "mean_diff")
    upper_std_col = kwargs.pop("upper_std_col", "std_diff")
    lower_mean_col = kwargs.pop("lower_mean_col", "mean_diff")
    lower_std_col = kwargs.pop("lower_std_col", "std_diff")
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    vmin_upper = kwargs.pop("vmin_upper", None)
    vmax_upper = kwargs.pop("vmax_upper", None)
    vmin_lower = kwargs.pop("vmin_lower", None)
    vmax_lower = kwargs.pop("vmax_lower", None)
    tick_label_fontsize = kwargs.pop("tick_label_fontsize", export_config.tick_label_fontsize)
    axis_label_fontsize = kwargs.pop("axis_label_fontsize", export_config.axes_label_fontsize)
    cbar_tick_fontsize = kwargs.pop("cbar_tick_fontsize", export_config.tick_label_fontsize)
    cbar_label_fontsize = kwargs.pop("cbar_label_fontsize", export_config.axes_label_fontsize - 1)
    cbar_label_upper = kwargs.pop("cbar_label_upper", "Mean (Upper)")
    cbar_label_lower = kwargs.pop("cbar_label_lower", "Mean (Lower)")
    show_title = bool(kwargs.pop("show_title", False))
    show_annotation_box = bool(kwargs.pop("show_annotation_box", False))
    cell_annot_fontsize = kwargs.pop("cell_annot_fontsize", 8)
    cell_value_decimals = kwargs.pop("cell_value_decimals", 1)
    y_axis_upper_tri_label = kwargs.pop("y_axis_upper_tri_label", r"Voxel $i$")
    x_axis_upper_tri_label = kwargs.pop("x_axis_upper_tri_label", r"Voxel $j$")
    y_axis_lower_tri_label = kwargs.pop("y_axis_lower_tri_label", r"Voxel $j$")
    x_axis_lower_tri_label = kwargs.pop("x_axis_lower_tri_label", r"Voxel $i$")
    color_bar_positions = kwargs.pop("color_bar_positions", "top_bottom")
    cmap = kwargs.pop("cmap", "coolwarm")
    cbar_pad = kwargs.pop("cbar_pad", 0.34)
    cbar_label_pad = kwargs.pop("cbar_label_pad", 8.0)
    annotation_info = kwargs.pop("annotation_info", None)

    panels = _build_voxel_pair_heatmap_panels(
        upper_df,
        lower_df,
        patient_id_col=patient_id_col,
        bx_index_col=bx_index_col,
        bx_id_col=bx_id_col,
        biopsy_label_map=biopsy_label_map,
        upper_mean_col=upper_mean_col,
        upper_std_col=upper_std_col,
        lower_mean_col=lower_mean_col,
        lower_std_col=lower_std_col,
        vmin=vmin,
        vmax=vmax,
        vmin_upper=vmin_upper,
        vmax_upper=vmax_upper,
        vmin_lower=vmin_lower,
        vmax_lower=vmax_lower,
        cell_value_decimals=cell_value_decimals,
    )
    out_paths: list[Path] = []
    for panel in panels:
        with _font_rc(export_config):
            fig, ax = plt.subplots(figsize=(9.8, 7.9), dpi=export_config.dpi)
            heading_axes = _draw_voxel_pair_heatmap_axis(
                ax,
                panel=panel,
                export_config=export_config,
                tick_label_fontsize=tick_label_fontsize,
                axis_label_fontsize=axis_label_fontsize,
                cbar_tick_fontsize=cbar_tick_fontsize,
                cbar_label_fontsize=cbar_label_fontsize,
                cbar_label_upper=cbar_label_upper,
                cbar_label_lower=cbar_label_lower,
                show_title=show_title,
                title_text=str(save_name_base),
                cell_annot_fontsize=cell_annot_fontsize,
                y_axis_upper_tri_label=y_axis_upper_tri_label,
                x_axis_upper_tri_label=x_axis_upper_tri_label,
                y_axis_lower_tri_label=y_axis_lower_tri_label,
                x_axis_lower_tri_label=x_axis_lower_tri_label,
                color_bar_positions=color_bar_positions,
                cmap=cmap,
                cbar_pad=cbar_pad,
                cbar_label_pad=cbar_label_pad,
                show_annotation_box=show_annotation_box,
                annotation_info=annotation_info,
            )
            fig.canvas.draw()
            _add_heatmap_group_heading(
                fig,
                ax,
                heading_axes,
                panel["label"],
                export_config,
                pad=0.022,
            )
            file_stem = f"{save_name_base}_{_sanitize_file_label(str(panel['label']))}"
            out_paths.extend(_save_figure_multi(fig, save_dir, file_stem, export_config))
            plt.close(fig)
    return out_paths


def plot_exemplar_voxel_pair_heatmap_pair(
    upper_df,
    lower_df,
    biopsies: Sequence[tuple[str, int]],
    save_dir: str | Path,
    save_name_base: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    **kwargs: Any,
) -> list[Path]:
    patient_id_col = kwargs.pop("patient_id_col", "Patient ID")
    bx_index_col = kwargs.pop("bx_index_col", "Bx index")
    bx_id_col = kwargs.pop("bx_id_col", "Bx ID")
    biopsy_label_map = kwargs.pop("biopsy_label_map", None)
    upper_mean_col = kwargs.pop("upper_mean_col", "mean_diff")
    upper_std_col = kwargs.pop("upper_std_col", "std_diff")
    lower_mean_col = kwargs.pop("lower_mean_col", "mean_diff")
    lower_std_col = kwargs.pop("lower_std_col", "std_diff")
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    vmin_upper = kwargs.pop("vmin_upper", None)
    vmax_upper = kwargs.pop("vmax_upper", None)
    vmin_lower = kwargs.pop("vmin_lower", None)
    vmax_lower = kwargs.pop("vmax_lower", None)
    tick_label_fontsize = kwargs.pop("tick_label_fontsize", export_config.tick_label_fontsize)
    axis_label_fontsize = kwargs.pop("axis_label_fontsize", export_config.axes_label_fontsize)
    cbar_tick_fontsize = kwargs.pop("cbar_tick_fontsize", export_config.tick_label_fontsize)
    cbar_label_fontsize = kwargs.pop("cbar_label_fontsize", export_config.axes_label_fontsize - 1)
    cbar_label_upper = kwargs.pop("cbar_label_upper", "Mean (Upper)")
    cbar_label_lower = kwargs.pop("cbar_label_lower", "Mean (Lower)")
    show_title = bool(kwargs.pop("show_title", False))
    show_annotation_box = bool(kwargs.pop("show_annotation_box", False))
    cell_annot_fontsize = kwargs.pop("cell_annot_fontsize", 8)
    cell_value_decimals = kwargs.pop("cell_value_decimals", 1)
    y_axis_upper_tri_label = kwargs.pop("y_axis_upper_tri_label", r"Voxel $i$")
    x_axis_upper_tri_label = kwargs.pop("x_axis_upper_tri_label", r"Voxel $j$")
    y_axis_lower_tri_label = kwargs.pop("y_axis_lower_tri_label", r"Voxel $j$")
    x_axis_lower_tri_label = kwargs.pop("x_axis_lower_tri_label", r"Voxel $i$")
    color_bar_positions = kwargs.pop("color_bar_positions", "left_right")
    cmap = kwargs.pop("cmap", "coolwarm")
    cbar_pad = kwargs.pop("cbar_pad", 0.34)
    cbar_label_pad = kwargs.pop("cbar_label_pad", 8.0)
    annotation_info = kwargs.pop("annotation_info", None)

    panels = _build_voxel_pair_heatmap_panels(
        upper_df,
        lower_df,
        patient_id_col=patient_id_col,
        bx_index_col=bx_index_col,
        bx_id_col=bx_id_col,
        biopsy_label_map=biopsy_label_map,
        upper_mean_col=upper_mean_col,
        upper_std_col=upper_std_col,
        lower_mean_col=lower_mean_col,
        lower_std_col=lower_std_col,
        vmin=vmin,
        vmax=vmax,
        vmin_upper=vmin_upper,
        vmax_upper=vmax_upper,
        vmin_lower=vmin_lower,
        vmax_lower=vmax_lower,
        cell_value_decimals=cell_value_decimals,
        requested_biopsies=biopsies,
    )
    if not panels:
        return []

    with _font_rc(export_config):
        fig, axes = plt.subplots(len(panels), 1, figsize=(9.6, 6.9 * len(panels)), dpi=export_config.dpi)
        if len(panels) == 1:
            axes = [axes]
        heading_specs: list[tuple[Any, Sequence[Any], str]] = []
        for ax, panel in zip(axes, panels):
            heading_axes = _draw_voxel_pair_heatmap_axis(
                ax,
                panel=panel,
                export_config=export_config,
                tick_label_fontsize=tick_label_fontsize,
                axis_label_fontsize=axis_label_fontsize,
                cbar_tick_fontsize=cbar_tick_fontsize,
                cbar_label_fontsize=cbar_label_fontsize,
                cbar_label_upper=cbar_label_upper,
                cbar_label_lower=cbar_label_lower,
                show_title=show_title,
                title_text=str(save_name_base),
                cell_annot_fontsize=cell_annot_fontsize,
                y_axis_upper_tri_label=y_axis_upper_tri_label,
                x_axis_upper_tri_label=x_axis_upper_tri_label,
                y_axis_lower_tri_label=y_axis_lower_tri_label,
                x_axis_lower_tri_label=x_axis_lower_tri_label,
                color_bar_positions=color_bar_positions,
                cmap=cmap,
                cbar_pad=cbar_pad,
                cbar_label_pad=cbar_label_pad,
                show_annotation_box=show_annotation_box,
                annotation_info=annotation_info,
            )
            heading_specs.append((ax, heading_axes, str(panel["label"])))
        fig.subplots_adjust(top=0.98, bottom=0.06, hspace=0.24)
        fig.canvas.draw()
        for ax, heading_axes, label in heading_specs:
            _add_heatmap_group_heading(
                fig,
                ax,
                heading_axes,
                label,
                export_config,
                pad=0.022,
            )
        out_paths = _save_figure_multi(fig, save_dir, save_name_base, export_config)
        plt.close(fig)
        return out_paths


def plot_exemplar_ridgeline_pair(
    point_df: pd.DataFrame,
    cohort_by_voxel_df: pd.DataFrame,
    biopsies: Sequence[tuple[str, int]],
    save_dir: str | Path,
    file_stem: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    x_label: str = "Dose (Gy)",
    y_label: str = r"Voxel index (axial span along biopsy $z$ in mm)",
) -> list[Path]:
    def _mi(df: pd.DataFrame, name: str, sub: str = ""):
        key = (name, sub)
        if isinstance(df.columns, pd.MultiIndex) and key in df.columns:
            return key
        return name

    with _font_rc(export_config):
        fig, axes = plt.subplots(1, len(biopsies), figsize=(7.2 * len(biopsies), 8.6), dpi=export_config.dpi, sharey=False)
        if len(biopsies) == 1:
            axes = [axes]
        legend_handles: list[Any] | None = None
        for idx, (ax, pair) in enumerate(zip(axes, biopsies)):
            pid, bx = pair
            label = _biopsy_display_label(pair, biopsy_label_map)
            pt_sub = point_df[
                (point_df["Patient ID"].astype(str) == str(pid))
                & (pd.to_numeric(point_df["Bx index"], errors="coerce") == int(bx))
            ].copy()
            stats_sub = cohort_by_voxel_df[
                (cohort_by_voxel_df[_mi(cohort_by_voxel_df, "Patient ID")] == str(pid))
                & (pd.to_numeric(cohort_by_voxel_df[_mi(cohort_by_voxel_df, "Bx index")], errors="coerce") == int(bx))
            ].copy()
            if pt_sub.empty or stats_sub.empty:
                continue
            pt_sub["Voxel index"] = pd.to_numeric(pt_sub["Voxel index"], errors="coerce")
            pt_sub["Dose (Gy)"] = pd.to_numeric(pt_sub["Dose (Gy)"], errors="coerce")
            pt_sub = pt_sub.dropna(subset=["Voxel index", "Dose (Gy)"])
            voxel_order = sorted(int(v) for v in pt_sub["Voxel index"].unique())
            biopsy_dose_values = pd.to_numeric(pt_sub["Dose (Gy)"], errors="coerce").dropna().to_numpy(dtype=float)
            summary_cols = [
                _mi(stats_sub, "Dose (Gy)", "quantile_05"),
                _mi(stats_sub, "Dose (Gy)", "quantile_25"),
                _mi(stats_sub, "Dose (Gy)", "quantile_50"),
                _mi(stats_sub, "Dose (Gy)", "quantile_75"),
                _mi(stats_sub, "Dose (Gy)", "quantile_95"),
                _mi(stats_sub, "Dose (Gy)", "argmax_density"),
                _mi(stats_sub, "Dose (Gy)", "mean"),
                _mi(stats_sub, "Dose (Gy)", "nominal"),
            ]
            summary_vals = np.concatenate(
                [
                    pd.to_numeric(stats_sub[col], errors="coerce").dropna().to_numpy(dtype=float)
                    for col in summary_cols
                ]
            )
            q03 = float(np.nanquantile(biopsy_dose_values, 0.03))
            q97 = float(np.nanquantile(biopsy_dose_values, 0.97))
            x_min = min(q03, float(np.nanmin(summary_vals)))
            x_max = max(q97, float(np.nanmax(summary_vals)))
            x_pad = 0.04 * max(x_max - x_min, 1e-6)
            x_min -= x_pad
            x_max += x_pad
            x_grid = np.linspace(x_min, x_max, 500)
            z_begin_col = _mi(stats_sub, "Voxel begin (Z)")
            z_end_col = _mi(stats_sub, "Voxel end (Z)")
            scale_mm = 0.62 * float(np.nanmedian(pd.to_numeric(stats_sub[z_end_col], errors="coerce") - pd.to_numeric(stats_sub[z_begin_col], errors="coerce")))
            for voxel in voxel_order:
                vals = pt_sub.loc[pt_sub["Voxel index"] == voxel, "Dose (Gy)"].dropna().to_numpy(dtype=float)
                row = stats_sub[pd.to_numeric(stats_sub[_mi(stats_sub, "Voxel index")], errors="coerce") == int(voxel)].iloc[0]
                z_begin = float(row[z_begin_col])
                z_end = float(row[z_end_col])
                z_center = 0.5 * (z_begin + z_end)
                if len(vals) >= 2 and np.nanstd(vals) > 1e-12:
                    kde = gaussian_kde(vals)
                    density = kde(x_grid)
                else:
                    density = np.exp(-0.5 * ((x_grid - float(np.nanmean(vals))) / 0.15) ** 2) if len(vals) else np.zeros_like(x_grid)
                density = density / np.nanmax(density) if np.nanmax(density) > 0 else density
                ridge = z_center + density * scale_mm
                ax.fill_between(x_grid, z_center, ridge, color="#8ea7cf", alpha=0.42, zorder=1)
                ax.plot(x_grid, ridge, color="black", linewidth=1.1, zorder=2)
                for q_name in ["quantile_05", "quantile_25", "quantile_50", "quantile_75", "quantile_95"]:
                    q_val = float(row[_mi(stats_sub, "Dose (Gy)", q_name)])
                    ax.vlines(q_val, z_center, z_center + 0.88 * scale_mm, color="gray", linestyle="--", linewidth=0.9, zorder=3)
                ax.vlines(float(row[_mi(stats_sub, "Dose (Gy)", "argmax_density")]), z_center, z_center + 0.88 * scale_mm, color=PROFILE_MODE_COLOR, linewidth=1.1, zorder=3)
                ax.vlines(float(row[_mi(stats_sub, "Dose (Gy)", "mean")]), z_center, z_center + 0.88 * scale_mm, color=PROFILE_MEAN_COLOR, linewidth=1.1, zorder=3)
                ax.vlines(float(row[_mi(stats_sub, "Dose (Gy)", "nominal")]), z_center, z_center + 0.88 * scale_mm, color=PROFILE_NOMINAL_COLOR, linewidth=1.1, zorder=3)

            tick_rows = stats_sub.sort_values(_mi(stats_sub, "Voxel begin (Z)"))
            z_tick_vals = 0.5 * (
                pd.to_numeric(tick_rows[z_begin_col], errors="coerce").to_numpy(dtype=float)
                + pd.to_numeric(tick_rows[z_end_col], errors="coerce").to_numpy(dtype=float)
            )
            voxel_ticks = pd.to_numeric(tick_rows[_mi(stats_sub, "Voxel index")], errors="coerce").astype(int).to_numpy()
            z_begins = pd.to_numeric(tick_rows[z_begin_col], errors="coerce").to_numpy(dtype=float)
            z_ends = pd.to_numeric(tick_rows[z_end_col], errors="coerce").to_numpy(dtype=float)
            ax.set_yticks(z_tick_vals)
            ax.set_yticklabels(
                [f"{int(v)} ({zb:.1f}-{ze:.1f})" for v, zb, ze in zip(voxel_ticks, z_begins, z_ends)],
                fontsize=max(8, export_config.tick_label_fontsize - 2),
            )
            ax.set_xlabel(x_label, fontsize=export_config.axes_label_fontsize)
            ax.set_ylabel(y_label if idx == 0 else "", fontsize=export_config.axes_label_fontsize)
            _apply_publication_axis_style(ax, export_config, show_minor_x=True, show_minor_y=False)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis="x", which="minor", bottom=True, top=False, length=3.2, width=0.7)
            ax.set_xlim(x_min, x_max)
            _add_panel_label(ax, label, export_config)
            if legend_handles is None:
                legend_handles = [
                    Patch(facecolor="#8ea7cf", edgecolor="#8ea7cf", alpha=0.42, label="Dose density"),
                    Line2D([0], [0], color=PROFILE_MODE_COLOR, lw=1.1, label="Mode"),
                    Line2D([0], [0], color=PROFILE_MEAN_COLOR, lw=1.1, label="Mean"),
                    Line2D([0], [0], color=PROFILE_NOMINAL_COLOR, lw=1.1, label="Nominal"),
                    Line2D([0], [0], color="gray", lw=0.9, linestyle="--", label="Quantiles (5, 25, 50, 75, 95%)"),
                ]
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=len(legend_handles),
                frameon=True,
                fancybox=True,
                facecolor="white",
                edgecolor="black",
                framealpha=0.95,
                fontsize=export_config.legend_fontsize,
            )
        fig.subplots_adjust(top=0.86, bottom=0.10, wspace=0.30)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)
