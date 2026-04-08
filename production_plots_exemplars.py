from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import FancyArrowPatch
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


def _add_shared_direction_arrow(
    fig,
    *,
    label: str,
    export_config: FigureExportConfig,
    y_text: float = 0.048,
    y_arrow: float = 0.086,
    x_start: float = 0.34,
    x_end: float = 0.66,
) -> None:
    arrow = FancyArrowPatch(
        (x_start, y_arrow),
        (x_end, y_arrow),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.2,
        color="black",
    )
    fig.add_artist(arrow)
    fig.text(
        0.5,
        y_text,
        label,
        ha="center",
        va="center",
        fontsize=export_config.axes_label_fontsize,
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
) -> None:
    if not trial_curves:
        return
    x_max = max(float(curve["x_grid"][-1]) for curve in trial_curves)
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
    x_text = x_max + 0.04 * max(x_max - float(np.min([curve["x_grid"][0] for curve in trial_curves])), 1.0)
    ax.set_xlim(right=x_text + 0.08 * max(x_max, 1.0))
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
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9),
            clip_on=False,
        )


def _add_monotone_trial_labels(
    ax,
    trial_curves: Sequence[dict[str, object]],
    *,
    target_y: float = 50.0,
    y_spacing: float = 4.0,
    x_offset: float = 0.45,
    color: str = "black",
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

    for (x_src, y_src, trial), y_text in zip(placements, y_targets):
        ax.annotate(
            str(trial),
            xy=(x_src, y_src),
            xytext=(x_src + x_offset, float(y_text)),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=12,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9),
            clip_on=False,
        )


def _apply_publication_axis_style(ax, export_config: FigureExportConfig) -> None:
    ax.grid(True, which="major", color="#b8b8b8", linewidth=0.6, alpha=0.25)
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=export_config.tick_label_fontsize, length=5, width=0.9, direction="out", top=False, right=False)
    ax.tick_params(axis="both", which="minor", length=3, width=0.6, direction="out", top=False, right=False)


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
        fig, axes = plt.subplots(1, len(biopsies), figsize=(6.6 * len(biopsies), 5.2), dpi=export_config.dpi, sharey=True)
        if len(biopsies) == 1:
            axes = [axes]

        legend_handles = None
        legend_labels = None

        for ax, pair in zip(axes, biopsies):
            patient_id, bx_index = pair
            label = (biopsy_label_map or {}).get(pair, f"{patient_id}, Bx {bx_index}")
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

            fill_1 = ax.fill_between(x_grid, curves["q05"], curves["q25"], color="#62d2a2", alpha=0.65)
            fill_2 = ax.fill_between(x_grid, curves["q25"], curves["q75"], color="#7da8de", alpha=0.55)
            fill_3 = ax.fill_between(x_grid, curves["q75"], curves["q95"], color="#62d2a2", alpha=0.65)
            ax.plot(x_grid, curves["q05"], linestyle=":", linewidth=1.1, color="black")
            ax.plot(x_grid, curves["q25"], linestyle=":", linewidth=1.1, color="black")
            ax.plot(x_grid, curves["q75"], linestyle=":", linewidth=1.1, color="black")
            ax.plot(x_grid, curves["q95"], linestyle=":", linewidth=1.1, color="black")
            median_line, = ax.plot(x_grid, curves["q50"], color="black", linewidth=2.0)
            nominal_line, = ax.plot(x_grid, curves["nominal"], color="red", linewidth=2.0)
            mode_line, = ax.plot(x_grid, curves["mode"], color="magenta", linewidth=1.8)
            mean_line, = ax.plot(x_grid, curves["mean"], color="orange", linewidth=1.8)

            for trial_curve in trial_curves:
                ax.plot(trial_curve["x_grid"], trial_curve["y_curve"], color="black", linewidth=1.0, linestyle="--", alpha=0.8)
            _add_right_edge_trial_labels(ax, trial_curves, color="black")
            ann_lines = [str(curve["annotation"]) for curve in trial_curves if curve["annotation"]]
            if ann_lines:
                ax.text(
                    1.02,
                    1.005,
                    "\n".join(ann_lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=export_config.annotation_fontsize,
                    bbox=ANNOT_BBOX,
                )

            ax.set_xlabel(r"Axial position along biopsy $z$ (mm)", fontsize=export_config.axes_label_fontsize)
            ax.set_ylabel(y_label if ax is axes[0] else "", fontsize=export_config.axes_label_fontsize)
            _apply_publication_axis_style(ax, export_config)
            _add_panel_label(ax, label, export_config)

            if legend_handles is None:
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
        )
        fig.subplots_adjust(top=0.76, bottom=0.18, wspace=0.12)
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
        fig, axes = plt.subplots(1, len(biopsies), figsize=(6.6 * len(biopsies), 5.2), dpi=export_config.dpi, sharey=True)
        if len(biopsies) == 1:
            axes = [axes]

        legend_handles = None
        legend_labels = None

        for ax, pair in zip(axes, biopsies):
            patient_id, bx_index = pair
            label = (biopsy_label_map or {}).get(pair, f"{patient_id}, Bx {bx_index}")
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
                Y[i] = _interp_curve_linear(g["Dose (Gy)"].to_numpy(dtype=float), g["Percent volume"].to_numpy(dtype=float), x_grid)
            q05 = np.percentile(Y, 5, axis=0)
            q25 = np.percentile(Y, 25, axis=0)
            q50 = np.percentile(Y, 50, axis=0)
            q75 = np.percentile(Y, 75, axis=0)
            q95 = np.percentile(Y, 95, axis=0)

            fill_1 = ax.fill_between(x_grid, q05, q25, color="#62d2a2", alpha=0.65)
            fill_2 = ax.fill_between(x_grid, q25, q75, color="#7da8de", alpha=0.55)
            fill_3 = ax.fill_between(x_grid, q75, q95, color="#62d2a2", alpha=0.65)
            ax.plot(x_grid, q05, linestyle=":", linewidth=1.1, color="black")
            ax.plot(x_grid, q25, linestyle=":", linewidth=1.1, color="black")
            ax.plot(x_grid, q75, linestyle=":", linewidth=1.1, color="black")
            ax.plot(x_grid, q95, linestyle=":", linewidth=1.1, color="black")
            median_line, = ax.plot(x_grid, q50, color="black", linewidth=2.0)

            if 0 in trial_frames:
                nominal_curve = _interp_curve_linear(
                    trial_frames[0]["Dose (Gy)"].to_numpy(dtype=float),
                    trial_frames[0]["Percent volume"].to_numpy(dtype=float),
                    x_grid,
                )
                nominal_line, = ax.plot(x_grid, nominal_curve, color="red", linewidth=2.0)
            else:
                nominal_line, = ax.plot([], [], color="red", linewidth=2.0)

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
                target_y=50.0,
                y_spacing=4.0,
                x_offset=0.45,
                color="black",
            )
            ann_lines = [str(curve["annotation"]) for curve in trial_curves if curve["annotation"]]
            if ann_lines:
                ax.text(
                    1.02,
                    1.005,
                    "\n".join(ann_lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=export_config.annotation_fontsize,
                    bbox=ANNOT_BBOX,
                )

            ax.set_xlabel(r"Dose $D$ (Gy)", fontsize=export_config.axes_label_fontsize)
            ax.set_ylabel(r"Percent volume (\%)" if ax is axes[0] else "", fontsize=export_config.axes_label_fontsize)
            _apply_publication_axis_style(ax, export_config)
            _add_panel_label(ax, label, export_config)

            if legend_handles is None:
                legend_handles = [fill_1, fill_2, fill_3, median_line, nominal_line]
                legend_labels = ["5th-25th Q", "25th-75th Q", "75th-95th Q", "Median (Q50)", "Nominal"]

        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=3,
                frameon=True,
                fancybox=True,
                facecolor="white",
                edgecolor="black",
                framealpha=0.95,
                fontsize=export_config.legend_fontsize + 1,
                bbox_to_anchor=(0.5, 1.10),
            )
        fig.subplots_adjust(top=0.76, bottom=0.14, wspace=0.12)
        out_paths = _save_figure_multi(fig, save_dir, file_stem, export_config)
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
    kwargs.setdefault("axes_label_fontsize", export_config.axes_label_fontsize)
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("title", None)
    with _font_rc(export_config):
        return production_plots.plot_biopsy_deltas_line_multi(
            deltas_df,
            biopsies=biopsies,
            save_dir=save_dir,
            fig_name=fig_name,
            save_formats=export_config.save_formats,
            **kwargs,
        )


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
    kwargs.setdefault("axes_label_fontsize", export_config.axes_label_fontsize)
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("biopsy_label_map", biopsy_label_map)
    with _font_rc(export_config):
        return production_plots.plot_voxel_dualboxes_by_biopsy_lanes(
            deltas_df,
            biopsies=biopsies,
            output_dir=output_dir,
            plot_name_base=plot_name_base,
            save_formats=export_config.save_formats,
            **kwargs,
        )


def plot_exemplar_length_scale_boxes(
    df,
    save_dir: str | Path,
    file_name: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    **kwargs: Any,
):
    kwargs.setdefault("dpi", export_config.dpi)
    kwargs.setdefault("x_col", "length_scale")
    kwargs.setdefault("y_col", "dose_diff_abs")
    kwargs.setdefault("axis_label_font_size", export_config.axes_label_fontsize)
    kwargs.setdefault("tick_label_font_size", export_config.tick_label_fontsize)
    with _font_rc(export_config):
        result = production_plots.plot_dose_vs_length_with_summary_mutlibox(
            df,
            save_dir=save_dir,
            file_name=file_name,
            save_formats=export_config.save_formats,
            **kwargs,
        )
    if result is None:
        return _existing_export_paths(save_dir, file_name, export_config, fallback_formats=("png", "svg"))
    return result


def plot_exemplar_voxel_pair_heatmap(
    upper_df,
    lower_df,
    save_dir: str | Path,
    save_name_base: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    **kwargs: Any,
):
    kwargs.setdefault("patient_id_col", "Patient ID")
    kwargs.setdefault("bx_index_col", "Bx index")
    kwargs.setdefault("bx_id_col", "Bx ID")
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("axis_label_fontsize", export_config.axes_label_fontsize)
    kwargs.setdefault("cbar_tick_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("cbar_label_fontsize", export_config.legend_fontsize)
    with _font_rc(export_config):
        result = production_plots.plot_diff_stats_heatmap_upper_lower(
            upper_df=upper_df,
            lower_df=lower_df,
            save_dir=save_dir,
            save_name_base=save_name_base,
            save_formats=export_config.save_formats,
            **kwargs,
        )
    if result is None:
        return _matching_export_paths(save_dir, save_name_base, export_config)
    return result
