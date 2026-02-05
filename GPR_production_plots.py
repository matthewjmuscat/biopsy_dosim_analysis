"""
Production-quality plotting utilities for GPR variogram outputs.
Plots are intended for publication (e.g., Physics in Medicine & Biology),
with flexible typography, sizing, and export formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import inspect
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import GPR_analysis_pipeline_functions as gpr_pf
from matplotlib.ticker import AutoMinorLocator

# ----------------------------------------------------------------------
# Global plotting defaults (aligned with production_plots.py)
# ----------------------------------------------------------------------
mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType for vector outputs
mpl.rcParams["ps.fonttype"] = 42

# Use STIX fonts and keep LaTeX optional per-call
MPL_FONT_RC = {
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "axes.unicode_minus": True,
}

# Keep a clean, white canvas with black axes for publication figures
MPL_FACE_RC = {
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}

mpl.rcParams.update(MPL_FONT_RC | MPL_FACE_RC)
sns.set_theme(style="white", rc=MPL_FONT_RC | MPL_FACE_RC)

# Global typography controls (uniform across plots)
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 12
ANNOTATION_FONTSIZE = 12
TITLE_FONTSIZE = 17


def _fs_label(val=None):
    return val if val is not None else AXIS_LABEL_FONTSIZE


def _fs_tick(val=None):
    return val if val is not None else TICK_FONTSIZE


def _fs_legend(val=None):
    return val if val is not None else LEGEND_FONTSIZE


def _fs_annot(val=None):
    return val if val is not None else ANNOTATION_FONTSIZE


def _fs_title(val=None):
    return val if val is not None else TITLE_FONTSIZE
PRIMARY_LINE_COLOR = "#0b3b8a"  # deep blue for main semivariogram
OVERLAY_LINE_COLOR = "#000000"   # neutral gray for overlays
GRID_COLOR = "#b8b8b8"
# Match production_plots histogram fill color ('C0' default)
HIST_FILL_COLOR = "#1f77b4"
ANNOT_BBOX = dict(facecolor="white", edgecolor="black", alpha=1.0, linewidth=0.6, boxstyle="round,pad=0.25")
KERNEL_PALETTE = [
    "#0b3b8a",
    "#c75000",
    "#2a9d8f",
    "#7a5195",
    "#dd5182",
]
KERNEL_LABEL_MAP = {
    "matern_nu_1_5": r"Matérn $\nu = 3/2$",
    "matern_nu_2_5": r"Matérn $\nu = 5/2$",
    "rbf": r"RBF",
    "exp": r"Exponential",
}

plt.ioff()

# Global figure size for single-panel per-biopsy plots
PER_BIOPSY_FIGSIZE = (6.4, 4.0)
COHORT_SQUARE_FIGSIZE = (5.6, 5.6)
DEFAULT_SAVE_FORMATS = ("pdf", "svg")

_ORIG_SAVEFIG = mpl.figure.Figure.savefig
_SAVEFIG_CONTEXT = {"formats": None, "seen": None}
_SAVEFIG_PATCHED = False
_ORIG_PLT_SAVEFIG = plt.savefig


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def _setup_matplotlib_defaults(
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
):
    # Re-apply rcParams each call to guard against external resets.
    mpl.rcParams.update(MPL_FONT_RC | MPL_FACE_RC)
    # apply global fontsize defaults
    mpl.rcParams["axes.labelsize"] = AXIS_LABEL_FONTSIZE
    mpl.rcParams["xtick.labelsize"] = TICK_FONTSIZE
    mpl.rcParams["ytick.labelsize"] = TICK_FONTSIZE
    mpl.rcParams["legend.fontsize"] = LEGEND_FONTSIZE
    sns.set_theme(style=seaborn_style, context=seaborn_context, rc=MPL_FONT_RC | MPL_FACE_RC)
    if font_scale is not None:
        sns.set_context(
            seaborn_context,
            font_scale=font_scale,
            rc={
                "axes.labelsize": AXIS_LABEL_FONTSIZE,
                "xtick.labelsize": TICK_FONTSIZE,
                "ytick.labelsize": TICK_FONTSIZE,
                "legend.fontsize": LEGEND_FONTSIZE,
            },
        )


def _save_figure(
    fig,
    base_path: Path | str,
    formats=("pdf", "svg"),
    tight_layout: bool = True,
    dpi: int = 400,
    show: bool = False,
    create_subdir_for_stem: bool = True,
):
    if tight_layout:
        try:
            fig.tight_layout()
        except Exception:
            pass
    base_path = Path(base_path)
    # Option: when base_path has no suffix, either treat it as a folder (old behavior)
    # or as a stem in the parent directory.
    if base_path.suffix == "":
        if create_subdir_for_stem:
            base_dir = base_path
            stem = base_path.name
        else:
            base_dir = base_path.parent if base_path.parent != Path("") else Path(".")
            stem = base_path.name
    else:
        base_dir = base_path.parent
        stem = base_path.stem
    base_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for ext in formats:
        ext = ext.lstrip(".")
        out = base_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        out_paths.append(out)
    if show:
        plt.show()
    plt.close(fig)
    return out_paths


def _fd_bins(
    data: np.ndarray,
    min_bins: int | None = 10,
    max_bins: int | None = 30,
    verbose: bool = True,
    context: str | None = None,
) -> int:
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    n = data.size
    if n == 0:
        if verbose:
            prefix = f"[FD bins]{f' [{context}]' if context else ''}"
            print(f"{prefix} empty data → using min_bins")
        return min_bins if min_bins is not None else 1
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr > 0:
        bin_width = 2 * iqr / np.cbrt(data.size)
        if bin_width > 0:
            raw_bins = (data.max() - data.min()) / bin_width
            low = min_bins if min_bins is not None else -np.inf
            high = max_bins if max_bins is not None else np.inf
            bins = int(np.clip(raw_bins, low, high))
            bins = max(1, bins)
            if verbose:
                clamp_note = ""
                if min_bins is not None and bins == min_bins:
                    clamp_note = " [CLAMPED to min]"
                elif max_bins is not None and bins == max_bins:
                    clamp_note = " [CLAMPED to max]"
                prefix = f"[FD bins]{f' [{context}]' if context else ''}"
                print(
                    f"{prefix} n={n}, range={data.min():.3g}–{data.max():.3g}, "
                    f"IQR={iqr:.3g}, bin_width={bin_width:.3g}, "
                    f"raw_bins≈{raw_bins:.1f} → bins={bins}{clamp_note}"
                )
            return bins
    # fallback sqrt rule
    bins_raw = np.sqrt(data.size)
    low = min_bins if min_bins is not None else -np.inf
    high = max_bins if max_bins is not None else np.inf
    bins = int(np.clip(bins_raw, low, high))
    bins = max(1, bins)
    if verbose:
        clamp_note = ""
        if min_bins is not None and bins == min_bins:
            clamp_note = " [CLAMPED to min]"
        elif max_bins is not None and bins == max_bins:
            clamp_note = " [CLAMPED to max]"
        prefix = f"[FD bins]{f' [{context}]' if context else ''}"
        print(
            f"{prefix} fallback sqrt: n={n}, IQR={iqr:.3g} → sqrt(n)={bins_raw:.1f} → bins={bins}{clamp_note}"
        )
    return bins


def _label_is_numeric(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _axis_is_categorical_x(ax: mpl.axes.Axes) -> bool:
    """
    Heuristic to decide if x is categorical (box/violin/cat):
    - FixedLocator with a modest number of ticks -> categorical.
    - OR majority of labels are non-numeric.
    """
    if getattr(ax, "_force_numeric_x_minor", False):
        return False

    loc = getattr(ax.xaxis, "major_locator", None)
    if isinstance(loc, mpl.ticker.FixedLocator):
        try:
            if len(loc.locs) <= 50:
                return True
        except Exception:
            pass

    labels = [tick.get_text().strip() for tick in ax.get_xticklabels()]
    labels = [l for l in labels if l]
    if not labels:
        return False
    numeric_like = sum(1 for l in labels if _label_is_numeric(l))
    return numeric_like < (0.5 * len(labels))


def _apply_global_tick_style(fig: mpl.figure.Figure):
    """
    Add outward major + minor ticks to x/y axes for all axes in the figure.
    Skip x-minor ticks for categorical x-axes (e.g., box/violin/cat plots).
    """
    for ax in fig.get_axes():
        if not isinstance(ax, mpl.axes.Axes):
            continue

        suppress_x_minor = getattr(ax, "_suppress_x_minor", False)
        x_is_categorical = _axis_is_categorical_x(ax)

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=6,
            width=1.0,
            bottom=True,
            top=False,
            left=True,
            right=False,
        )

        if not x_is_categorical:
            ax.tick_params(
                axis="x",
                which="minor",
                direction="out",
                length=3,
                width=0.8,
                bottom=True,
                top=False,
            )
        elif suppress_x_minor:
            ax.tick_params(
                axis="x",
                which="minor",
                bottom=False,
                top=False,
            )

        for spine in ("bottom", "left"):
            if spine in ax.spines:
                ax.spines[spine].set_visible(True)
        ax.tick_params(axis="x", bottom=True, top=False)
        ax.tick_params(axis="y", left=True, right=False)


def _patched_plt_savefig(*args, **kwargs):
    fig = plt.gcf()
    _apply_global_tick_style(fig)
    return _ORIG_PLT_SAVEFIG(*args, **kwargs)


@contextmanager
def _save_formats_context(formats: Sequence[str] | None):
    old_formats = _SAVEFIG_CONTEXT.get("formats")
    old_seen = _SAVEFIG_CONTEXT.get("seen")
    fmt_list = None if formats is None else [str(f).lower().lstrip(".") for f in formats]
    _SAVEFIG_CONTEXT["formats"] = fmt_list
    _SAVEFIG_CONTEXT["seen"] = set()
    try:
        yield
    finally:
        _SAVEFIG_CONTEXT["formats"] = old_formats
        _SAVEFIG_CONTEXT["seen"] = old_seen


def _patched_savefig(self, fname, *args, **kwargs):
    formats = _SAVEFIG_CONTEXT.get("formats")
    if not formats:
        _apply_global_tick_style(self)
        return _ORIG_SAVEFIG(self, fname, *args, **kwargs)
    if isinstance(fname, (str, Path)):
        base = Path(fname)
        if base.suffix:
            base = base.with_suffix("")
        seen = _SAVEFIG_CONTEXT.get("seen")
        if seen is not None and str(base) in seen:
            return None
        if seen is not None:
            seen.add(str(base))
        results = []
        _apply_global_tick_style(self)
        for fmt in formats:
            out = base.with_suffix(f".{fmt}")
            kw = dict(kwargs)
            kw.pop("format", None)
            kw["format"] = fmt
            results.append(_ORIG_SAVEFIG(self, out, *args, **kw))
        return results
    _apply_global_tick_style(self)
    return _ORIG_SAVEFIG(self, fname, *args, **kwargs)


def _ensure_savefig_patch():
    global _SAVEFIG_PATCHED
    if _SAVEFIG_PATCHED:
        return
    mpl.figure.Figure.savefig = _patched_savefig
    plt.savefig = _patched_plt_savefig
    _SAVEFIG_PATCHED = True

def _compute_shrinkage_stats(gp_res: dict):
    """Return mean MC SD, mean GP SD, and percent reduction."""
    indep_sd = np.sqrt(np.maximum(gp_res.get("var_n", np.array([])), 0))
    gp_sd = np.asarray(gp_res.get("sd_X", np.array([])), dtype=float)
    mean_mc = float(np.nanmean(indep_sd)) if indep_sd.size else np.nan
    mean_gp = float(np.nanmean(gp_sd)) if gp_sd.size else np.nan
    shrink = float(100.0 * (1 - mean_gp / mean_mc)) if np.isfinite(mean_mc) and mean_mc > 0 else np.nan
    return mean_mc, mean_gp, shrink



def _place_legend_top(ax, ncol: int = 2, y: float = 1.30):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
                  ncol=ncol, frameon=False)


def _finalize_legend_and_header(
    ax,
    header: str | None = None,
    *,
    ncol: int = 2,
    header_fontsize: int | None = None,
    header_loc: str = "center",
    handles: list | None = None,
    labels: list | None = None,
    row_major: bool = False,
    legend_width_mode: str = "figure",  # "figure", "axes", or "subplot"
) -> float:
    """
    Place legend above the axes (horizontal rows) and header beneath it, both outside
    the plotting area. Expands the figure height as needed to avoid squashing axes.
    Returns the final figure top margin ratio (for optional use).
    """
    header_fontsize = _fs_legend(header_fontsize)
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    fig = ax.figure
    renderer = None
    if fig and fig.canvas:
        fig.canvas.draw_idle()
        renderer = fig.canvas.get_renderer()

    axes_width = ax.get_window_extent(renderer=renderer).width if renderer is not None else None
    # Subplot (axes bbox) width in pixels
    subplot_width_px = None
    if renderer is not None:
        ax_pos = ax.get_position()  # in figure fraction
        fig_px_width = ax.figure.get_size_inches()[0] * ax.figure.dpi if ax.figure is not None else None
        if fig_px_width is not None:
            subplot_width_px = fig_px_width * ax_pos.width

    # Helper to measure text width in pixels
    def _measure_text_width(text: str) -> float:
        if renderer is None:
            return 0.0
        tmp = ax.text(0, 0, text, fontsize=header_fontsize, fontstyle="italic", transform=ax.transAxes)
        bb = tmp.get_window_extent(renderer=renderer)
        tmp.remove()
        return bb.width

    # Wrap header if it is wider than axes
    header_text = header
    if header_text and renderer is not None:
        axes_width = ax.get_window_extent(renderer=renderer).width
        if _measure_text_width(header_text) > axes_width:
            parts = [p.strip() for p in header_text.split(",")]
            if len(parts) > 1:
                lines = []
                current = parts[0]
                for part in parts[1:]:
                    candidate = current + ", " + part
                    if _measure_text_width(candidate) <= axes_width:
                        current = candidate
                    else:
                        lines.append(current)
                        current = part
                lines.append(current)
                header_text = "\n".join(lines)

    # Cache original axes/figure sizes for later resizing.
    ax_pos = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_h_in = ax_pos.height * fig_h
    ax_bottom_in = ax_pos.y0 * fig_h

    # Place header first (just above axes) with extra padding
    header_artist = None
    header_top = 1.0
    if header_text:
        if header_loc == "right":
            x, ha = 0.98, "right"
        elif header_loc == "left":
            x, ha = 0.02, "left"
        else:
            x, ha = 0.5, "center"
        header_artist = ax.text(
            x,
            1.03,
            header_text,
            transform=ax.transAxes,
            ha=ha,
            va="bottom",
            fontsize=header_fontsize,
            fontstyle="italic",
            bbox=ANNOT_BBOX,
        )
        if renderer is not None:
            hb = header_artist.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
            header_top = hb.y1

    def _row_major_reorder(handles_in, labels_in, cols):
        if not handles_in or cols <= 1:
            return handles_in, labels_in
        n = len(handles_in)
        rows = int(np.ceil(n / cols))
        ordered_h = []
        ordered_l = []
        for c in range(cols):
            for r in range(rows):
                idx = r * cols + c
                if idx < n:
                    ordered_h.append(handles_in[idx])
                    ordered_l.append(labels_in[idx])
        return ordered_h, ordered_l

    # Place legend above header, adjusting columns until it fits the chosen width limit.
    legend = None
    legend_top = header_top
    if handles:
        max_cols = max(1, len(handles))
        fig_px_width = ax.figure.get_size_inches()[0] * ax.figure.dpi if ax.figure is not None else None
        if legend_width_mode == "axes":
            width_limit = axes_width
        elif legend_width_mode == "subplot":
            width_limit = subplot_width_px if subplot_width_px is not None else axes_width
        else:  # "figure" default
            width_limit = fig_px_width
        for cols in range(max_cols, 0, -1):
            if legend is not None:
                legend.remove()
            h_use, l_use = handles, labels
            if row_major:
                h_use, l_use = _row_major_reorder(handles, labels, cols)
            legend = ax.legend(
                h_use,
                l_use,
                loc="lower center",
                bbox_to_anchor=(0.5, header_top + 0.005),
                ncol=cols,
                frameon=False,
            )
            if renderer is None or width_limit is None:
                break
            lbbox = legend.get_window_extent(renderer=renderer)
            if lbbox.width <= width_limit:
                break
        if renderer is not None and legend is not None:
            lb = legend.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
            legend_top = lb.y1

    # Expand figure height to accommodate legend/header without shrinking axes.
    extra_axes = max(0.0, legend_top - 1.0)
    extra_in = extra_axes * ax_h_in + 0.01  # extra padding for legend/header separation
    if extra_in > 0:
        new_fig_h = fig_h + extra_in
        fig.set_size_inches(fig_w, new_fig_h, forward=True)
        new_bottom = ax_bottom_in / new_fig_h
        new_height = ax_h_in / new_fig_h
        ax.set_position([ax_pos.x0, new_bottom, ax_pos.width, new_height])
    return max(0.6, 1.0 - (extra_in / (fig_h + extra_in)) - 0.02)


def _wrap_plot_function(func):
    sig = inspect.signature(func)
    has_save_formats = "save_formats" in sig.parameters

    def _wrapped(*args, **kwargs):
        save_formats = kwargs.get("save_formats", DEFAULT_SAVE_FORMATS)
        _setup_matplotlib_defaults(
            font_scale=kwargs.get("font_scale", 1.0),
            seaborn_style=kwargs.get("seaborn_style", "white"),
            seaborn_context=kwargs.get("seaborn_context", "paper"),
        )
        _ensure_savefig_patch()
        with _save_formats_context(save_formats):
            if not has_save_formats:
                kwargs.pop("save_formats", None)
            return func(*args, **kwargs)

    _wrapped.__wrapped__ = func
    _wrapped._save_formats_wrapped = True  # type: ignore[attr-defined]
    return _wrapped


def _wrap_all_plot_functions():
    for name, obj in list(globals().items()):
        if not callable(obj):
            continue
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != __name__:
            continue
        if getattr(obj, "_save_formats_wrapped", False):
            continue
        globals()[name] = _wrap_plot_function(obj)


# ----------------------------------------------------------------------
# Variogram plotting
# ----------------------------------------------------------------------
def plot_variogram_from_df(
    semivariogram_df: pd.DataFrame,
    patient_id,
    bx_index,
    *,
    overlay_df: pd.DataFrame | None = None,
    include_title_meta: bool = True,
    save_path: str | Path | None = None,   # directory or full file path stem
    file_name: str | None = None,          # if dir provided, base name (ext optional)
    save_formats: Sequence[str] = ("pdf", "svg"),# e.g., ("png","pdf","svg")
    dpi: int = 400,
    figsize: tuple[float, float] = (7.0, 4.5),
    line_kwargs: dict | None = None,       # matplotlib line props for semivariogram
    overlay_kwargs: dict | None = None,    # matplotlib line props for overlays
    grid_alpha: float = 0.25,
    x_label: str = r"Lag $h$ (mm)",
    y_label: str = r"Semivariance $\gamma_b(h)$ (Gy$^2$)",
    label_fontsize: int | None = None,
    tick_labelsize: int | None = None,
    title_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    use_tex: bool = False,
    return_path: bool = False,
    show: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    """
    Render an empirical variogram for a single biopsy with publication-ready
    styling and flexible export options.
    """
    required_cols = {"Patient ID", "Bx index", "h_mm", "semivariance"}
    missing = required_cols - set(semivariogram_df.columns)
    if missing:
        raise ValueError(f"semivariogram_df missing columns: {sorted(missing)}")

    # Filter to the requested biopsy
    mask = (semivariogram_df["Patient ID"] == patient_id) & (semivariogram_df["Bx index"] == bx_index)
    sv = semivariogram_df.loc[mask].copy().sort_values("h_mm")
    if sv.empty:
        raise ValueError(f"No semivariogram rows for Patient ID={patient_id}, Bx index={bx_index}")

    # Prepare overlay if provided
    ov = None
    if overlay_df is not None and not overlay_df.empty:
        ov = overlay_df.copy()
        if {"Patient ID", "Bx index"} <= set(ov.columns):
            ov = ov[(ov["Patient ID"] == patient_id) & (ov["Bx index"] == bx_index)]
        if not ov.empty and "h_mm" in ov.columns:
            ov = ov.sort_values("h_mm")

    # Style defaults tuned for publication aesthetics
    base_line_kwargs = dict(
        marker="o",
        markersize=4.2,
        markeredgewidth=0.9,
        markerfacecolor="white",
        markeredgecolor=PRIMARY_LINE_COLOR,
        linewidth=2.6,
        color=PRIMARY_LINE_COLOR,
    )
    if line_kwargs:
        base_line_kwargs.update(line_kwargs)

    base_overlay_kwargs = dict(linewidth=1.8, color=OVERLAY_LINE_COLOR)
    if overlay_kwargs:
        base_overlay_kwargs.update(overlay_kwargs)

    # Build rc context for LaTeX toggle without polluting global state
    rc_local = MPL_FONT_RC | MPL_FACE_RC | {"text.usetex": bool(use_tex)}
    with mpl.rc_context(rc_local):
        fig, ax = plt.subplots(figsize=figsize)

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("black")

        # Main variogram
        ax.plot(
            sv["h_mm"],
            sv["semivariance"],
            label=r"Empirical $\widehat{\gamma}_b(h)$",
            **base_line_kwargs,
        )

        # Optional overlays
        if ov is not None and not ov.empty:
            if "median_absdiff" in ov.columns:
                ax.plot(
                    ov["h_mm"],
                    ov["median_absdiff"],
                    marker="s",
                    linestyle="--",
                    label="Median |Δdose|",
                    **base_overlay_kwargs,
                )
            if "mean_absdiff" in ov.columns:
                ax.plot(
                    ov["h_mm"],
                    ov["mean_absdiff"],
                    marker="^",
                    linestyle=":",
                    label="Mean |Δdose|",
                    **base_overlay_kwargs,
                )

        if include_title_meta:
            title_bits = [f"Semivariogram, Patient {patient_id}"]
            if "Fraction" in sv.columns:
                frac_vals = sv["Fraction"].dropna().unique()
                if len(frac_vals) == 1:
                    title_bits[-1] += f" (F{frac_vals[0]})"
            title_bits.append(f"Biopsy {bx_index}")
            ax.set_title(", ".join(title_bits), fontsize=_fs_title(title_fontsize), pad=6)

        ax.set_xlabel(x_label, fontsize=_fs_label(label_fontsize))
        ax.set_ylabel(y_label, fontsize=_fs_label(label_fontsize))
        ax.tick_params(axis="both", which="major", labelsize=_fs_tick(tick_labelsize), length=4, width=0.9)
        ax.tick_params(axis="both", which="minor", length=2, width=0.6)
        ax.minorticks_on()
        _apply_per_biopsy_ticks(ax)

        if xlim:
            ax.set_xlim(xlim)
        else:
            x_max = float(sv["h_mm"].max())
            ax.set_xlim(left=0)
            if x_max > 0:
                ax.set_xlim(right=x_max * 1.05)

        if ylim:
            ax.set_ylim(ylim)
        else:
            y_candidates = [sv["semivariance"].max()]
            if ov is not None:
                for col in ("median_absdiff", "mean_absdiff"):
                    if col in ov.columns and not ov[col].dropna().empty:
                        y_candidates.append(ov[col].max())
            y_max = float(np.nanmax(y_candidates))
            ax.set_ylim(bottom=0)
            if y_max > 0:
                ax.set_ylim(top=y_max * 1.05)

        ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=grid_alpha)
        ax.legend(
            frameon=True,
            fancybox=True,
            fontsize=_fs_legend(legend_fontsize),
            handlelength=2.6,
            borderaxespad=0.5,
            facecolor="white",
            edgecolor="black",
            framealpha=0.9,
        )
        fig.tight_layout()

        saved_paths: list[str] = []
        if save_path is not None:
            save_path = Path(save_path)
            if save_path.suffix:  # full path with extension provided
                save_dir = save_path.parent
                base_name = save_path.stem
            else:
                save_dir = save_path
                base_name = file_name or f"variogram_patient{patient_id}_bx{bx_index}"
            save_dir.mkdir(parents=True, exist_ok=True)

            fmt_list = list(dict.fromkeys([fmt.lstrip(".").lower() for fmt in save_formats]))
            for fmt in fmt_list:
                out_path = save_dir / f"{base_name}.{fmt}"
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                saved_paths.append(str(out_path))

        if show:
            plt.show()
        plt.close(fig)

    if return_path:
        saved_return: str | list[str] | None
        if len(saved_paths) == 1:
            saved_return = saved_paths[0]
        elif len(saved_paths) == 0:
            saved_return = None
        else:
            saved_return = saved_paths
        return sv, saved_return

    return sv


# ----------------------------------------------------------------------
# Kernel sensitivity plots
# ----------------------------------------------------------------------
def _apply_axis_style(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.25)
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, length=4, width=0.9)
    ax.tick_params(axis="both", which="minor", length=2, width=0.6)
    ax.minorticks_on()


def _apply_per_biopsy_ticks(ax):
    """Ensure clear major/minor ticks on both axes for per-biopsy plots."""
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", length=5, width=0.9, bottom=True, left=True)
    ax.tick_params(axis="both", which="minor", length=3, width=0.6, bottom=True, left=True)
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False)




def plot_kernel_sensitivity_histogram(
    metrics_df,
    value_col: str,
    y_label: str,
    save_dir,
    file_name_base: str = "kernel_sensitivity_boxplot",
    file_types=("pdf", "svg"),
    show_title: bool = False,
    figsize=(6.0, 4.0),
    label_fontsize: int | None = None,
    tick_fontsize: int | None = None,
    title_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    modes: Sequence[str] = ("histogram",),
    kde_bw_scale: float | None = None,
):
    """
    Histogram of a metric (e.g., ell, mean_ratio) grouped by kernel_label.
    Expects metrics_df to include 'kernel_label' and value_col.
    modes: choose any of {"histogram","kde"}; order controls draw order. KDE lines
           use the same bandwidth across kernels; if kde_bw_scale is None, Scott's
           rule on the pooled data is used; otherwise kde_bw_scale multiplies Scott's
           factor and is passed as bw_method to gaussian_kde (applied uniformly).
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    if "kernel_label" not in metrics_df.columns:
        raise ValueError("metrics_df must contain column 'kernel_label'.")
    if value_col not in metrics_df.columns:
        raise ValueError(f"metrics_df missing column '{value_col}'.")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = metrics_df.copy()
    df["kernel_pretty"] = df["kernel_label"].map(KERNEL_LABEL_MAP).fillna(df["kernel_label"])
    kernels = list(pd.unique(df["kernel_pretty"].dropna()))
    data = [df.loc[df["kernel_pretty"] == k, value_col].dropna() for k in kernels]
    all_vals = pd.concat(data) if len(data) else pd.Series([], dtype=float)
    # Compute per-kernel FD bin widths, then share a common width across kernels
    kernel_bin_widths = []
    for vals in data:
        arr = vals.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            b = _fd_bins(arr, min_bins=None, max_bins=None, context=f"{file_name_base}_per_kernel")
            width = (arr.max() - arr.min()) / b if b and (arr.max() > arr.min()) else np.nan
            kernel_bin_widths.append(width)
    shared_bin_width = float(np.nanmean(kernel_bin_widths)) if kernel_bin_widths else np.nan
    # Build shared bin edges using pooled range
    if all_vals.size and np.isfinite(shared_bin_width) and shared_bin_width > 0:
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        n_bins = max(1, int(np.ceil((vmax - vmin) / shared_bin_width)))
        bins = np.linspace(vmin, vmax, n_bins + 1)
    else:
        bins = _fd_bins(all_vals.to_numpy(), min_bins=None, max_bins=None, context=file_name_base)
    # Accept a single string (e.g., "histogram") or an iterable of strings.
    if isinstance(modes, str):
        modes = (modes,)
    modes = tuple(m.lower() for m in modes)

    with mpl.rc_context({"legend.fontsize": _fs_legend(legend_fontsize)}):
        fig, ax = plt.subplots(figsize=figsize)
        fs_label = _fs_label(label_fontsize)
        fs_tick = _fs_tick(tick_fontsize)
        handles, labels = [], []
        colors = KERNEL_PALETTE * ((len(kernels) // len(KERNEL_PALETTE)) + 1)
        # Pre-compute KDE bandwidth factor (shared)
        bw_method = None
        kde_bw_disp = None
        if "kde" in modes and all_vals.size:
            pooled = all_vals.to_numpy(dtype=float)
            pooled = pooled[np.isfinite(pooled)]
            if pooled.size:
                pooled_kde = stats.gaussian_kde(pooled)
                base_factor = pooled_kde.factor
                factor = kde_bw_scale if kde_bw_scale is not None else 1.0
                bw_method = factor
                kde_bw_disp = factor * base_factor * np.std(pooled, ddof=1)

        # Mean bin width for scaling KDE to histogram counts
        if np.isscalar(bins):
            data_min = float(np.nanmin(all_vals)) if all_vals.size else 0.0
            data_max = float(np.nanmax(all_vals)) if all_vals.size else 1.0
            span = data_max - data_min
            bin_width = span / bins if (bins and span > 0) else 1.0
        else:
            bin_width = float(np.nanmean(np.diff(bins))) if len(bins) > 1 else 1.0

        for k, vals, color in zip(kernels, data, colors):
            vals_arr = vals.to_numpy(dtype=float)
            vals_arr = vals_arr[np.isfinite(vals_arr)]
            if "histogram" in modes:
                h = ax.hist(
                    vals_arr,
                    bins=bins,
                    histtype="step",
                    color=color,
                    edgecolor=color,
                    linewidth=1.1,
                    alpha=1.0,
                    label=k,
                )
                if len(h) >= 3:
                    handles.append(h[2][0])
                    labels.append(k)
            if "kde" in modes and vals_arr.size:
                try:
                    kde = stats.gaussian_kde(vals_arr, bw_method=bw_method)
                    xs = np.linspace(vals_arr.min(), vals_arr.max(), 300)
                    yk = kde(xs) * len(vals_arr) * bin_width  # scale density to match count axis
                    line, = ax.plot(xs, yk, color=color, lw=1.4, label=f"{k} KDE")
                    handles.append(line)
                    labels.append(f"{k} KDE")
                    if kde_bw_disp is None:
                        kde_bw_disp = float(kde.factor * np.std(vals_arr, ddof=1))
                except Exception:
                    print(f"[kernel_sensitivity_histogram] KDE failed for {k}")

    if file_name_base == "kernel_sensitivity_ell":
        ax.set_xlabel(r"$\widehat{\ell}_b$ (mm)", fontsize=fs_label)
        ax.set_ylabel("Count", fontsize=fs_label)
    elif file_name_base == "kernel_sensitivity_mean_ratio":
        ax.set_xlabel("Mean voxelwise shrinkage ratio", fontsize=fs_label)
        ax.set_ylabel("Count", fontsize=fs_label)
    elif file_name_base == "kernel_sensitivity_sv_rmse":
        ax.set_xlabel(r"Semivariogram $\mathrm{RMSE}_b^{(\gamma)}$ (Gy$^2$)", fontsize=fs_label)
        ax.set_ylabel("Count", fontsize=fs_label)
    else:
        ax.set_xlabel("Value", fontsize=fs_label)
        ax.set_ylabel("Count", fontsize=fs_label)

    ax.tick_params(axis="both", labelsize=fs_tick)
    _apply_axis_style(ax)
    # Legend in the standardized style (above axes) matching ratio scatter
    _finalize_legend_and_header(
        ax,
        header=None,
        ncol=len(handles) if handles else 1,
        header_loc="center",
        header_fontsize=_fs_legend(legend_fontsize),
        handles=handles if handles else None,
        labels=labels if labels else None,
    )

    ann_lines = []
    if "kde" in modes and kde_bw_disp is not None and np.isfinite(kde_bw_disp):
        ann_lines.append(rf"$\mathrm{{KDE\ bandwidth}} = {kde_bw_disp:.3g}$")
    if "histogram" in modes and bin_width is not None and np.isfinite(bin_width):
        ann_lines.append(rf"$\mathrm{{bin\ width}} = {bin_width:.3g}$")
    if ann_lines:
        ax.text(
            0.98,
            0.95,
            "\n".join(ann_lines),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=_fs_annot(),
            bbox=ANNOT_BBOX,
        )

    saved_paths = _save_figure(fig, Path(save_dir) / file_name_base, formats=file_types, dpi=400, create_subdir_for_stem=False)
    print(f"[kernel_sensitivity_histogram] saved {file_name_base} -> {', '.join(map(str, saved_paths))}")
    plt.close(fig)





def plot_kernel_sensitivity_scatter(
    metrics_df,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    save_dir,
    file_name_base: str = "kernel_sensitivity_scatter",
    file_types=("pdf", "svg"),
    show_title: bool = False,
    figsize=(6.4, 4.4),
    label_fontsize: int | None = None,
    tick_fontsize: int | None = None,
    title_fontsize: int | None = None,
):
    """
    Scatter plot of two metrics colored by kernel_label.
    Expects metrics_df to include 'kernel_label', x_col, and y_col.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np

    if "kernel_label" not in metrics_df.columns:
        raise ValueError("metrics_df must contain column 'kernel_label'.")
    for col in (x_col, y_col):
        if col not in metrics_df.columns:
            raise ValueError(f"metrics_df missing column '{col}'.")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = metrics_df.copy()
    df["kernel_pretty"] = df["kernel_label"].map(KERNEL_LABEL_MAP).fillna(df["kernel_label"])
    kernels = list(df["kernel_pretty"].dropna().unique())
    color_cycle = KERNEL_PALETTE * ((len(kernels) // len(KERNEL_PALETTE)) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    fs_label = _fs_label(label_fontsize)
    fs_tick = _fs_tick(tick_fontsize)
    for k, color in zip(kernels, color_cycle):
        subset = df.loc[df["kernel_pretty"] == k]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            color=color,
            alpha=0.75,
            s=28,
            label=k,
            edgecolors="white",
            linewidths=0.6,
        )

    # Axis labels per requested outputs
    if file_name_base == "kernel_sensitivity_ratio_scatter":
        ax.set_xlabel("Mean voxelwise shrinkage ratio", fontsize=fs_label)
        ax.set_ylabel(r"$100\,(1 - U_b^{\mathrm{GP}}/U_b^{\mathrm{MC}})$ [% reduction]", fontsize=fs_label)
        header = None
    else:
        ax.set_xlabel(x_label, fontsize=fs_label)
        ax.set_ylabel(y_label, fontsize=fs_label)
        header = None
    ax.tick_params(axis="both", labelsize=fs_tick)
    _apply_axis_style(ax)
    handles, labels = ax.get_legend_handles_labels()
    top = _finalize_legend_and_header(
        ax,
        header=header,
        ncol=len(handles) if handles else 1,
        header_loc="center",
        header_fontsize=_fs_legend(int(fs_label * 0.9)),
    )
    # layout handled by _finalize_legend_and_header (figure height expansion)

    _save_figure(fig, Path(save_dir) / file_name_base, formats=file_types, dpi=400, create_subdir_for_stem=False)
    plt.close(fig)


# ----------------------------------------------------------------------
# Production versions of GP pipeline plots
# ----------------------------------------------------------------------
def plot_gp_profile_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    save_formats=("pdf", "svg"),
    ci_level="both",
    title_on=True,
    figsize=PER_BIOPSY_FIGSIZE,
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    show: bool = False,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    X_star = gp_res["X_star"]
    mu_star = gp_res["mu_star"]
    sd_star = gp_res["sd_star"]
    X = gp_res["X"]
    y = gp_res["y"]
    indep_sd = np.sqrt(np.maximum(gp_res["var_n"], 0))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(X_star, mu_star, lw=2.4, color=PRIMARY_LINE_COLOR, label=r"$\mu^{\mathrm{GP}}_{b,v}$")

    if ci_level == "both":
        ax.fill_between(X_star, mu_star - 1.96 * sd_star, mu_star + 1.96 * sd_star, alpha=0.12, color=PRIMARY_LINE_COLOR, label="95% band")
        ax.fill_between(X_star, mu_star - 1.0 * sd_star, mu_star + 1.0 * sd_star, alpha=0.22, color=PRIMARY_LINE_COLOR, label="68% band")
        ax.errorbar(X, y, yerr=2 * indep_sd, fmt="s", ms=3.5, lw=1.0, color="#1b8a5a", label=r"$\widetilde{D}_{b,v}\pm2\widehat{\sigma}_{b,v}$")
        ax.errorbar(X, y, yerr=indep_sd, fmt="o", ms=3.5, lw=1.0, color="#c75000", label=r"$\widetilde{D}_{b,v}\pm\widehat{\sigma}_{b,v}$")
    elif ci_level in (0.68, 1):
        ax.fill_between(X_star, mu_star - 1.0 * sd_star, mu_star + 1.0 * sd_star, alpha=0.2, color=PRIMARY_LINE_COLOR, label="68% band")
        ax.errorbar(X, y, yerr=indep_sd, fmt="o", ms=3.5, lw=1.0, color="#c75000", label=r"$\widetilde{D}_{b,v}\pm\widehat{\sigma}_{b,v}$")
    elif ci_level in (0.95, 2):
        ax.fill_between(X_star, mu_star - 1.96 * sd_star, mu_star + 1.96 * sd_star, alpha=0.15, color=PRIMARY_LINE_COLOR, label="95% band")
        ax.errorbar(X, y, yerr=2 * indep_sd, fmt="s", ms=3.5, lw=1.0, color="#1b8a5a", label=r"$\widetilde{D}_{b,v}\pm2\widehat{\sigma}_{b,v}$")
    else:
        raise ValueError(f"Unsupported ci_level={ci_level}")

    ax.set_xlabel(r"$z\ \text{(mm)}$", fontsize=_fs_label())
    ax.set_ylabel(r"Dose along core $D_b(z)$ (Gy)", fontsize=_fs_label())
    ymin, ymax = ax.get_ylim()
    if np.isfinite(ymax):
        ax.set_ylim(bottom=0 if ymin < 0 else ymin, top=ymax)
    elif ymin < 0:
        ax.set_ylim(bottom=0)
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    mean_sd_mc, mean_sd_gp, shrink = _compute_shrinkage_stats(gp_res)
    metrics_str = rf"$\hat{{\ell}}_b = {gp_res['hyperparams'].ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {shrink:.1f}\%$"
    # Reorder legend entries: mu, 68% band, 95% band, 1σ, 2σ
    handles, labels = ax.get_legend_handles_labels()
    order_keys = [
        r"$\mu^{\mathrm{GP}}_{b,v}$",
        "68% band",
        "95% band",
        r"$\widetilde{D}_{b,v}\pm\widehat{\sigma}_{b,v}$",
        r"$\widetilde{D}_{b,v}\pm2\widehat{\sigma}_{b,v}$",
    ]
    order_map = {lab: i for i, lab in enumerate(order_keys)}
    ordered = sorted(zip(handles, labels), key=lambda hl: order_map.get(hl[1], 99))
    if ordered:
        handles, labels = zip(*ordered)
        handles, labels = list(handles), list(labels)
    top = _finalize_legend_and_header(
        ax,
        header=metrics_str,
        ncol=len(handles) if ordered else 2,
        header_loc="center",
        header_fontsize=None,
        handles=handles if ordered else None,
        labels=labels if ordered else None,
        row_major=True,
    )

    fig = ax.figure
    if title_on:
        fig.suptitle(f"GP profile — Patient {patient_id}, Bx {bx_index}", y=1.02, fontsize=_fs_title())
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show, tight_layout=False)


def plot_noise_profile_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    save_formats=("pdf", "svg"),
    figsize=PER_BIOPSY_FIGSIZE,
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    ylog: bool = False,
    title_on: bool = False,
    show: bool = False,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    per_voxel = gp_res.get("per_voxel")
    if per_voxel is None:
        raise ValueError("gp_res must contain 'per_voxel' with x_mm and var_n columns.")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(per_voxel["x_mm"], np.sqrt(np.maximum(per_voxel["var_n"], 0)), marker="o", ms=4, lw=1.2, color=PRIMARY_LINE_COLOR, label=r"MC SD $\widehat{\sigma}_{b,v}$")
    ax.plot(gp_res["X"], gp_res["sd_X"], marker="s", ms=3.5, lw=1.1, color=OVERLAY_LINE_COLOR, label=r"GP SD $\sigma^{\mathrm{GP}}_{b,v}$")
    ax.set_xlabel(r"$z\ \text{(mm)}$", fontsize=_fs_label())
    ax.set_ylabel(r"Dose standard deviation (Gy)", fontsize=_fs_label())
    if title_on:
        ax.set_title(f"Noise profile — Patient {patient_id}, Bx {bx_index}")
    if ylog:
        ax.set_yscale("log")
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    ax.legend(frameon=True, fancybox=True, fontsize=_fs_legend(), facecolor="white", edgecolor="black", framealpha=0.95)
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show, tight_layout=False)


def plot_uncertainty_reduction_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    save_formats=("pdf", "svg"),
    figsize=PER_BIOPSY_FIGSIZE,
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    title_on: bool = False,
    show: bool = False,
    label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    X = gp_res["X"]
    indep_sd = np.sqrt(np.maximum(gp_res["var_n"], 0))
    sd_X = gp_res["sd_X"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(X, indep_sd, "o-", ms=4, lw=1.2, label=r"MC SD $\widehat{\sigma}_{b,v}$", color=OVERLAY_LINE_COLOR)
    ax.plot(X, sd_X, "o-", ms=4, lw=1.2, label=r"GP SD $\sigma^{\mathrm{GP}}_{b,v}$", color=PRIMARY_LINE_COLOR)
    ax.fill_between(X, sd_X, indep_sd, where=indep_sd>=sd_X, color=PRIMARY_LINE_COLOR, alpha=0.12)
    fs_label = _fs_label(label_fontsize)
    ax.set_xlabel(r"$z\ \text{(mm)}$", fontsize=fs_label)
    ax.set_ylabel(r"Dose standard deviation (Gy)", fontsize=fs_label)
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    # Annotate integrated reduction (header string)
    spacing = float(gpr_pf._safe_spacing(X))
    int_mc = float(np.nansum(indep_sd) * spacing)
    int_gp = float(np.nansum(sd_X) * spacing)
    red_pct = 100.0 * (1 - int_gp / int_mc) if int_mc > 0 else np.nan
    metrics_str = rf"$\hat{{\ell}}_b = {gp_res['hyperparams'].ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {red_pct:.1f}\%$"
    top = _finalize_legend_and_header(ax, header=metrics_str, ncol=2, header_loc="center", header_fontsize=_fs_legend(legend_fontsize))
    fig = ax.figure
    if title_on:
        fig.suptitle(f"Uncertainty reduction — Patient {patient_id}, Bx {bx_index}", y=1.02, fontsize=_fs_title())
    # layout handled by _finalize_legend_and_header (figure height expansion)
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show, tight_layout=False)


def plot_uncertainty_ratio_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    save_formats=("pdf", "svg"),
    figsize=PER_BIOPSY_FIGSIZE,
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    title_on: bool = False,
    show: bool = False,
    label_fontsize: int | None = None,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    X = gp_res["X"]
    indep_sd = np.sqrt(np.maximum(gp_res["var_n"], 0))
    ratio = np.divide(indep_sd, gp_res["sd_X"], out=np.ones_like(indep_sd), where=gp_res["sd_X"] > 0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(X, ratio, "-o", ms=4, lw=1.2, color=PRIMARY_LINE_COLOR)
    ax.axhline(1.0, color="black", lw=0.9, ls="--", alpha=0.7, label=r"$R_{b,v}=1$")
    ax.axhline(1.25, color="#c75000", lw=0.9, ls=":", alpha=0.7, label=r"$R_{b,v}=1.25$")
    ax.axhline(1.5, color="#7a5195", lw=0.9, ls=":", alpha=0.7, label=r"$R_{b,v}=1.5$")
    ax.fill_between(ax.get_xlim(), 1.25, ax.get_ylim()[1], color="#c75000", alpha=0.08)
    fs_label = _fs_label(label_fontsize)
    ax.set_xlabel(r"$z\ \text{(mm)}$", fontsize=fs_label)
    ax.set_ylabel(r"$R_{b,v} = \widehat{\sigma}_{b,v} / \sigma^{\mathrm{GP}}_{b,v}$", fontsize=fs_label)
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    mean_sd_mc, mean_sd_gp, shrink = _compute_shrinkage_stats(gp_res)
    metrics_str = rf"$\hat{{\ell}}_b = {gp_res['hyperparams'].ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {shrink:.1f}\%$"
    top = _finalize_legend_and_header(ax, header=metrics_str, ncol=3, header_loc="center", header_fontsize=None)
    fig = ax.figure
    if title_on:
        fig.suptitle(f"Uncertainty ratio — Patient {patient_id}, Bx {bx_index}", y=1.02, fontsize=_fs_title())
    # layout handled by _finalize_legend_and_header (figure height expansion)
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show, tight_layout=False)


def plot_residuals_vs_z_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    save_formats=("pdf", "svg"),
    figsize=PER_BIOPSY_FIGSIZE,
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    standardized: bool = False,
    xlim: tuple | None = None,
    show: bool = False,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    X = gp_res["X"]
    y = gp_res["y"]
    mu_X = gp_res["mu_X"]
    sd_X = gp_res["sd_X"]
    res = y - mu_X
    if standardized:
        res = np.divide(res, np.maximum(sd_X, 1e-12))
        y_label = r"Standardized residual $r^{\mathrm{std}}_{b,v}$"
    else:
        y_label = r"Residual $r_{b,v}$ (Gy)"
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="black", lw=1.0, alpha=0.7)
    if standardized:
        for lvl, color in zip([1, 2, 3], ["#b0b0b0", "#c75000", "#7a5195"]):
            ax.axhline(lvl, color=color, lw=0.9, ls="--", alpha=0.6)
            ax.axhline(-lvl, color=color, lw=0.9, ls="--", alpha=0.6)
    sigma_ref = np.nanmedian(sd_X)
    if sigma_ref > 0 and not standardized:
        ax.fill_between(X, -sigma_ref, sigma_ref, color=PRIMARY_LINE_COLOR, alpha=0.08)
    ax.scatter(X, res, s=28, color=PRIMARY_LINE_COLOR, alpha=0.9, edgecolors="black", linewidths=0.4)
    if xlim is None:
        xmin, xmax = np.nanmin(X), np.nanmax(X)
        pad = 0.02 * (xmax - xmin) if xmax > xmin else 1.0
        xlim = (xmin - pad, xmax + pad)
    ax.set_xlim(xlim)
    if standardized:
        max_abs = float(np.nanmax(np.abs(res))) if res.size else 0
        lim = max(3.5, np.ceil(max_abs * 2) / 2 if max_abs > 3.5 else 3.5)
        ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$z$ (mm)", fontsize=_fs_label())
    ax.set_ylabel(y_label, fontsize=_fs_label())
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    mean_mc, mean_gp, shrink = _compute_shrinkage_stats(gp_res)
    metrics_str = rf"$\hat{{\ell}}_b = {gp_res['hyperparams'].ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {shrink:.1f}\%$"
    top = _finalize_legend_and_header(ax, header=metrics_str, ncol=1, header_loc="center", header_fontsize=None)
    fig = ax.figure
    # layout handled by _finalize_legend_and_header (figure height expansion)
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show, tight_layout=False)


def plot_standardized_residuals_hist_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    save_formats=("pdf", "svg"),
    figsize=PER_BIOPSY_FIGSIZE,
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    show: bool = False,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    res_std = (gp_res["y"] - gp_res["mu_X"]) / np.maximum(gp_res["sd_X"], 1e-12)
    res_std = res_std[np.isfinite(res_std)]
    bins = _fd_bins(res_std, min_bins=None, max_bins=None, context=file_name_base)
    fig, ax = plt.subplots(figsize=figsize)
    counts, edges, _ = ax.hist(
        res_std,
        bins=bins,
        density=True,
        alpha=0.5,
        color=HIST_FILL_COLOR,
        edgecolor="black",
        linewidth=0.5,
        histtype="stepfilled",
    )
    ax.hist(
        res_std,
        bins=bins,
        density=True,
        histtype="step",
        color="0.4",
        linewidth=0.5,
    )
    # rug
    ymin = ax.get_ylim()[0]
    ax.plot(res_std, np.full_like(res_std, ymin + 0.01*(ax.get_ylim()[1]-ymin)), "|", color=OVERLAY_LINE_COLOR, alpha=0.6, markersize=6)
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, stats.norm.pdf(xs), color=OVERLAY_LINE_COLOR, lw=1.2, label=r"$\mathcal{N}(0,1)$")
    ax.axvline(0, color="black", lw=0.9, ls="-", label=r"$r^{\mathrm{std}}=0$")
    m = float(np.nanmean(res_std))
    ax.axvline(m, color="red", lw=0.9, ls="-", label="Mean")
    lim = max(3, np.percentile(np.abs(res_std), 99, method="linear") if res_std.size else 3)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(r"Standardized residual $r^{\mathrm{std}}_{b,v}$", fontsize=_fs_label())
    ax.set_ylabel("Density", fontsize=_fs_label())
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    bin_width = float(np.nanmean(np.diff(edges))) if edges.size > 1 else np.nan
    s = float(np.nanstd(res_std, ddof=1)) if res_std.size > 1 else np.nan
    ann = "\n".join([
        rf"$\mathrm{{mean}} = {m:.2f}\;\mathrm{{(red\,line)}}$",
        rf"$\mathrm{{SD}} = {s:.2f}$",
        rf"$\mathrm{{bin\ width}} = {bin_width:.3f}$",
    ])
    ax.text(
        0.98,
        0.92,
        ann,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=_fs_legend(),
        bbox=ANNOT_BBOX,
    )
    mean_mc, mean_gp, shrink = _compute_shrinkage_stats(gp_res)
    metrics_str = rf"$\hat{{\ell}}_b = {gp_res['hyperparams'].ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {shrink:.1f}\%$"
    top = _finalize_legend_and_header(ax, header=metrics_str, ncol=1, header_loc="center", header_fontsize=None)
    fig = ax.figure
    # layout handled by _finalize_legend_and_header (figure height expansion)
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show, tight_layout=False)


def plot_standardized_residuals_qq_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    save_formats=("pdf", "svg"),
    figsize=PER_BIOPSY_FIGSIZE,
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    show: bool = False,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    res_std = (gp_res["y"] - gp_res["mu_X"]) / np.maximum(gp_res["sd_X"], 1e-12)
    res_std = res_std[np.isfinite(res_std)]
    n = len(res_std)
    if n == 0:
        return []
    res_sorted = np.sort(res_std)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = stats.norm.ppf(probs)
    lim = max(3, np.percentile(np.abs(res_sorted), 99, method="linear"))
    lim = max(lim, np.max(np.abs(theo)))
    lim = np.ceil(lim * 2) / 2
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(theo, res_sorted, s=22, color=PRIMARY_LINE_COLOR, alpha=0.9, edgecolors="black", linewidths=0.4)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0, label=r"$y=x$")
    # 95% envelope
    i = np.arange(1, n + 1)
    lower_p = stats.beta.ppf(0.025, i, n + 1 - i)
    upper_p = stats.beta.ppf(0.975, i, n + 1 - i)
    lower = stats.norm.ppf(lower_p)
    upper = stats.norm.ppf(upper_p)
    ax.fill_between(
        theo,
        lower,
        upper,
        color=OVERLAY_LINE_COLOR,
        alpha=0.12,
        label="95% envelope",
        edgecolor="none",
        linewidth=0,
    )
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"Theoretical quantiles $\Phi^{-1}(p)$", fontsize=_fs_label())
    ax.set_ylabel(r"Sample quantiles of $r^{\mathrm{std}}_{b,v}$", fontsize=_fs_label())
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    mean_mc, mean_gp, shrink = _compute_shrinkage_stats(gp_res)
    metrics_str = rf"$\hat{{\ell}}_b = {gp_res['hyperparams'].ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {shrink:.1f}\%$"
    top = _finalize_legend_and_header(ax, header=metrics_str, ncol=2, header_loc="center", header_fontsize=None)
    fig = ax.figure
    # layout handled by _finalize_legend_and_header (figure height expansion)
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show, tight_layout=False)


def plot_standardized_residuals_ecdf_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    save_formats=("pdf", "svg"),
    figsize=PER_BIOPSY_FIGSIZE,
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    show: bool = False,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    res_std = (gp_res["y"] - gp_res["mu_X"]) / np.maximum(gp_res["sd_X"], 1e-12)
    res_std = res_std[np.isfinite(res_std)]
    n = len(res_std)
    if n == 0:
        return []
    data = np.sort(res_std)
    y = np.arange(1, n + 1) / n
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(data, y, where="post", color=PRIMARY_LINE_COLOR, label="ECDF")
    xs = np.linspace(min(-4, data.min()), max(4, data.max()), 400)
    ax.plot(xs, stats.norm.cdf(xs), color=OVERLAY_LINE_COLOR, lw=1.2, label=r"$\Phi(x)$")
    ax.set_xlabel(r"Standardized residual $r^{\mathrm{std}}_{b,v}$", fontsize=_fs_label())
    ax.set_ylabel("Empirical CDF", fontsize=_fs_label())
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    mean_mc, mean_gp, shrink = _compute_shrinkage_stats(gp_res)
    metrics_str = rf"$\hat{{\ell}}_b = {gp_res['hyperparams'].ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {shrink:.1f}\%$"
    top = _finalize_legend_and_header(ax, header=metrics_str, ncol=2, header_loc="center", header_fontsize=None)
    fig = ax.figure
    # layout handled by _finalize_legend_and_header (figure height expansion)
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show, tight_layout=False)


def plot_residuals_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    save_formats=("pdf", "svg"),
    figsize=(9.0, 4.2),
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    title_on: bool = True,
    show: bool = False,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    X = gp_res["X"]
    y = gp_res["y"]
    mu_X = gp_res["mu_X"]
    sd_X = gp_res["sd_X"]
    res = y - mu_X
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].axhline(0, color="black", lw=1.0, alpha=0.6)
    axes[0].plot(X, res, "o-", ms=4, lw=1.1, color=PRIMARY_LINE_COLOR)
    axes[0].set_xlabel(r"$z\ \text{(mm)}$", fontsize=_fs_label())
    axes[0].set_ylabel(r"$r_{b,v} = \widetilde{D}_{b,v} - \mu^{\mathrm{GP}}_{b,v}\ \text{(Gy)}$", fontsize=_fs_label())
    _apply_axis_style(axes[0])
    _apply_per_biopsy_ticks(axes[0])

    axes[1].hist(
        res / np.maximum(sd_X, 1e-12),
        bins=20,
        density=True,
        alpha=0.5,
        color=HIST_FILL_COLOR,
        edgecolor="black",
        linewidth=0.5,
        histtype="stepfilled",
    )
    axes[1].hist(
        res / np.maximum(sd_X, 1e-12),
        bins=20,
        density=True,
        histtype="step",
        color="0.4",
        linewidth=0.5,
    )
    axes[1].set_xlabel(r"Standardised residual $r^{\mathrm{std}}_{b,v}$", fontsize=_fs_label())
    axes[1].set_ylabel("Density", fontsize=_fs_label())
    # overlay standard normal
    xs = np.linspace(-4, 4, 200)
    axes[1].plot(xs, 1/np.sqrt(2*np.pi)*np.exp(-0.5*xs**2), color=OVERLAY_LINE_COLOR, lw=1.2, label=r"$\mathcal{N}(0,1)$")
    _apply_axis_style(axes[1])
    _apply_per_biopsy_ticks(axes[1])
    if title_on:
        fig.suptitle(f"Diagnostics — Patient {patient_id}, Bx {bx_index}", fontsize=_fs_title())
    mean_rs = float(np.nanmean(res / np.maximum(sd_X, 1e-12)))
    std_rs = float(np.nanstd(res / np.maximum(sd_X, 1e-12)))
    axes[1].text(
        0.98,
        0.95,
        "\n".join([
            f"mean={mean_rs:.2f}",
            f"sd={std_rs:.2f}",
        ]),
        ha="right",
        va="top",
        transform=axes[1].transAxes,
        fontsize=_fs_legend(),
        bbox=ANNOT_BBOX,
    )
    axes[1].legend(frameon=True, fancybox=True, facecolor="white", edgecolor="black", framealpha=0.9, fontsize=_fs_legend())
    fig.tight_layout()
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show)


def plot_variogram_overlay_production(
    semivariogram_df: pd.DataFrame,
    patient_id,
    bx_index,
    hyperparams,
    *,
    save_dir: Path,
    file_name_base: str,
    save_formats=("pdf", "svg"),
    figsize=PER_BIOPSY_FIGSIZE,
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    title_on: bool = False,
    show: bool = False,
    add_sill: bool = False,
    add_nugget: bool = False,
    metrics_str: str | None = None,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    sv = semivariogram_df.query("`Patient ID`==@patient_id and `Bx index`==@bx_index").sort_values("h_mm")
    h = sv["h_mm"].to_numpy(float)
    gamma_hat = sv["semivariance"].to_numpy(float)
    kernel = getattr(hyperparams, "kernel", "matern")
    if kernel == "rbf":
        gamma_model = gpr_pf.rbf_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0) + hyperparams.nugget
        label_model = r"Fitted $\gamma_b(h)$ (RBF)"
    elif kernel == "exp":
        gamma_model = gpr_pf.exp_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0) + hyperparams.nugget
        label_model = r"Fitted $\gamma_b(h)$ (Exp)"
    else:
        label_model = rf"Fitted $\gamma_b(h)$ (Matérn, $\nu={hyperparams.nu}$)"
        gamma_model = gpr_pf.matern_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0, hyperparams.nu) + hyperparams.nugget

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(h, gamma_hat, "o", ms=4, color=PRIMARY_LINE_COLOR, label=r"Empirical $\widehat{\gamma}_b(h)$")
    ax.plot(h, gamma_model, "-", lw=2.0, color=OVERLAY_LINE_COLOR, label=label_model)
    if add_sill:
        ax.axhline(hyperparams.sigma_f2 + hyperparams.nugget, color="#bbbbbb", ls="--", lw=0.9, label=r"Sill $\sigma_{f,b}^2$")
    if add_nugget:
        ax.axhline(hyperparams.nugget, color="#999999", ls=":", lw=0.9, label=r"Nugget $\tau_b^2$")
    ax.set_xlabel(r"Lag $h\ \text{(mm)}$", fontsize=_fs_label())
    ax.set_ylabel(r"Semivariance $\gamma_b(h)$ (Gy$^2$)", fontsize=_fs_label())
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    if metrics_str is None:
        metrics_str = rf"$\hat{{\ell}}_b = {hyperparams.ell:.1f}~\mathrm{{mm}}$"
    top = _finalize_legend_and_header(ax, header=metrics_str, ncol=2, header_loc="center", header_fontsize=None)
    if title_on:
        fig.suptitle(f"Variogram overlay — Patient {patient_id}, Bx {bx_index}", y=1.02, fontsize=_fs_title())
    # layout handled by _finalize_legend_and_header (figure height expansion)
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show)


def plot_variogram_and_profile_pair(
    semivariogram_df: pd.DataFrame,
    patient_id,
    bx_index,
    gp_res: dict,
    *,
    save_dir: Path,
    file_name_base: str = "variogram_profile_pair",
    save_formats=("pdf", "svg"),
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    show: bool = False,
    add_sill: bool = False,
    add_nugget: bool = False,
):
    """
    Two-panel figure: semivariogram overlay (left) + GP profile (right),
    with aligned plot areas.
    """
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(PER_BIOPSY_FIGSIZE[0] * 2, PER_BIOPSY_FIGSIZE[1]),
        gridspec_kw={"wspace": 0.25},
    )

    # Left: variogram overlay
    ax = axes[0]
    sv = semivariogram_df.query("`Patient ID`==@patient_id and `Bx index`==@bx_index").sort_values("h_mm")
    h = sv["h_mm"].to_numpy(float)
    gamma_hat = sv["semivariance"].to_numpy(float)
    hyperparams = gp_res["hyperparams"]
    kernel = getattr(hyperparams, "kernel", "matern")
    if kernel == "rbf":
        gamma_model = gpr_pf.rbf_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0) + hyperparams.nugget
        label_model = r"Fitted $\gamma_b(h)$ (RBF)"
    elif kernel == "exp":
        gamma_model = gpr_pf.exp_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0) + hyperparams.nugget
        label_model = r"Fitted $\gamma_b(h)$ (Exp)"
    else:
        label_model = rf"Fitted $\gamma_b(h)$ (Matérn, $\nu={hyperparams.nu}$)"
        gamma_model = gpr_pf.matern_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0, hyperparams.nu) + hyperparams.nugget

    ax.plot(h, gamma_hat, "o", ms=4, color=PRIMARY_LINE_COLOR, label=r"Empirical $\widehat{\gamma}_b(h)$")
    ax.plot(h, gamma_model, "-", lw=2.0, color=OVERLAY_LINE_COLOR, label=label_model)
    if add_sill:
        ax.axhline(hyperparams.sigma_f2 + hyperparams.nugget, color="#bbbbbb", ls="--", lw=0.9, label=r"Sill $\sigma_{f,b}^2$")
    if add_nugget:
        ax.axhline(hyperparams.nugget, color="#999999", ls=":", lw=0.9, label=r"Nugget $\tau_b^2$")
    ax.set_xlabel(r"Lag $h\ \text{(mm)}$", fontsize=_fs_label())
    ax.set_ylabel(r"Semivariance $\gamma_b(h)$ (Gy$^2$)", fontsize=_fs_label())
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    metrics_left = rf"$\hat{{\ell}}_b = {hyperparams.ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {100.0 * (1 - np.nanmean(gp_res['sd_X']) / np.nanmean(np.sqrt(np.maximum(gp_res['var_n'], 0)))):.1f}\%$"
    top_left = _finalize_legend_and_header(ax, header=metrics_left, ncol=2, header_loc="center", header_fontsize=None, legend_width_mode="subplot")

    # Right: GP profile
    ax = axes[1]
    X_star = gp_res["X_star"]
    mu_star = gp_res["mu_star"]
    sd_star = gp_res["sd_star"]
    X = gp_res["X"]
    y = gp_res["y"]
    indep_sd = np.sqrt(np.maximum(gp_res["var_n"], 0))
    ax.plot(X_star, mu_star, lw=2.4, color=PRIMARY_LINE_COLOR, label=r"$\mu^{\mathrm{GP}}_{b,v}$")
    ax.fill_between(X_star, mu_star - 1.96 * sd_star, mu_star + 1.96 * sd_star, alpha=0.12, color=PRIMARY_LINE_COLOR, label="95% band")
    ax.fill_between(X_star, mu_star - 1.0 * sd_star, mu_star + 1.0 * sd_star, alpha=0.22, color=PRIMARY_LINE_COLOR, label="68% band")
    ax.errorbar(X, y, yerr=2 * indep_sd, fmt="s", ms=3.5, lw=1.0, color="#1b8a5a", label=r"$\widetilde{D}_{b,v}\pm2\widehat{\sigma}_{b,v}$")
    ax.errorbar(X, y, yerr=indep_sd, fmt="o", ms=3.5, lw=1.0, color="#c75000", label=r"$\widetilde{D}_{b,v}\pm\widehat{\sigma}_{b,v}$")
    ax.set_xlabel(r"$z\ \text{(mm)}$", fontsize=_fs_label())
    ax.set_ylabel(r"Dose along core $D_b(z)$ (Gy)", fontsize=_fs_label())
    ymin, ymax = ax.get_ylim()
    if np.isfinite(ymax):
        ax.set_ylim(bottom=0 if ymin < 0 else ymin, top=ymax)
    elif ymin < 0:
        ax.set_ylim(bottom=0)
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    # Legend order for profile
    handles, labels = ax.get_legend_handles_labels()
    order_keys = [
        r"$\mu^{\mathrm{GP}}_{b,v}$",
        "68% band",
        "95% band",
        r"$\widetilde{D}_{b,v}\pm\widehat{\sigma}_{b,v}$",
        r"$\widetilde{D}_{b,v}\pm2\widehat{\sigma}_{b,v}$",
    ]
    order_map = {lab: i for i, lab in enumerate(order_keys)}
    ordered = sorted(zip(handles, labels), key=lambda hl: order_map.get(hl[1], 99))
    if ordered:
        handles, labels = zip(*ordered)
        handles, labels = list(handles), list(labels)
    metrics_right = rf"$\hat{{\ell}}_b = {hyperparams.ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {100.0 * (1 - np.nanmean(gp_res['sd_X']) / np.nanmean(np.sqrt(np.maximum(gp_res['var_n'], 0)))):.1f}\%$"
    top_right = _finalize_legend_and_header(
        ax,
        header=metrics_right,
        ncol=len(handles),
        header_loc="center",
        header_fontsize=None,
        handles=handles,
        labels=labels,
        row_major=True,
        legend_width_mode="subplot",
    )

    # Align axes heights/positions
    pos_l = axes[0].get_position()
    pos_r = axes[1].get_position()
    new_y0 = min(pos_l.y0, pos_r.y0)
    new_h = min(pos_l.height, pos_r.height)
    axes[0].set_position([pos_l.x0, new_y0, pos_l.width, new_h])
    axes[1].set_position([pos_r.x0, new_y0, pos_r.width, new_h])

    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show, tight_layout=False)


def plot_uncertainty_pair(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str = "uncertainty_pair",
    save_formats=("pdf", "svg"),
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    show: bool = False,
):
    """
    Two-panel figure: uncertainty reduction (left) + uncertainty ratio (right),
    with aligned plot areas.
    """
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(PER_BIOPSY_FIGSIZE[0] * 2, PER_BIOPSY_FIGSIZE[1]),
        gridspec_kw={"wspace": 0.25},
    )

    X = gp_res["X"]
    indep_sd = np.sqrt(np.maximum(gp_res["var_n"], 0))
    sd_X = gp_res["sd_X"]

    # Left: uncertainty reduction
    ax = axes[0]
    ax.plot(X, indep_sd, "o-", ms=4, lw=1.2, label=r"MC SD $\widehat{\sigma}_{b,v}$", color=OVERLAY_LINE_COLOR)
    ax.plot(X, sd_X, "o-", ms=4, lw=1.2, label=r"GP SD $\sigma^{\mathrm{GP}}_{b,v}$", color=PRIMARY_LINE_COLOR)
    ax.fill_between(X, sd_X, indep_sd, where=indep_sd>=sd_X, color=PRIMARY_LINE_COLOR, alpha=0.12)
    ax.set_xlabel(r"$z\ \text{(mm)}$", fontsize=_fs_label())
    ax.set_ylabel(r"Dose standard deviation (Gy)", fontsize=_fs_label())
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    spacing = float(gpr_pf._safe_spacing(X))
    int_mc = float(np.nansum(indep_sd) * spacing)
    int_gp = float(np.nansum(sd_X) * spacing)
    red_pct = 100.0 * (1 - int_gp / int_mc) if int_mc > 0 else np.nan
    metrics_left = rf"$\hat{{\ell}}_b = {gp_res['hyperparams'].ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {red_pct:.1f}\%$"
    top_left = _finalize_legend_and_header(ax, header=metrics_left, ncol=2, header_loc="center", header_fontsize=None, legend_width_mode="subplot")

    # Right: uncertainty ratio
    ax = axes[1]
    ratio = np.divide(indep_sd, sd_X, out=np.ones_like(indep_sd), where=sd_X > 0)
    ax.plot(X, ratio, "-o", ms=4, lw=1.2, color=PRIMARY_LINE_COLOR)
    ax.axhline(1.0, color="black", lw=0.9, ls="--", alpha=0.7, label=r"$R_{b,v}=1$")
    ax.axhline(1.25, color="#c75000", lw=0.9, ls=":", alpha=0.7, label=r"$R_{b,v}=1.25$")
    ax.axhline(1.5, color="#7a5195", lw=0.9, ls=":", alpha=0.7, label=r"$R_{b,v}=1.5$")
    ax.fill_between(ax.get_xlim(), 1.25, ax.get_ylim()[1], color="#c75000", alpha=0.08)
    ax.set_xlabel(r"$z\ \text{(mm)}$", fontsize=_fs_label())
    ax.set_ylabel(r"$R_{b,v} = \widehat{\sigma}_{b,v} / \sigma^{\mathrm{GP}}_{b,v}$", fontsize=_fs_label())
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    metrics_right = rf"$\hat{{\ell}}_b = {gp_res['hyperparams'].ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {100.0 * (1 - np.nanmean(sd_X) / np.nanmean(indep_sd)):.1f}\%$"
    top_right = _finalize_legend_and_header(ax, header=metrics_right, ncol=3, header_loc="center", header_fontsize=None, legend_width_mode="subplot")

    # Align axes heights/positions
    pos_l = axes[0].get_position()
    pos_r = axes[1].get_position()
    new_y0 = min(pos_l.y0, pos_r.y0)
    new_h = min(pos_l.height, pos_r.height)
    axes[0].set_position([pos_l.x0, new_y0, pos_l.width, new_h])
    axes[1].set_position([pos_r.x0, new_y0, pos_r.width, new_h])

    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, show=show, tight_layout=False)


def cohort_plots_production(
    metrics_df: pd.DataFrame,
    save_dir: Path,
    save_formats=("pdf", "svg"),
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    dpi: int = 400,
    boxplot_metrics=("mean_ratio", "integrated_reduction", "frac_high"),
    boxplot_label_fontsize: int | None = None,
):
    _setup_matplotlib_defaults(font_scale=font_scale, seaborn_style=seaborn_style, seaborn_context=seaborn_context)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _hist(series, xlabel, fname, unit_label: str, var_label: str, figsize=PER_BIOPSY_FIGSIZE, bins_override: int | None = None):
        fig, ax = plt.subplots(figsize=figsize)
        data = series.dropna().to_numpy(dtype=float)
        # Bin count is selected via Freedman–Diaconis rule (_fd_bins); width is derived from data range.
        bins = bins_override if bins_override is not None else _fd_bins(data, min_bins=None, max_bins=None, context=fname)
        hist_counts, hist_edges = np.histogram(data, bins=bins)
        bin_width = float(np.nanmean(np.diff(hist_edges))) if hist_edges.size > 1 else np.nan
        ax.hist(data, bins=bins, alpha=0.5, color=HIST_FILL_COLOR, edgecolor="black", linewidth=0.5, histtype="stepfilled")
        ax.hist(data, bins=bins, histtype="step", color="0.4", linewidth=0.5)
        ax.set_xlabel(xlabel, fontsize=_fs_label())
        ax.set_ylabel("Number of biopsies", fontsize=_fs_label())
        med = float(np.nanmedian(data)) if data.size else np.nan
        ax.axvline(med, color="red", lw=0.9, ls="-")
        unit_txt = f"\\ \\mathrm{{{unit_label}}}" if unit_label else ""
        # Format selection: median/bin width shown to two decimals (per request)
        if unit_label == "mm":
            bw_fmt = "{:.2f}"
            med_fmt = "{:.2f}"
        elif "tau_b^2" in var_label:
            bw_fmt = "{:.2f}"
            med_fmt = "{:.2f}"
        else:
            bw_fmt = "{:.2f}"
            med_fmt = "{:.2f}"
        bw_txt = f"\\mathrm{{bin\\ width}} = {bw_fmt.format(bin_width)}{unit_txt}" if np.isfinite(bin_width) else "\\mathrm{bin\\ width}=nan"
        med_txt = (
            f"\\mathrm{{median}}({var_label}) = {med_fmt.format(med)}{unit_txt} \\;\\text{{(red\\ line)}}"
            if np.isfinite(med)
            else f"\\mathrm{{median}}({var_label})=nan"
        )
        ann_lines = [
            rf"$n={data.size}$",
            rf"${med_txt}$",
            rf"${bw_txt}$",
        ]
        ax.text(
            0.98,
            0.95,
            "\n".join(ann_lines),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=_fs_legend(),
            bbox=ANNOT_BBOX,
        )
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        _apply_axis_style(ax)
        _apply_per_biopsy_ticks(ax)
        print(f"[{fname}] bin width: {bin_width:.4g} {unit_label}".strip())
        fig.tight_layout()
        _save_figure(fig, save_dir / fname, formats=save_formats, dpi=dpi, create_subdir_for_stem=False)

    _hist(metrics_df["mean_ratio"], r"Mean shrinkage ratio $\overline{R}_b$", "cohort_hist_mean_ratio", unit_label="", var_label=r"\overline{R}_b")
    _hist(metrics_df["ell"], r"Fitted axial coherence length $\widehat{\ell}_b$ (mm)", "cohort_hist_length_scale", unit_label="mm", var_label=r"\ell_b", bins_override=4)
    _hist(metrics_df.get("nugget_fraction", metrics_df["nugget"]), r"Nugget fraction $\tau_b^2 / (\sigma_{f,b}^2 + \tau_b^2)$", "cohort_hist_nugget_fraction", unit_label="", var_label=r"\tau_b^2/(\sigma_{f,b}^2+\tau_b^2)")
    if "sv_rmse" in metrics_df.columns:
        _hist(metrics_df["sv_rmse"], r"Semivariogram $\mathrm{RMSE}_b^{(\gamma)}$ (Gy$^2$)", "cohort_hist_variogram_rmse", unit_label="Gy^2", var_label=r"\mathrm{RMSE}_b^{(\gamma)}")

    # Boxplot of selected cohort metrics
    mean_ratio = metrics_df["mean_ratio"].dropna()
    integ_red = metrics_df.get("delta_int_percent", metrics_df.get("pct_reduction_integ_sd")).dropna()
    if isinstance(integ_red, pd.Series):
        integ_red = integ_red / 100.0
    frac_high = metrics_df.get("pct_vox_ge_20", np.nan)
    if isinstance(frac_high, pd.Series):
        frac_high = (frac_high.dropna() / 100.0)
    else:
        frac_high = pd.Series([], dtype=float)

    metric_map = {
        "mean_ratio": (mean_ratio, r"Mean shrinkage ratio $\overline{R}_b$"),
        "integrated_reduction": (integ_red, r"Integrated SD reduction $\Delta_b^{\mathrm{int}}$"),
        "frac_high": (frac_high, r"Fraction with $R_{b,v} \geq 1.25$"),
        # backward-compatible aliases
        "mean": (mean_ratio, r"Mean shrinkage ratio $\overline{R}_b$"),
        "median": (frac_high, r"Fraction with $R_{b,v} \geq 1.25$"),
        "delta_int": (integ_red, r"Integrated SD reduction $\Delta_b^{\mathrm{int}}$"),
    }
    selected = [m for m in boxplot_metrics if m in metric_map]
    if selected:
        fig, ax = plt.subplots(figsize=COHORT_SQUARE_FIGSIZE)
        data = [metric_map[m][0] for m in selected]
        labels = [metric_map[m][1] for m in selected]
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.1),
            boxprops=dict(linewidth=1.0, facecolor=HIST_FILL_COLOR, alpha=0.3),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )
        xticks = ax.get_xticklabels()
        inferred_fs = xticks[0].get_fontsize() if xticks else 12
        ax.set_ylabel(r"Per-biopsy summary statistic", fontsize=_fs_label(boxplot_label_fontsize))
        # y-limits shared across selected metrics
        all_vals = pd.concat([d for d in data if isinstance(d, pd.Series)])
        ymax = float(np.nanmax(all_vals)) if all_vals.size else 1.0
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
        _apply_axis_style(ax)
        _apply_per_biopsy_ticks(ax)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        _save_figure(fig, save_dir / "cohort_boxplot_uncertainty_reduction", formats=save_formats, dpi=dpi, create_subdir_for_stem=False)

    fig, ax = plt.subplots(figsize=PER_BIOPSY_FIGSIZE)
    ax.scatter(metrics_df["mean_indep_sd"], metrics_df["mean_gp_sd"], s=24, alpha=0.85, color=PRIMARY_LINE_COLOR)
    lim_hi = float(np.nanmax([metrics_df["mean_indep_sd"].max(), metrics_df["mean_gp_sd"].max(), 0]))
    lims = [0, lim_hi * 1.05 if lim_hi > 0 else 1.0]
    ax.plot(lims, lims, "k--", lw=1.0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r"Mean MC SD $\overline{\widehat{\sigma}}_b\ \text{(Gy)}$", fontsize=_fs_label())
    ax.set_ylabel(r"Mean GP SD $\overline{\sigma}^{\mathrm{GP}}_b\ \text{(Gy)}$", fontsize=_fs_label())
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    fig.tight_layout()
    _save_figure(fig, save_dir / "cohort_mean_sd_scatter", formats=save_formats, dpi=dpi, create_subdir_for_stem=False)


def plot_mean_sd_scatter_with_fits_production(
    metrics_df: pd.DataFrame,
    reg_stats: pd.DataFrame,
    save_dir: Path,
    file_name_base: str = "cohort_mean_sd_scatter_with_fits",
    save_formats=("pdf", "svg"),
    show_ols: bool = False,
    show_deming: bool = True,
    show_deming_ci: bool = True,
    deming_ci_bootstrap: int = 1000,
    deming_ci_alpha: float = 0.05,
    add_origin_fit: bool = False,
    add_ci_ribbon: bool = True,
    add_pred_band: bool = False,
):
    x = metrics_df["mean_indep_sd"].to_numpy(dtype=float)
    y = metrics_df["mean_gp_sd"].to_numpy(dtype=float)
    msk = np.isfinite(x) & np.isfinite(y)
    x, y = x[msk], y[msk]
    s = reg_stats.iloc[0]
    lim_hi = float(np.nanmax([x.max() if x.size else 0, y.max() if y.size else 0]))
    lims = [0.0, lim_hi * 1.05 if lim_hi > 0 else 1.0]
    xs = np.linspace(lims[0], lims[1], 200)

    fig, ax = plt.subplots(figsize=COHORT_SQUARE_FIGSIZE)
    ax.scatter(x, y, s=24, alpha=0.85, color=PRIMARY_LINE_COLOR, label="Biopsies")
    ax.plot(lims, lims, "k--", lw=1.0, label="Identity")

    # Deming regression (default)
    deming_line = False
    deming_m = np.nan
    deming_b = np.nan
    if show_deming and np.isfinite(s.get("deming_slope", np.nan)) and np.isfinite(s.get("deming_intercept", np.nan)):
        deming_b, deming_m = float(s["deming_intercept"]), float(s["deming_slope"])
        ax.plot(xs, deming_b + deming_m * xs, lw=2.0, color="#2a9d8f", label="Deming fit")
        deming_line = True
        if show_deming_ci and x.size >= 3:
            rng = np.random.default_rng(0)
            yhat_samples = []
            for _ in range(int(deming_ci_bootstrap)):
                idx = rng.integers(0, x.size, size=x.size)
                xb = x[idx]
                yb = y[idx]
                xbar = xb.mean()
                ybar = yb.mean()
                Sxx = np.sum((xb - xbar) ** 2)
                Syy = np.sum((yb - ybar) ** 2)
                Sxy = np.sum((xb - xbar) * (yb - ybar))
                if Sxy == 0:
                    continue
                Delta = Syy - Sxx
                b_dem = (Delta + np.sqrt(Delta**2 + 4 * Sxy**2)) / (2 * Sxy)
                a_dem = ybar - b_dem * xbar
                yhat_samples.append(a_dem + b_dem * xs)
            if yhat_samples:
                yhat_arr = np.vstack(yhat_samples)
                lo = np.nanpercentile(yhat_arr, 100 * (deming_ci_alpha / 2.0), axis=0)
                hi = np.nanpercentile(yhat_arr, 100 * (1 - deming_ci_alpha / 2.0), axis=0)
                ax.fill_between(xs, lo, hi, color="#2a9d8f", alpha=0.15, label="95% CI (Deming)")

    # OLS regression (optional)
    if show_ols:
        have_ols = np.isfinite(s.get("ols_slope", np.nan)) and np.isfinite(s.get("ols_intercept", np.nan))
        if have_ols:
            a, b = float(s["ols_intercept"]), float(s["ols_slope"])
            ax.plot(xs, a + b * xs, lw=2.0, color="#c75000", label="OLS fit")
            if add_ci_ribbon:
                sigma2 = s.get("ols_sigma2", np.nan)
                xbar = s.get("ols_xbar", np.nan)
                Sxx = s.get("ols_Sxx", np.nan)
                df = s.get("ols_df", np.nan)
                tcrit = s.get("ols_tcrit", np.nan)
                if all(np.isfinite(v) for v in [sigma2, xbar, Sxx, df, tcrit]) and Sxx > 0 and df >= 1:
                    se_mean = np.sqrt(sigma2 * (1.0 / float(s["n"]) + (xs - xbar) ** 2 / Sxx))
                    yhat = a + b * xs
                    ax.fill_between(xs, yhat - tcrit * se_mean, yhat + tcrit * se_mean, alpha=0.15, color="#c75000", label="95% CI (mean)")
                    if add_pred_band:
                        se_pred = np.sqrt(sigma2 * (1.0 + 1.0 / float(s["n"]) + (xs - xbar) ** 2 / Sxx))
                        ax.fill_between(xs, yhat - tcrit * se_pred, yhat + tcrit * se_pred, alpha=0.10, color="#c75000", label="95% prediction")

    if add_origin_fit and np.isfinite(s.get("origin_slope", np.nan)):
        bo = float(s["origin_slope"])
        ax.plot(xs, bo * xs, lw=1.5, ls="-.", color="#7a5195", label=f"Through-origin: y={bo:.3f}x")

    ax.set_xlabel(r"Mean MC SD $\overline{\widehat{\sigma}}_b$ (Gy)", fontsize=_fs_label())
    ax.set_ylabel(r"Mean GP SD $\overline{\sigma}^{\mathrm{GP}}_b\ \text{(Gy)}$", fontsize=_fs_label())
    ax.set_xlim(lims); ax.set_ylim(lims)
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    # Place legend at upper-left with stable anchor for downstream annotation placement
    legend = ax.legend(
        frameon=True,
        fancybox=True,
        fontsize=_fs_legend(),
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
    )
    if legend:
        frame = legend.get_frame()
        frame.set_boxstyle("round,pad=0.25")
        frame.set_linewidth(0.6)
    # Annotation box for Deming results
    if deming_line:
        r = float(np.corrcoef(x, y)[0, 1]) if x.size and y.size else np.nan
        r2 = r ** 2 if np.isfinite(r) else np.nan
        ann = "\n".join([
            rf"$\mathrm{{slope}} = {deming_m:.2f}$",
            rf"$\mathrm{{intercept}} = {deming_b:.2f}\ \mathrm{{Gy}}$",
            rf"$R^2 = {r2:.2f}$",
        ])
        # Align annotation box left edge with legend and place just below it.
        fig = ax.figure
        ann_x, ann_y = 0.02, 0.75
        renderer = None
        if fig.canvas:
            fig.canvas.draw_idle()
            renderer = fig.canvas.get_renderer()
        if renderer is not None and legend is not None:
            lb = legend.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
            ann_x = lb.x0
            ann_y = lb.y0 - 0.02
            ann_y = max(0.02, ann_y)
        ax.text(
            ann_x,
            ann_y,
            ann,
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=_fs_legend(),
            bbox=ANNOT_BBOX,
        )
    fig.tight_layout()
    _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=400, create_subdir_for_stem=False)


def plot_kernel_sensitivity_mean_sd_with_fits(
    metrics_df: pd.DataFrame,
    save_dir: Path,
    file_name_base: str = "kernel_sensitivity_mean_sd_scatter_with_fits",
    save_formats=("pdf", "svg"),
    file_types=None,  # backward-compat with caller passing file_types
    deming_ci_bootstrap: int = 500,
    deming_ci_alpha: float = 0.05,
    legend_fontsize: int | None = 10,
):
    """
    Kernel sensitivity version of mean_sd scatter with Deming fits + CIs per kernel.
    Colors correspond to kernel_label; draws identity line, per-kernel scatter,
    Deming fit, and bootstrap CI ribbon.
    """
    if "kernel_label" not in metrics_df.columns:
        raise ValueError("metrics_df must contain column 'kernel_label'.")

    # allow file_types alias
    if file_types is not None:
        save_formats = file_types

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=COHORT_SQUARE_FIGSIZE)

    kernels = list(metrics_df["kernel_label"].dropna().unique())
    colors = KERNEL_PALETTE * ((len(kernels) // len(KERNEL_PALETTE)) + 1)
    lim_hi_candidates = []
    scatter_handles, fit_handles, ci_handles = [], [], []

    for k, color in zip(kernels, colors):
        sub = metrics_df.loc[metrics_df["kernel_label"] == k].copy()
        sub["kernel_pretty"] = sub["kernel_label"].map(KERNEL_LABEL_MAP).fillna(sub["kernel_label"])
        x = sub["mean_indep_sd"].to_numpy(dtype=float)
        y = sub["mean_gp_sd"].to_numpy(dtype=float)
        msk = np.isfinite(x) & np.isfinite(y)
        x, y = x[msk], y[msk]
        if not x.size:
            continue
        lim_hi_candidates.extend([x.max(), y.max()])
        pretty = sub["kernel_pretty"].iloc[0] if not sub["kernel_pretty"].empty else k
        sc = ax.scatter(x, y, s=22, alpha=0.85, color=color, edgecolors="white", linewidths=0.4, label=f"{pretty} biopsies")
        scatter_handles.append(sc)

        reg_stats = gpr_pf.fit_mean_sd_regressions(sub)
        s = reg_stats.iloc[0]
        if np.isfinite(s.get("deming_slope", np.nan)) and np.isfinite(s.get("deming_intercept", np.nan)) and x.size >= 3:
            lim_hi = float(np.nanmax([x.max(), y.max()]))
            xs = np.linspace(0.0, lim_hi * 1.05 if lim_hi > 0 else 1.0, 200)
            m = float(s["deming_slope"])
            b = float(s["deming_intercept"])
            line, = ax.plot(xs, b + m * xs, lw=2.0, color=color, label=f"{pretty} Deming fit")
            fit_handles.append(line)
            # bootstrap CI
            rng = np.random.default_rng(0)
            yhat_samples = []
            for _ in range(int(deming_ci_bootstrap)):
                idx = rng.integers(0, x.size, size=x.size)
                xb, yb = x[idx], y[idx]
                xbar, ybar = xb.mean(), yb.mean()
                Sxx = np.sum((xb - xbar) ** 2)
                Syy = np.sum((yb - ybar) ** 2)
                Sxy = np.sum((xb - xbar) * (yb - ybar))
                if Sxy == 0:
                    continue
                Delta = Syy - Sxx
                b_dem = (Delta + np.sqrt(Delta**2 + 4 * Sxy**2)) / (2 * Sxy)
                a_dem = ybar - b_dem * xbar
                yhat_samples.append(a_dem + b_dem * xs)
            if yhat_samples:
                yhat_arr = np.vstack(yhat_samples)
                lo = np.nanpercentile(yhat_arr, 100 * (deming_ci_alpha / 2.0), axis=0)
                hi = np.nanpercentile(yhat_arr, 100 * (1 - deming_ci_alpha / 2.0), axis=0)
                patch = ax.fill_between(xs, lo, hi, color=color, alpha=0.12, label=f"{pretty} 95% CI")
                ci_handles.append(patch)

    lim_hi = float(np.nanmax(lim_hi_candidates)) if lim_hi_candidates else 1.0
    lims = [0.0, lim_hi * 1.05 if lim_hi > 0 else 1.0]
    ax.set_xlim(lims); ax.set_ylim(lims)
    identity_handle, = ax.plot(lims, lims, "k--", lw=1.0, label="Identity")
    ax.set_xlabel(r"Mean MC SD $\overline{\widehat{\sigma}}_b$ (Gy)", fontsize=_fs_label())
    ax.set_ylabel(r"Mean GP SD $\overline{\sigma}^{\mathrm{GP}}_b\ \text{(Gy)}$", fontsize=_fs_label())
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)
    # Ordered legend: Identity → points → fits → CIs
    legend_handles = [identity_handle] + scatter_handles + fit_handles + ci_handles
    legend_labels = [h.get_label() for h in legend_handles]
    ax.legend(legend_handles, legend_labels, frameon=True, fancybox=True, fontsize=_fs_legend(legend_fontsize), facecolor="white", edgecolor="black", framealpha=1.0, loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=400, create_subdir_for_stem=False)


def plot_mean_sd_bland_altman_production(
    metrics_df: pd.DataFrame,
    save_dir: Path,
    file_name_base: str = "cohort_mean_sd_bland_altman",
    save_formats=("pdf", "svg"),
    source_csv_path: str | Path | None = None,
    dpi: int = 400,
    font_scale: float = 1.1,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    show_annotation: bool = True,
    nearline_fontsize: int | None = 12,
):
    """Production-quality Bland–Altman plot for mean MC vs GP SD."""
    _setup_matplotlib_defaults(
        font_scale=font_scale,
        seaborn_style=seaborn_style,
        seaborn_context=seaborn_context,
    )
    source_desc = str(source_csv_path) if source_csv_path is not None else "metrics_df (in-memory)"
    cols = ["mean_indep_sd", "mean_gp_sd"]
    print(f"[bland_altman] source: {source_desc}")
    print(f"[bland_altman] using columns: {cols}")

    x = metrics_df[cols[0]].to_numpy(dtype=float)
    y = metrics_df[cols[1]].to_numpy(dtype=float)
    msk = np.isfinite(x) & np.isfinite(y) & (x != 0)
    x, y = x[msk], y[msk]
    if x.size == 0:
        return []

    A = 0.5 * (x + y)
    D = 100.0 * (y - x) / x

    mean_diff = float(np.nanmean(D))
    sd_diff = float(np.nanstd(D, ddof=1))
    loa_low = mean_diff - 1.96 * sd_diff
    loa_high = mean_diff + 1.96 * sd_diff

    fig, ax = plt.subplots(figsize=PER_BIOPSY_FIGSIZE)
    ax.scatter(A, D, s=24, alpha=0.85, color=PRIMARY_LINE_COLOR, label="Biopsies")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.axhline(mean_diff, color=PRIMARY_LINE_COLOR, lw=1.6, alpha=0.9)
    ax.axhline(loa_low, color=PRIMARY_LINE_COLOR, lw=1.0, ls="--", alpha=0.6)
    ax.axhline(loa_high, color=PRIMARY_LINE_COLOR, lw=1.0, ls="--", alpha=0.6)

    ax.set_xlabel(r"Mean SD, $(\overline{\widehat{\sigma}}_b + \overline{\sigma}^{\mathrm{GP}}_b)/2$ (Gy)", fontsize=_fs_label())
    ax.set_ylabel(
        "Percent difference\n"
        r"($\overline{\sigma}^{\mathrm{GP}}_b$ vs $\overline{\widehat{\sigma}}_b$) (%)",
        fontsize=_fs_label(),
    )
    _apply_axis_style(ax)
    _apply_per_biopsy_ticks(ax)

    # Right-side line labels
    y_min, y_max = ax.get_ylim()
    y_pad = 0.02 * (y_max - y_min)
    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    if show_annotation:
        mean_lbl = "mean"
        hi_lbl = r"$+1.96\,\mathrm{SD}$"
        lo_lbl = r"$-1.96\,\mathrm{SD}$"
        sd_lbl = None
    else:
        mean_lbl = rf"$\mathrm{{mean}} = {mean_diff:.1f}\%$"
        hi_lbl = rf"$+1.96\,\mathrm{{SD}} = {loa_high:.1f}\%$"
        lo_lbl = rf"$-1.96\,\mathrm{{SD}} = {loa_low:.1f}\%$"
        sd_lbl = rf"$\mathrm{{SD}} = {sd_diff:.1f}\%$"
    label_box = dict(facecolor="white", edgecolor="none", alpha=1.0, boxstyle="round,pad=0.25")
    fs_near = _fs_annot(nearline_fontsize)
    label_shift = 0.25 * y_pad
    ax.text(0.98, mean_diff + 1.5 * y_pad + label_shift, mean_lbl, ha="right", va="bottom", transform=trans, fontsize=fs_near, bbox=label_box)
    ax.text(0.98, loa_high + y_pad + label_shift, hi_lbl, ha="right", va="bottom", transform=trans, fontsize=fs_near, bbox=label_box)
    ax.text(0.98, loa_low + y_pad + label_shift, lo_lbl, ha="right", va="bottom", transform=trans, fontsize=fs_near, bbox=label_box)
    if sd_lbl is not None:
        ax.text(0.98, loa_high + 5 * y_pad + label_shift, sd_lbl, ha="right", va="bottom", transform=trans, fontsize=fs_near, bbox=label_box)

    if show_annotation:
        ann = "\n".join([
            rf"$\mathrm{{mean}} = {mean_diff:.1f}\%$",
            rf"$\mathrm{{SD}} = {sd_diff:.1f}\%$",
            rf"$\mathrm{{LoA}} = [{loa_low:.1f}\%,\ {loa_high:.1f}\%]$",
        ])
        ax.text(
            0.98,
            0.92,
            ann,
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=_fs_legend(),
            bbox=ANNOT_BBOX,
        )
    fig.tight_layout()
    return _save_figure(fig, Path(save_dir) / file_name_base, formats=save_formats, dpi=dpi, create_subdir_for_stem=False)


def make_patient_level_gpr_plots(
    all_voxel_wise_dose_df: pd.DataFrame,
    semivariogram_df: pd.DataFrame,
    patient_id,
    bx_index,
    gp_res: dict,
    save_dir: Path,
    *,
    save_formats=("pdf", "svg"),
    font_scale: float = 1.0,
    seaborn_style: str = "white",
    seaborn_context: str = "paper",
    show_titles: bool = False,
    add_sill_line: bool = False,
    add_nugget_line: bool = False,
):
    """
    Generate all patient-level publication plots and save into concept-specific
    subfolders under save_dir/patient_id.
    """
    save_dir = Path(save_dir)
    concept_dirs = {
        "gp_profile": save_dir / "gp_profile",
        "uncertainty_reduction": save_dir / "uncertainty_reduction",
        "uncertainty_ratio": save_dir / "uncertainty_ratio",
        "residuals": save_dir / "residuals",
        "variogram_overlay": save_dir / "variogram_overlay",
    }
    for d in concept_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    mean_sd_mc, mean_sd_gp, shrink = _compute_shrinkage_stats(gp_res)
    overlay_metrics = rf"$\hat{{\ell}}_b = {gp_res['hyperparams'].ell:.1f}~\mathrm{{mm}},\ \Delta_b^{{(\mathrm{{SD}})}} = {shrink:.1f}\%$"

    plot_gp_profile_production(
        gp_res, patient_id, bx_index,
        save_dir=concept_dirs["gp_profile"],
        file_name_base=f"gp_profile_patient_{patient_id}_bx_{bx_index}",
        save_formats=save_formats,
        title_on=show_titles,
        font_scale=font_scale,
        seaborn_style=seaborn_style,
        seaborn_context=seaborn_context,
    )
    plot_uncertainty_reduction_production(
        gp_res, patient_id, bx_index,
        save_dir=concept_dirs["uncertainty_reduction"],
        file_name_base=f"uncertainty_reduction_patient_{patient_id}_bx_{bx_index}",
        save_formats=save_formats,
        title_on=show_titles,
        font_scale=font_scale,
        seaborn_style=seaborn_style,
        seaborn_context=seaborn_context,
    )
    plot_uncertainty_ratio_production(
        gp_res, patient_id, bx_index,
        save_dir=concept_dirs["uncertainty_ratio"],
        file_name_base=f"uncertainty_ratio_patient_{patient_id}_bx_{bx_index}",
        save_formats=save_formats,
        title_on=show_titles,
        font_scale=font_scale,
        seaborn_style=seaborn_style,
        seaborn_context=seaborn_context,
    )
    plot_residuals_vs_z_production(
        gp_res, patient_id, bx_index,
        save_dir=concept_dirs["residuals"],
        file_name_base=f"residuals_vs_z_patient_{patient_id}_bx_{bx_index}",
        save_formats=save_formats,
        font_scale=font_scale,
        seaborn_style=seaborn_style,
        seaborn_context=seaborn_context,
    )
    plot_residuals_vs_z_production(
        gp_res, patient_id, bx_index,
        save_dir=concept_dirs["residuals"],
        file_name_base=f"residuals_std_vs_z_patient_{patient_id}_bx_{bx_index}",
        save_formats=save_formats,
        standardized=True,
        font_scale=font_scale,
        seaborn_style=seaborn_style,
        seaborn_context=seaborn_context,
    )
    plot_standardized_residuals_hist_production(
        gp_res, patient_id, bx_index,
        save_dir=concept_dirs["residuals"],
        file_name_base=f"residuals_hist_patient_{patient_id}_bx_{bx_index}",
        save_formats=save_formats,
        font_scale=font_scale,
        seaborn_style=seaborn_style,
        seaborn_context=seaborn_context,
    )
    plot_standardized_residuals_qq_production(
        gp_res, patient_id, bx_index,
        save_dir=concept_dirs["residuals"],
        file_name_base=f"residuals_qq_patient_{patient_id}_bx_{bx_index}",
        save_formats=save_formats,
        font_scale=font_scale,
        seaborn_style=seaborn_style,
        seaborn_context=seaborn_context,
    )
    plot_standardized_residuals_ecdf_production(
        gp_res, patient_id, bx_index,
        save_dir=concept_dirs["residuals"],
        file_name_base=f"residuals_ecdf_patient_{patient_id}_bx_{bx_index}",
        save_formats=save_formats,
        font_scale=font_scale,
        seaborn_style=seaborn_style,
        seaborn_context=seaborn_context,
    )
    plot_variogram_overlay_production(
        semivariogram_df, patient_id, bx_index, gp_res["hyperparams"],
        save_dir=concept_dirs["variogram_overlay"],
        file_name_base=f"variogram_overlay_patient_{patient_id}_bx_{bx_index}",
        save_formats=save_formats,
        title_on=show_titles,
        font_scale=font_scale,
        seaborn_style=seaborn_style,
        seaborn_context=seaborn_context,
        add_sill=add_sill_line,
        add_nugget=add_nugget_line,
        metrics_str=overlay_metrics,
    )


_wrap_all_plot_functions()
