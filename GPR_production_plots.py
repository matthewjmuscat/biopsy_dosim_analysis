"""
Production-quality plotting utilities for GPR variogram outputs.
Plots are intended for publication (e.g., Physics in Medicine & Biology),
with flexible typography, sizing, and export formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

PRIMARY_LINE_COLOR = "#0b3b8a"  # deep blue for main semivariogram
OVERLAY_LINE_COLOR = "#4a4a4a"   # neutral gray for overlays
GRID_COLOR = "#b8b8b8"

plt.ioff()


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
    x_label: str = r"$h$ (mm)",
    y_label: str = r"$\gamma_b(h)$ [Gy$^2$]",
    label_fontsize: int = 13,
    tick_labelsize: int = 11,
    title_fontsize: int = 14,
    legend_fontsize: int = 11,
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
            label=r"Semivariogram $\gamma_b(h)$",
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
            ax.set_title(", ".join(title_bits), fontsize=title_fontsize, pad=6)

        ax.set_xlabel(x_label, fontsize=label_fontsize)
        ax.set_ylabel(y_label, fontsize=label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, length=4, width=0.9)
        ax.tick_params(axis="both", which="minor", length=2, width=0.6)
        ax.minorticks_on()

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
        ax.legend(frameon=False, fontsize=legend_fontsize, handlelength=2.6, borderaxespad=0.5)
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
