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
import GPR_analysis_pipeline_functions as gpr_pf

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
KERNEL_PALETTE = [
    "#0b3b8a",
    "#c75000",
    "#2a9d8f",
    "#7a5195",
    "#dd5182",
]

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


# ----------------------------------------------------------------------
# Kernel sensitivity plots
# ----------------------------------------------------------------------
def _apply_axis_style(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", labelsize=11, length=4, width=0.9)
    ax.tick_params(axis="both", which="minor", length=2, width=0.6)
    ax.minorticks_on()


def plot_kernel_sensitivity_boxplot(
    metrics_df,
    value_col: str,
    y_label: str,
    save_dir,
    file_name_base: str = "kernel_sensitivity_boxplot",
    file_types=("pdf", "svg"),
    show_title: bool = False,
    figsize=(6.0, 4.0),
    label_fontsize: int = 13,
    tick_fontsize: int = 11,
    title_fontsize: int = 14,
):
    """
    Boxplot of a metric (e.g., ell, mean_ratio) grouped by kernel_label.
    Expects metrics_df to include 'kernel_label' and value_col.
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

    kernels = sorted(metrics_df["kernel_label"].dropna().unique())
    data = [metrics_df.loc[metrics_df["kernel_label"] == k, value_col].dropna() for k in kernels]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        data,
        labels=kernels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=1.0, facecolor="white"),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="o", markerfacecolor="#888888", markersize=3, alpha=0.6),
    )
    # Color boxes
    for patch, color in zip(bp["boxes"], KERNEL_PALETTE * ((len(kernels) // len(KERNEL_PALETTE)) + 1)):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)

    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_xlabel("Kernel", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=10)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    if show_title:
        ax.set_title(f"{value_col} by kernel", fontsize=title_fontsize)

    _apply_axis_style(ax)
    fig.tight_layout()

    for ext in file_types:
        out_path = save_dir / f"{file_name_base}.{ext}"
        fig.savefig(out_path, dpi=400, bbox_inches="tight")
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
    label_fontsize: int = 13,
    tick_fontsize: int = 11,
    title_fontsize: int = 14,
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

    kernels = sorted(metrics_df["kernel_label"].dropna().unique())
    color_cycle = KERNEL_PALETTE * ((len(kernels) // len(KERNEL_PALETTE)) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    for k, color in zip(kernels, color_cycle):
        subset = metrics_df.loc[metrics_df["kernel_label"] == k]
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

    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    if show_title:
        ax.set_title(f"{y_col} vs {x_col} by kernel", fontsize=title_fontsize)

    _apply_axis_style(ax)
    ax.legend(frameon=False, fontsize=10, handlelength=1.8)
    fig.tight_layout()

    for ext in file_types:
        out_path = save_dir / f"{file_name_base}.{ext}"
        fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Production versions of GP pipeline plots
# ----------------------------------------------------------------------
def _save_multi(fig, save_dir: Path, file_name_base: str, file_types):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for ext in file_types:
        out_path = save_dir / f"{file_name_base}.{ext}"
        fig.savefig(out_path, dpi=400, bbox_inches="tight")
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def plot_gp_profile_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    file_types=("pdf", "svg"),
    ci_level="both",
    show_title=False,
    figsize=(7.0, 4.2),
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
):
    X_star = gp_res["X_star"]
    mu_star = gp_res["mu_star"]
    sd_star = gp_res["sd_star"]
    X = gp_res["X"]
    y = gp_res["y"]
    indep_sd = np.sqrt(np.maximum(gp_res["var_n"], 0))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(X_star, mu_star, lw=2.4, color=PRIMARY_LINE_COLOR, label="GP posterior mean")

    if ci_level == "both":
        ax.fill_between(X_star, mu_star - 1.96 * sd_star, mu_star + 1.96 * sd_star, alpha=0.12, color=PRIMARY_LINE_COLOR, label="95% band")
        ax.fill_between(X_star, mu_star - 1.0 * sd_star, mu_star + 1.0 * sd_star, alpha=0.22, color=PRIMARY_LINE_COLOR, label="68% band")
        ax.errorbar(X, y, yerr=2 * indep_sd, fmt="s", ms=3.5, lw=1.0, color="#1b8a5a", label="Voxel target ±2σ (MC)")
        ax.errorbar(X, y, yerr=indep_sd, fmt="o", ms=3.5, lw=1.0, color="#c75000", label="Voxel target ±1σ (MC)")
    elif ci_level in (0.68, 1):
        ax.fill_between(X_star, mu_star - 1.0 * sd_star, mu_star + 1.0 * sd_star, alpha=0.2, color=PRIMARY_LINE_COLOR, label="68% band")
        ax.errorbar(X, y, yerr=indep_sd, fmt="o", ms=3.5, lw=1.0, color="#c75000", label="Voxel target ±1σ (MC)")
    elif ci_level in (0.95, 2):
        ax.fill_between(X_star, mu_star - 1.96 * sd_star, mu_star + 1.96 * sd_star, alpha=0.15, color=PRIMARY_LINE_COLOR, label="95% band")
        ax.errorbar(X, y, yerr=2 * indep_sd, fmt="s", ms=3.5, lw=1.0, color="#1b8a5a", label="Voxel target ±2σ (MC)")
    else:
        raise ValueError(f"Unsupported ci_level={ci_level}")

    ax.set_xlabel("Distance along core (mm)", fontsize=label_fontsize)
    ax.set_ylabel("Dose (Gy)", fontsize=label_fontsize)
    if show_title:
        ax.set_title(f"GP profile — Patient {patient_id}, Bx {bx_index}", fontsize=legend_fontsize + 2)
    _apply_axis_style(ax)
    ax.legend(frameon=False, fontsize=legend_fontsize)
    fig.tight_layout()
    return _save_multi(fig, save_dir, file_name_base, file_types)


def plot_noise_profile_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    file_types=("pdf", "svg"),
    figsize=(6.5, 3.8),
    label_fontsize=13,
    tick_fontsize=11,
):
    per_voxel = gp_res.get("per_voxel")
    if per_voxel is None:
        raise ValueError("gp_res must contain 'per_voxel' with x_mm and var_n columns.")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(per_voxel["x_mm"], np.sqrt(np.maximum(per_voxel["var_n"], 0)), marker="o", ms=4, lw=1.2, color=PRIMARY_LINE_COLOR)
    ax.set_xlabel("Distance along core (mm)", fontsize=label_fontsize)
    ax.set_ylabel("Independent SD (Gy)", fontsize=label_fontsize)
    _apply_axis_style(ax)
    fig.tight_layout()
    return _save_multi(fig, save_dir, file_name_base, file_types)


def plot_uncertainty_reduction_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    file_types=("pdf", "svg"),
    figsize=(6.8, 3.8),
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
):
    X = gp_res["X"]
    indep_sd = np.sqrt(np.maximum(gp_res["var_n"], 0))
    sd_X = gp_res["sd_X"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(X, indep_sd, "o-", ms=4, lw=1.2, label="Independent SD", color=OVERLAY_LINE_COLOR)
    ax.plot(X, sd_X, "o-", ms=4, lw=1.2, label="GP posterior SD", color=PRIMARY_LINE_COLOR)
    ax.set_xlabel("Distance along core (mm)", fontsize=label_fontsize)
    ax.set_ylabel("SD (Gy)", fontsize=label_fontsize)
    _apply_axis_style(ax)
    ax.legend(frameon=False, fontsize=legend_fontsize)
    fig.tight_layout()
    return _save_multi(fig, save_dir, file_name_base, file_types)


def plot_uncertainty_ratio_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    file_types=("pdf", "svg"),
    figsize=(6.5, 3.5),
    label_fontsize=13,
    tick_fontsize=11,
):
    X = gp_res["X"]
    indep_sd = np.sqrt(np.maximum(gp_res["var_n"], 0))
    ratio = np.divide(indep_sd, gp_res["sd_X"], out=np.ones_like(indep_sd), where=gp_res["sd_X"] > 0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(X, ratio, "-o", ms=4, lw=1.2, color=PRIMARY_LINE_COLOR)
    ax.axhline(1.0, color="black", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlabel("Distance along core (mm)", fontsize=label_fontsize)
    ax.set_ylabel("Ratio (indep / GP SD)", fontsize=label_fontsize)
    _apply_axis_style(ax)
    fig.tight_layout()
    return _save_multi(fig, save_dir, file_name_base, file_types)


def plot_residuals_production(
    gp_res: dict,
    patient_id,
    bx_index,
    *,
    save_dir: Path,
    file_name_base: str,
    file_types=("pdf", "svg"),
    figsize=(9.0, 3.8),
    label_fontsize=13,
    tick_fontsize=11,
):
    X = gp_res["X"]
    y = gp_res["y"]
    mu_X = gp_res["mu_X"]
    sd_X = gp_res["sd_X"]
    res = y - mu_X
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].axhline(0, color="black", lw=1.0, alpha=0.6)
    axes[0].plot(X, res, "o-", ms=4, lw=1.1, color=PRIMARY_LINE_COLOR)
    axes[0].set_xlabel("Distance (mm)", fontsize=label_fontsize)
    axes[0].set_ylabel("Residual (Gy)", fontsize=label_fontsize)
    _apply_axis_style(axes[0])

    axes[1].hist(res / np.maximum(sd_X, 1e-12), bins=20, density=True, alpha=0.75, color=PRIMARY_LINE_COLOR)
    axes[1].set_xlabel("Standardized residual", fontsize=label_fontsize)
    axes[1].set_ylabel("Density", fontsize=label_fontsize)
    _apply_axis_style(axes[1])
    fig.suptitle(f"Diagnostics — Patient {patient_id}, Bx {bx_index}", fontsize=label_fontsize + 1)
    fig.tight_layout()
    return _save_multi(fig, save_dir, file_name_base, file_types)


def plot_variogram_overlay_production(
    semivariogram_df: pd.DataFrame,
    patient_id,
    bx_index,
    hyperparams,
    *,
    save_dir: Path,
    file_name_base: str,
    file_types=("pdf", "svg"),
    figsize=(6.4, 3.6),
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
):
    sv = semivariogram_df.query("`Patient ID`==@patient_id and `Bx index`==@bx_index").sort_values("h_mm")
    h = sv["h_mm"].to_numpy(float)
    gamma_hat = sv["semivariance"].to_numpy(float)
    kernel = getattr(hyperparams, "kernel", "matern")
    if kernel == "rbf":
        gamma_model = gpr_pf.rbf_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0) + hyperparams.nugget
        label_model = "RBF implied γ(h)"
    elif kernel == "exp":
        gamma_model = gpr_pf.exp_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0) + hyperparams.nugget
        label_model = "Exponential implied γ(h)"
    else:
        label_model = f"Matérn implied γ(h), ν={hyperparams.nu}"
        gamma_model = gpr_pf.matern_semivariogram(h, hyperparams.sigma_f2, hyperparams.ell, 0.0, hyperparams.nu) + hyperparams.nugget

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(h, gamma_hat, "o", ms=4, color=PRIMARY_LINE_COLOR, label="Empirical γ̂(h)")
    ax.plot(h, gamma_model, "-", lw=2.0, color=OVERLAY_LINE_COLOR, label=label_model)
    ax.set_xlabel("Lag h (mm)", fontsize=label_fontsize)
    ax.set_ylabel("Semivariance γ(h) (Gy²)", fontsize=label_fontsize)
    _apply_axis_style(ax)
    ax.legend(frameon=False, fontsize=legend_fontsize)
    fig.tight_layout()
    return _save_multi(fig, save_dir, file_name_base, file_types)


def cohort_plots_production(
    metrics_df: pd.DataFrame,
    save_dir: Path,
    file_types=("pdf", "svg"),
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _hist(series, title, xlabel, fname):
        fig, ax = plt.subplots(figsize=(6, 4))
        data = series.dropna()
        ax.hist(data, bins=20, alpha=0.8, color=PRIMARY_LINE_COLOR)
        if title:
            ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Count", fontsize=13)
        _apply_axis_style(ax)
        fig.tight_layout()
        _save_multi(fig, save_dir, fname, file_types)

    _hist(metrics_df["mean_ratio"], "Mean uncertainty ratio per biopsy", "Mean( SD_indep / SD_GP )", "cohort_uncertainty_ratio_hist")
    _hist(metrics_df["ell"], "GP length scale (ℓ) across biopsies", "Length scale ℓ (mm)", "cohort_length_scale_hist")
    _hist(metrics_df["nugget"], "GP nugget across biopsies", "Nugget (Gy²)", "cohort_nugget_hist")

    # Box summaries for ratios
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    data = [
        metrics_df["mean_ratio"].dropna(),
        metrics_df["median_ratio"].dropna(),
        metrics_df["integ_ratio"].dropna(),
    ]
    labels = ["Mean ratio", "Median ratio", "Integrated ratio"]
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.1),
                    boxprops=dict(linewidth=1.0, facecolor="white"),
                    whiskerprops=dict(color="black", linewidth=1.0),
                    capprops=dict(color="black", linewidth=1.0))
    for patch, color in zip(bp["boxes"], KERNEL_PALETTE * 2):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    ax.set_ylabel("Ratio (indep / GP)", fontsize=13)
    _apply_axis_style(ax)
    fig.tight_layout()
    _save_multi(fig, save_dir, "cohort_uncertainty_reduction_box", file_types)

    # Scatter mean SDs
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(metrics_df["mean_indep_sd"], metrics_df["mean_gp_sd"], s=24, alpha=0.85, color=PRIMARY_LINE_COLOR)
    lim_hi = float(np.nanmax([metrics_df["mean_indep_sd"].max(), metrics_df["mean_gp_sd"].max(), 0]))
    lims = [0, lim_hi * 1.05 if lim_hi > 0 else 1.0]
    ax.plot(lims, lims, "k--", lw=1.0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Independent mean SD (Gy)", fontsize=13)
    ax.set_ylabel("GP mean SD (Gy)", fontsize=13)
    _apply_axis_style(ax)
    fig.tight_layout()
    _save_multi(fig, save_dir, "cohort_mean_sd_scatter", file_types)


def plot_mean_sd_scatter_with_fits_production(
    metrics_df: pd.DataFrame,
    reg_stats: pd.DataFrame,
    save_dir: Path,
    file_name_base: str = "cohort_mean_sd_scatter_with_fits",
    file_types=("pdf", "svg"),
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

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(x, y, s=24, alpha=0.85, color=PRIMARY_LINE_COLOR, label="Biopsies")
    ax.plot(lims, lims, "k--", lw=1.0, label="Identity")
    have_ols = np.isfinite(s.get("ols_slope", np.nan)) and np.isfinite(s.get("ols_intercept", np.nan))
    if have_ols:
        a, b = float(s["ols_intercept"]), float(s["ols_slope"])
        R2 = s.get("ols_R2", np.nan)
        ax.plot(xs, a + b * xs, lw=2.0, color="#c75000", label=f"OLS: y={a:.3f}+{b:.3f}x (R²={R2:.3f})")
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

    if np.isfinite(s.get("deming_slope", np.nan)) and np.isfinite(s.get("deming_intercept", np.nan)):
        a_d, b_d = float(s["deming_intercept"]), float(s["deming_slope"])
        ax.plot(xs, a_d + b_d * xs, lw=2.0, ls=":", color="#2a9d8f", label=f"Deming: y={a_d:.3f}+{b_d:.3f}x")

    if add_origin_fit and np.isfinite(s.get("origin_slope", np.nan)):
        bo = float(s["origin_slope"])
        ax.plot(xs, bo * xs, lw=1.5, ls="-.", color="#7a5195", label=f"Through-origin: y={bo:.3f}x")

    ax.set_xlabel("Independent mean SD (Gy)", fontsize=13)
    ax.set_ylabel("GP mean SD (Gy)", fontsize=13)
    ax.set_xlim(lims); ax.set_ylim(lims)
    _apply_axis_style(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    _save_multi(fig, Path(save_dir), file_name_base, file_types)
