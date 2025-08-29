import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPR_analysis_pipeline_functions
from scipy.stats import gaussian_kde
from pathlib import Path




plt.ioff()  # keep figures from popping up; you already close after saving

# --------- Utilities ----------

def _ensure_ext(path, default_ext=".svg"):
    root, ext = os.path.splitext(path)
    return path if ext else (root + default_ext)

def _resolve_save_path(save_path, file_name, default_fname):
    if save_path is None:
        return None
    sp = str(save_path)
    looks_like_dir = (os.path.isdir(sp) or sp.endswith(os.sep) or os.path.splitext(sp)[1] == "")
    if looks_like_dir:
        os.makedirs(sp, exist_ok=True)
        fname = file_name if file_name else default_fname
        full = os.path.join(sp, fname)
        return _ensure_ext(full)
    else:
        parent = os.path.dirname(sp)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return _ensure_ext(sp)

def _finalize_save(fig, resolved_path):
    if resolved_path is None:
        return None
    fig.savefig(resolved_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return resolved_path

def _style_axes(ax, title, xlabel, ylabel,
                title_size=14, label_size=12, tick_size=10,
                legend_size=10, grid=True):
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    if grid:
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.36, color="black")
        ax.grid(True, which="minor", alpha=0.18, color="gray")
    # resize legend if present
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontsize(legend_size)

def _finite(arr):
    arr = np.asarray(arr)
    return arr[np.isfinite(arr)]

# --------- GP helper ----------

def predict_at_X(all_voxel_wise_dose_df=None,
                 semivariogram_df=None,
                 patient_id=None,
                 bx_index=None,
                 target_stat="median",
                 nu=1.5,
                 res=None):
    """
    Run GP for one biopsy and return posterior mean/SD evaluated at voxel centers.
    If `res` (output of run_gp_for_biopsy) is provided, reuse it instead of recomputing.
    """
    if res is None:
        res = GPR_analysis_pipeline_functions.run_gp_for_biopsy(
            all_voxel_wise_dose_df, semivariogram_df,
            patient_id=patient_id, bx_index=bx_index,
            target_stat=target_stat, nu=nu
        )
    hyp   = res["hyperparams"]
    X     = res["X"]; y = res["y"]; var_n = res["var_n"]
    mu_X, sd_X = GPR_analysis_pipeline_functions.gp_posterior(X, y, var_n, hyp, X_star=X)
    res = dict(res)
    res.update(dict(mu_X=mu_X, sd_X=sd_X))
    return res

# -------------------------
# 1) GP posterior profile
# -------------------------
def plot_gp_profile(all_voxel_wise_dose_df, semivariogram_df,
                    patient_id, bx_index,
                    target_stat="median", nu=1.5,
                    ax=None,
                    save_path=None, file_name=None, return_path=False, gp_res=None,
                    title_size=14, label_size=12, tick_size=10, legend_size=10,
                    grid=True, figsize=(8,4.5), ci_level="both"):
    """
    Plot GP regression profile along core with uncertainty bands.

    Parameters
    ----------
    ci_level : {"both", 0.68, 0.95, 1, 2}, optional
        - "both": plot both 68% (±1σ) and 95% (±1.96σ)
        - 0.68 or 1: only 68% band (±1σ)
        - 0.95 or 2: only 95% band (±1.96σ)
    """
    plt.ioff()   # disable interactive plotting

    out = predict_at_X(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index,
                       target_stat, nu, res=gp_res)
    X_star, mu_star, sd_star = out["X_star"], out["mu_star"], out["sd_star"]
    X, y, var_n = out["X"], out["y"], out["var_n"]
    indep_sd = np.sqrt(var_n)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    # GP posterior mean
    ax.plot(X_star, mu_star, lw=2, label="GP posterior mean", color="C0")

    if ci_level == "both":
        # 95% band
        ax.fill_between(X_star, mu_star - 1.96*sd_star, mu_star + 1.96*sd_star,
                        alpha=0.15, color="C0", label="95% band (±1.96σ)")
        ax.errorbar(X, y, yerr=2*indep_sd, fmt='s', ms=3, lw=1, alpha=1.0,
                    label="Voxel target ±2σ (MC)", color="C2")
        
        # 68% band
        ax.fill_between(X_star, mu_star - 1*sd_star, mu_star + 1*sd_star,
                        alpha=0.3, color="C0", label="68% band (±1σ)")
        ax.errorbar(X, y, yerr=indep_sd, fmt='o', ms=3, lw=1, alpha=1.0,
                    label="Voxel target ±1σ (MC)", color="C1")

    elif ci_level in (0.68, 1):
        ax.fill_between(X_star, mu_star - 1*sd_star, mu_star + 1*sd_star,
                        alpha=0.25, color="C0", label="68% band (±1σ)")
        ax.errorbar(X, y, yerr=indep_sd, fmt='o', ms=3, lw=1, alpha=0.8,
                    label="Voxel target ±1σ (MC)", color="C1")

    elif ci_level in (0.95, 2):
        ax.fill_between(X_star, mu_star - 1.96*sd_star, mu_star + 1.96*sd_star,
                        alpha=0.2, color="C0", label="95% band (±1.96σ)")
        ax.errorbar(X, y, yerr=2*indep_sd, fmt='o', ms=3, lw=1, alpha=1.0,
                    label="Voxel target ±2σ (MC)", color="C1")

    else:
        raise ValueError(f"Unsupported ci_level={ci_level}")

    # Style
    _style_axes(ax,
        title=f"GP dose profile — Patient {patient_id}, Bx {bx_index}",
        xlabel="Distance along core (mm)",
        ylabel="Dose (Gy)",
        title_size=title_size, label_size=label_size, tick_size=tick_size,
        legend_size=legend_size, grid=grid
    )
    ax.legend(fontsize=legend_size)

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"gp_profile_patient_{patient_id}_bx_{bx_index}.svg")
    if created_fig and resolved is not None:
        saved = _finalize_save(fig, resolved)
        return saved if return_path else None
    return ax if not return_path else resolved


# -----------------------------------------------
# 2) Per-voxel MC uncertainty (noise profile)
# -----------------------------------------------
def plot_noise_profile(all_voxel_wise_dose_df, patient_id, bx_index,
                       target_stat="median",
                       ax=None,
                       save_path=None, file_name=None, return_path=False,
                       title_size=14, label_size=12, tick_size=10, legend_size=10,
                       grid=True, figsize=(8,4)):
    
    plt.ioff()   # disable interactive plotting


    df_bx = all_voxel_wise_dose_df.query("`Patient ID`==@patient_id and `Bx index`==@bx_index")
    per_voxel = (
        df_bx.groupby("Voxel index")
             .agg(x_mm=("Z (Bx frame)", "mean"),
                  y=("Dose (Gy)", "median" if target_stat=="median" else "mean"),
                  var_n=("Dose (Gy)", "var"))
             .sort_values("x_mm").reset_index()
    )
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(per_voxel["x_mm"], np.sqrt(per_voxel["var_n"].clip(lower=0)), marker='o', lw=1)
    _style_axes(ax,
        title=f"Per-voxel MC uncertainty — Patient {patient_id}, Bx {bx_index}",
        xlabel="Distance along core (mm)", ylabel="Independent SD (Gy)",
        title_size=title_size, label_size=label_size, tick_size=tick_size,
        legend_size=legend_size, grid=grid
    )

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"noise_profile_patient_{patient_id}_bx_{bx_index}.svg")
    if created_fig and resolved is not None:
        saved = _finalize_save(fig, resolved)
        return saved if return_path else None
    return ax if not return_path else resolved

# ----------------------------------------------------------
# 3) Uncertainty reduction (SD comparison, no inset)
# ----------------------------------------------------------
def plot_uncertainty_reduction(all_voxel_wise_dose_df, semivariogram_df,
                               patient_id, bx_index,
                               target_stat="median", nu=1.5,
                               ax=None,
                               save_path=None, file_name=None, return_path=False, gp_res=None,
                               title_size=14, label_size=12, tick_size=10, legend_size=10,
                               grid=True, figsize=(8,4)):
    plt.ioff()   # disable interactive plotting


    out = predict_at_X(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index,
                       target_stat, nu, res=gp_res)
    X, var_n, sd_X = out["X"], out["var_n"], out["sd_X"]
    indep_sd = np.sqrt(var_n)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(X, indep_sd, 'o-', ms=3, lw=1, label="Independent voxel SD")
    ax.plot(X, sd_X, 'o-', ms=3, lw=1, label="GP posterior SD")

    _style_axes(ax,
        title=f"Uncertainty reduction — Patient {patient_id}, Bx {bx_index}",
        xlabel="Distance along core (mm)", ylabel="SD (Gy)",
        title_size=title_size, label_size=label_size, tick_size=tick_size,
        legend_size=legend_size, grid=grid
    )
    ax.legend(fontsize=legend_size)

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"uncertainty_reduction_patient_{patient_id}_bx_{bx_index}.svg")
    if created_fig and resolved is not None:
        saved = _finalize_save(fig, resolved)
        return saved if return_path else None
    return ax if not return_path else resolved

# ----------------------------------------------------------
# 3b) New: Uncertainty ratio (Indep / GP SD) as its own fig
# ----------------------------------------------------------
def plot_uncertainty_ratio(all_voxel_wise_dose_df, semivariogram_df,
                           patient_id, bx_index,
                           target_stat="median", nu=1.5,
                           ax=None,
                           save_path=None, file_name=None, return_path=False, gp_res=None,
                           title_size=14, label_size=12, tick_size=10,
                           grid=True, figsize=(7,3.5)):
    plt.ioff()   # disable interactive plotting


    out = predict_at_X(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index,
                       target_stat, nu, res=gp_res)
    X, var_n, sd_X = out["X"], out["var_n"], out["sd_X"]
    indep_sd = np.sqrt(var_n)
    ratio = np.divide(indep_sd, sd_X, out=np.ones_like(indep_sd), where=sd_X > 0)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(X, ratio, '-o', lw=1)
    _style_axes(ax,
        title=f"Uncertainty ratio (Indep / GP) — Patient {patient_id}, Bx {bx_index}",
        xlabel="Distance along core (mm)", ylabel="Ratio",
        title_size=title_size, label_size=label_size, tick_size=tick_size,
        legend_size=10, grid=grid
    )

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"uncertainty_ratio_patient_{patient_id}_bx_{bx_index}.svg")
    if created_fig and resolved is not None:
        saved = _finalize_save(fig, resolved)
        return saved if return_path else None
    return ax if not return_path else resolved

# ---------------------------------------------
# 4) Residual diagnostics (two-panel figure)
# ---------------------------------------------
def plot_residuals(all_voxel_wise_dose_df, semivariogram_df,
                   patient_id, bx_index,
                   target_stat="median", nu=1.5,
                   save_path=None, file_name=None, return_path=False, gp_res=None,
                   title_size=14, label_size=12, tick_size=10,
                   grid=True, figsize=(10,3.8)):
    plt.ioff()   # disable interactive plotting


    out = predict_at_X(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index,
                       target_stat, nu, res=gp_res)
    X, y, mu_X, sd_X = out["X"], out["y"], out["mu_X"], out["sd_X"]
    res = y - mu_X

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Left: residuals vs distance
    axes[0].axhline(0, color='k', lw=1, alpha=0.5)
    axes[0].plot(X, res, 'o-', ms=3, lw=1)
    _style_axes(axes[0],
        title="Residuals vs distance", xlabel="Distance (mm)", ylabel="Residual (Gy)",
        title_size=title_size, label_size=label_size, tick_size=tick_size,
        legend_size=10, grid=grid
    )
    # Right: standardized residuals histogram
    axes[1].hist(res / np.maximum(sd_X, 1e-12), bins=20, density=True, alpha=0.75)
    _style_axes(axes[1],
        title="Standardized residuals", xlabel="r / sd", ylabel="Density",
        title_size=title_size, label_size=label_size, tick_size=tick_size,
        legend_size=10, grid=grid
    )
    fig.suptitle(f"Diagnostics — Patient {patient_id}, Bx {bx_index}", fontsize=title_size)

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"residuals_patient_{patient_id}_bx_{bx_index}.svg")
    saved = _finalize_save(fig, resolved) if resolved is not None else None
    return (saved if return_path else axes)

# ------------------------------------------------------------
# 5) Variogram overlay (empirical vs kernel-implied Matérn)
# ------------------------------------------------------------
def plot_variogram_overlay(semivariogram_df, patient_id, bx_index, hyperparams,
                           ax=None, nu=None,
                           save_path=None, file_name=None, return_path=False,
                           title_size=14, label_size=12, tick_size=10, legend_size=10,
                           grid=True, figsize=(7,3.8)):
    plt.ioff()   # disable interactive plotting


    sv = semivariogram_df.query("`Patient ID`==@patient_id and `Bx index`==@bx_index").sort_values("h_mm")
    h = sv["h_mm"].to_numpy(float)
    gamma_hat = sv["semivariance"].to_numpy(float)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(h, gamma_hat, 'o', ms=4, label="Empirical γ̂(h)")
    nu_use = hyperparams.nu if nu is None else nu
    gamma_model = (GPR_analysis_pipeline_functions.matern_semivariogram(
                        h, hyperparams.sigma_f2, hyperparams.ell, 0.0, nu_use)
                   + hyperparams.nugget)
    ax.plot(h, gamma_model, '-', lw=2, label=f"Implied Matérn γ(h), ν={nu_use}")

    _style_axes(ax,
        title=f"Variogram overlay — Patient {patient_id}, Bx {bx_index}",
        xlabel="Lag h (mm)", ylabel="Semivariance γ(h) (Gy²)",
        title_size=title_size, label_size=label_size, tick_size=tick_size,
        legend_size=legend_size, grid=grid
    )
    ax.legend(fontsize=legend_size)

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"variogram_overlay_patient_{patient_id}_bx_{bx_index}.svg")
    if created_fig and resolved is not None:
        saved = _finalize_save(fig, resolved)
        return saved if return_path else None
    return ax if not return_path else resolved






def cohort_plots(metrics_df,
                 output_fig_directory):

    plt.ioff()  # keep figures from popping up; you already close after saving

    # 4a) Distribution of mean uncertainty ratio (indep SD / GP SD) across biopsies
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(_finite(metrics_df["mean_ratio"]), bins=20, density=False, alpha=0.8)
    _style_axes(ax, title="Cohort: mean uncertainty ratio per biopsy",
                xlabel="Mean( SD_indep / SD_GP )", ylabel="Count")
    save_path = output_fig_directory.joinpath("cohort_uncertainty_ratio_hist.svg")
    fig.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close(fig)

    # 4b) Violin/box of ratios (per-biopsy summaries)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot([_finite(metrics_df["mean_ratio"]),
                _finite(metrics_df["median_ratio"]),
                _finite(metrics_df["integ_ratio"])],
            labels=["Mean ratio", "Median ratio", "Integrated ratio"])
    _style_axes(ax, title="Cohort: uncertainty reduction summaries",
                xlabel="", ylabel="Ratio (indep / GP)")
    save_path = output_fig_directory.joinpath("cohort_uncertainty_reduction_box.svg")
    fig.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close(fig)

    # 4c) Hyperparameter distributions — length scale and nugget
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(_finite(metrics_df["ell"]), bins=20, alpha=0.85)
    _style_axes(ax, title="GP length scale (ℓ) across biopsies",
                xlabel="Length scale ℓ (mm)", ylabel="Count")
    fig.savefig(output_fig_directory.joinpath("cohort_length_scale_hist.svg"), dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(_finite(metrics_df["nugget"]), bins=20, alpha=0.85)
    _style_axes(ax, title="GP nugget across biopsies",
                xlabel="Nugget (Gy²)", ylabel="Count")
    fig.savefig(output_fig_directory.joinpath("cohort_nugget_hist.svg"), dpi=300, bbox_inches="tight"); plt.close(fig)

    # 4d) Residual standardization check: std should be ~1 if calibrated
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(_finite(metrics_df["std_resstd"]), bins=20, alpha=0.85)
    _style_axes(ax, title="Std of standardized residuals (per biopsy)",
                xlabel="Std(r / sd_GP)", ylabel="Count")
    fig.savefig(output_fig_directory.joinpath("cohort_resstd_std_hist.svg"), dpi=300, bbox_inches="tight"); plt.close(fig)



    # Histogram + KDE of % reduction (mean SD)
    data = metrics_df["pct_reduction_mean_sd"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(data, bins=20, density=True, alpha=0.7)
    # optional KDE overlay
    if data.size >= 5:
        kde = gaussian_kde(data)
        xs = np.linspace(np.percentile(data, 0.5), np.percentile(data, 99.5), 200)
        ax.plot(xs, kde(xs), lw=2)
    _style_axes(ax, "Cohort: % reduction in mean voxel SD", "% reduction (%)", "Density")
    fig.savefig(output_fig_directory.joinpath("cohort_pct_reduction_mean_sd_hist.svg"), dpi=300, bbox_inches="tight"); plt.close(fig)

    # ECDF (very clean)
    data = np.sort(metrics_df["pct_reduction_mean_sd"].dropna().to_numpy())
    y = np.arange(1, data.size+1) / data.size
    fig, ax = plt.subplots(figsize=(6,4))
    ax.step(data, y, where="post")
    _style_axes(ax, "ECDF: % reduction in mean voxel SD", "% reduction (%)", "Proportion ≤ x")
    fig.savefig(output_fig_directory.joinpath("cohort_pct_reduction_mean_sd_ecdf.svg"), dpi=300, bbox_inches="tight"); plt.close(fig)



    fig, ax = plt.subplots(figsize=(5.2,5))
    ax.scatter(metrics_df["mean_indep_sd"], metrics_df["mean_gp_sd"], s=20, alpha=0.8)
    lims = [0, np.nanmax([metrics_df["mean_indep_sd"].max(), metrics_df["mean_gp_sd"].max()])]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    _style_axes(ax, "Per-biopsy mean SD: independent vs GP", "Independent mean SD (Gy)", "GP mean SD (Gy)")
    fig.savefig(output_fig_directory.joinpath("cohort_mean_sd_scatter.svg"), dpi=300, bbox_inches="tight"); plt.close(fig)


    x = 0.5*(metrics_df["mean_indep_sd"] + metrics_df["mean_gp_sd"])
    d = metrics_df["mean_indep_sd"] - metrics_df["mean_gp_sd"]
    m, s = np.nanmean(d), np.nanstd(d, ddof=1)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(x, d, s=20, alpha=0.8)
    ax.axhline(m, color='k', lw=1)
    ax.axhline(m + 1.96*s, color='k', lw=1, ls='--')
    ax.axhline(m - 1.96*s, color='k', lw=1, ls='--')
    _style_axes(ax, "Bland–Altman: mean SD (indep − GP)", "Mean of SDs (Gy)", "Difference in SD (Gy)")
    fig.savefig(output_fig_directory.joinpath("cohort_bland_altman_mean_sd.svg"), dpi=300, bbox_inches="tight"); plt.close(fig)





# ---------------------------------------------------------------------
# 2) Plot function: uses the saved/returned stats (no refitting)
# ---------------------------------------------------------------------
def plot_mean_sd_scatter_with_fits(
    metrics_df: pd.DataFrame,
    reg_stats: pd.DataFrame,
    save_svg_path: Path,
    x_col: str = "mean_indep_sd",
    y_col: str = "mean_gp_sd",
    title: str = "Per-biopsy mean SD: independent vs GP",
    add_origin_fit: bool = False,
    add_ci_ribbon: bool = True,
    add_pred_band: bool = False,  # 95% prediction band
):
    x = metrics_df[x_col].to_numpy(dtype=float)
    y = metrics_df[y_col].to_numpy(dtype=float)
    msk = np.isfinite(x) & np.isfinite(y)
    x, y = x[msk], y[msk]

    s = reg_stats.iloc[0]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(x, y, s=22, alpha=0.85, label="Biopsies")

    lim_hi = float(np.nanmax([x.max() if x.size else 0, y.max() if y.size else 0]))
    lims = [0.0, lim_hi * 1.05 if lim_hi > 0 else 1.0]
    ax.plot(lims, lims, "k--", lw=1.0, label="Identity (y = x)")
    ax.set_xlim(lims); ax.set_ylim(lims)

    xs = np.linspace(lims[0], lims[1], 200)

    # ---- OLS line + CI ribbon ----
    have_ols = np.isfinite(s.get("ols_slope", np.nan)) and np.isfinite(s.get("ols_intercept", np.nan))
    if have_ols:
        a, b = float(s["ols_intercept"]), float(s["ols_slope"])
        R2 = s.get("ols_R2", np.nan)
        ax.plot(xs, a + b*xs, lw=2.0, label=f"OLS: y={a:.3f}+{b:.3f}x (R²={R2:.3f})")

        if add_ci_ribbon:
            sigma2 = s.get("ols_sigma2", np.nan)
            xbar   = s.get("ols_xbar", np.nan)
            Sxx    = s.get("ols_Sxx", np.nan)
            df     = s.get("ols_df", np.nan)
            tcrit  = s.get("ols_tcrit", np.nan)

            if all(np.isfinite(v) for v in [sigma2, xbar, Sxx, df, tcrit]) and Sxx > 0 and df >= 1:
                se_mean = np.sqrt(sigma2 * (1.0/float(s["n"]) + (xs - xbar)**2 / Sxx))
                yhat = a + b*xs
                ax.fill_between(xs, yhat - tcrit*se_mean, yhat + tcrit*se_mean,
                                alpha=0.18, label="95% CI (mean)")
                if add_pred_band:
                    se_pred = np.sqrt(sigma2 * (1.0 + 1.0/float(s["n"]) + (xs - xbar)**2 / Sxx))
                    ax.fill_between(xs, yhat - tcrit*se_pred, yhat + tcrit*se_pred,
                                    alpha=0.10, label="95% prediction")

    # ---- Deming line ----
    if np.isfinite(s.get("deming_slope", np.nan)) and np.isfinite(s.get("deming_intercept", np.nan)):
        a_d, b_d = float(s["deming_intercept"]), float(s["deming_slope"])
        ax.plot(xs, a_d + b_d*xs, lw=2.0, ls=":", label=f"Deming: y={a_d:.3f}+{b_d:.3f}x")

    # ---- Through-origin (optional) ----
    if add_origin_fit and np.isfinite(s.get("origin_slope", np.nan)):
        bo = float(s["origin_slope"])
        ax.plot(xs, bo*xs, lw=1.5, ls="-.", label=f"Through-origin: y={bo:.3f}x")

    # Style
    try:
        from GPR_analysis_plotting_functions_manual_methods import _style_axes
        _style_axes(ax, title=title, xlabel="Independent mean SD (Gy)", ylabel="GP mean SD (Gy)")
    except Exception:
        ax.set_title(title); ax.set_xlabel("Independent mean SD (Gy)"); ax.set_ylabel("GP mean SD (Gy)")
        ax.grid(True, which="major", linewidth=1.0, alpha=0.6); ax.grid(True, which="minor", linewidth=0.5, alpha=0.3); ax.minorticks_on()

    ax.legend(fontsize=10, frameon=True)

    save_svg_path = Path(save_svg_path)
    save_svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_svg_path, dpi=300, bbox_inches="tight")
    plt.close(fig)