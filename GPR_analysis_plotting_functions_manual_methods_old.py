import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPR_analysis_pipeline_functions

# ---- Helpers (use your earlier code objects): run_gp_for_biopsy, gp_posterior, matern_semivariogram ----
# Assumes you’ve defined in GPR_analysis_pipeline_functions:
#   run_gp_for_biopsy, gp_posterior, matern_semivariogram, GPHyperparams

# -----------------------------
# Internal save-path utilities
# -----------------------------
def _ensure_ext(path, default_ext=".png"):
    root, ext = os.path.splitext(path)
    return path if ext else (root + default_ext)

def _resolve_save_path(save_path, file_name, default_fname):
    """
    If save_path is None -> return None (no save).
    If save_path is a directory (exists or endswith os.sep or has no extension) -> join with file_name or default.
    Else treat save_path as full file path. Ensure extension.
    """
    if save_path is None:
        return None

    sp = str(save_path)

    # Heuristics to decide "directory-like"
    looks_like_dir = (os.path.isdir(sp) or sp.endswith(os.sep) or os.path.splitext(sp)[1] == "")

    if looks_like_dir:
        os.makedirs(sp, exist_ok=True)
        fname = file_name if file_name else default_fname
        full = os.path.join(sp, fname)
        return _ensure_ext(full)
    else:
        # full path to file
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

# -----------------------------
# GP prediction helper (unchanged)
# -----------------------------
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
        # Normal case: compute from scratch
        res = GPR_analysis_pipeline_functions.run_gp_for_biopsy(
            all_voxel_wise_dose_df, semivariogram_df,
            patient_id=patient_id, bx_index=bx_index,
            target_stat=target_stat, nu=nu
        )

    hyp   = res["hyperparams"]
    X     = res["X"]
    y     = res["y"]
    var_n = res["var_n"]

    # Evaluate posterior at the observed X for residuals/diagnostics
    mu_X, sd_X = GPR_analysis_pipeline_functions.gp_posterior(
        X, y, var_n, hyp, X_star=X
    )

    res = dict(res)  # copy so we don’t mutate original
    res.update(dict(mu_X=mu_X, sd_X=sd_X))
    return res


# -----------------------------------------------
# 1) GP posterior: dose vs distance with 95% band
# -----------------------------------------------
def plot_gp_profile(all_voxel_wise_dose_df, semivariogram_df,
                    patient_id, bx_index,
                    target_stat="median", nu=1.5,
                    ax=None,
                    save_path=None, file_name=None, return_path=False, gp_res=None):
    out = predict_at_X(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index, target_stat, nu, res=gp_res)
    X_star, mu_star, sd_star = out["X_star"], out["mu_star"], out["sd_star"]
    X, y, var_n = out["X"], out["y"], out["var_n"]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(X_star, mu_star, lw=2, label="GP posterior mean")
    ax.fill_between(X_star, mu_star - 1.96*sd_star, mu_star + 1.96*sd_star, alpha=0.2, label="95% band")
    ax.errorbar(X, y, yerr=np.sqrt(var_n), fmt='o', ms=3, lw=1, alpha=0.7, label="Voxel target ± SD (MC)")
    ax.set_xlabel("Distance along core (mm)")
    ax.set_ylabel("Dose (Gy)")
    ax.set_title(f"GP dose profile — Patient {patient_id}, Bx {bx_index}")
    ax.legend()

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"gp_profile_patient_{patient_id}_bx_{bx_index}.png")
    if created_fig and resolved is not None:
        saved = _finalize_save(fig, resolved)
        return saved if return_path else None
    return ax if not return_path else resolved

# ----------------------------------------------------------
# 2) Heteroscedastic noise profile (per-voxel SD vs distance)
# ----------------------------------------------------------
def plot_noise_profile(all_voxel_wise_dose_df, patient_id, bx_index,
                       target_stat="median",
                       ax=None,
                       save_path=None, file_name=None, return_path=False):
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
        fig, ax = plt.subplots(figsize=(7, 3.5))
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(per_voxel["x_mm"], np.sqrt(per_voxel["var_n"].clip(lower=0)), marker='o', lw=1)
    ax.set_xlabel("Distance along core (mm)")
    ax.set_ylabel("Independent SD (Gy)")
    ax.set_title(f"Per-voxel MC uncertainty — Patient {patient_id}, Bx {bx_index}")

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"noise_profile_patient_{patient_id}_bx_{bx_index}.png")
    if created_fig and resolved is not None:
        saved = _finalize_save(fig, resolved)
        return saved if return_path else None
    return ax if not return_path else resolved

# ----------------------------------------------------------------------
# 3) Uncertainty reduction: independent SD vs GP posterior SD at voxels
# ----------------------------------------------------------------------
def plot_uncertainty_reduction(all_voxel_wise_dose_df, semivariogram_df,
                               patient_id, bx_index,
                               target_stat="median", nu=1.5,
                               ax=None,
                               save_path=None, file_name=None, return_path=False, gp_res=None):
    out = predict_at_X(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index, target_stat, nu, res=gp_res)
    X, var_n, sd_X = out["X"], out["var_n"], out["sd_X"]
    indep_sd = np.sqrt(var_n)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(X, indep_sd, 'o-', ms=3, lw=1, label="Independent voxel SD")
    ax.plot(X, sd_X, 'o-', ms=3, lw=1, label="GP posterior SD")
    ax.set_xlabel("Distance along core (mm)")
    ax.set_ylabel("SD (Gy)")
    ax.set_title(f"Uncertainty reduction — Patient {patient_id}, Bx {bx_index}")
    ax.legend()

    # Optional inset with ratio
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset = inset_axes(ax, width="35%", height="45%", loc="upper right", borderpad=1)
        ratio = np.divide(indep_sd, sd_X, out=np.ones_like(indep_sd), where=sd_X > 0)
        inset.plot(X, ratio, '-', lw=1)
        inset.set_title("indep / GP SD")
        inset.tick_params(labelsize=8)
    except Exception:
        pass

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"uncertainty_reduction_patient_{patient_id}_bx_{bx_index}.png")
    if created_fig and resolved is not None:
        saved = _finalize_save(fig, resolved)
        return saved if return_path else None
    return ax if not return_path else resolved

# -------------------------------------------------------------
# 4) Residual diagnostics: residuals vs distance + std hist
# -------------------------------------------------------------
def plot_residuals(all_voxel_wise_dose_df, semivariogram_df,
                   patient_id, bx_index,
                   target_stat="median", nu=1.5,
                   save_path=None, file_name=None, return_path=False, gp_res=None):
    out = predict_at_X(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index, target_stat, nu, res=gp_res)
    X, y, mu_X, sd_X = out["X"], out["y"], out["mu_X"], out["sd_X"]
    res = y - mu_X

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].axhline(0, color='k', lw=1, alpha=0.5)
    axes[0].plot(X, res, 'o-', ms=3, lw=1)
    axes[0].set_xlabel("Distance (mm)"); axes[0].set_ylabel("Residual (Gy)")
    axes[0].set_title("Residuals vs distance")

    axes[1].hist(res / np.maximum(sd_X, 1e-12), bins=20, density=True, alpha=0.7)
    axes[1].set_title("Standardized residuals")
    axes[1].set_xlabel("r / sd"); axes[1].set_ylabel("Density")
    fig.suptitle(f"Diagnostics — Patient {patient_id}, Bx {bx_index}")

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"residuals_patient_{patient_id}_bx_{bx_index}.png")
    saved = _finalize_save(fig, resolved) if resolved is not None else None
    return (saved if return_path else axes)

# -----------------------------------------------------------------
# 5) Implied vs empirical semivariogram overlay (kernel-implied γ)
# -----------------------------------------------------------------
def plot_variogram_overlay(semivariogram_df, patient_id, bx_index, hyperparams,
                           ax=None, nu=None,
                           save_path=None, file_name=None, return_path=False):
    sv = semivariogram_df.query("`Patient ID`==@patient_id and `Bx index`==@bx_index").sort_values("h_mm")
    h = sv["h_mm"].to_numpy(float)
    gamma_hat = sv["semivariance"].to_numpy(float)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(h, gamma_hat, 'o', ms=4, label="Empirical γ̂(h)")

    nu_use = hyperparams.nu if nu is None else nu
    gamma_model = (GPR_analysis_pipeline_functions.matern_semivariogram(
                        h, hyperparams.sigma_f2, hyperparams.ell, 0.0, nu_use)
                   + hyperparams.nugget)
    ax.plot(h, gamma_model, '-', lw=2, label=f"Implied Matérn γ(h), ν={nu_use}")

    ax.set_xlabel("Lag h (mm)"); ax.set_ylabel("Semivariance γ(h) (Gy²)")
    ax.set_title(f"Variogram overlay — Patient {patient_id}, Bx {bx_index}")
    ax.legend()

    resolved = _resolve_save_path(save_path, file_name,
                                  default_fname=f"variogram_overlay_patient_{patient_id}_bx_{bx_index}.png")
    if created_fig and resolved is not None:
        saved = _finalize_save(fig, resolved)
        return saved if return_path else None
    return ax if not return_path else resolved
