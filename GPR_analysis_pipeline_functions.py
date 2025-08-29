from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Tuple, Dict, Optional
from scipy.optimize import curve_fit
from scipy.linalg import cho_factor, cho_solve
from math import isfinite
from scipy import stats
from pathlib import Path

# ----------------------------
# 0) Utilities / kernel models
# ----------------------------

@dataclass
class GPHyperparams:
    sigma_f2: float   # signal variance (sill - nugget)
    ell: float        # range / length-scale (mm)
    nugget: float     # micro-scale variance τ^2 (added to diagonal)
    nu: float = 1.5   # Matern smoothness (use 1.5 or 2.5 typically)

def matern_covariance(h: np.ndarray, sigma_f2: float, ell: float, nu: float) -> np.ndarray:
    """Isotropic Matérn covariance in 1D for distances h >= 0."""
    h = np.asarray(h, dtype=float)
    h_over_ell = np.maximum(h, 0.0) / max(ell, 1e-12)

    if nu == 0.5:
        # Exponential kernel
        return sigma_f2 * np.exp(-h_over_ell)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3.0)
        z = sqrt3 * h_over_ell
        return sigma_f2 * (1.0 + z) * np.exp(-z)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5.0)
        z = sqrt5 * h_over_ell
        return sigma_f2 * (1.0 + z + (z**2)/3.0) * np.exp(-z)
    else:
        # Fallback: use RBF as a smooth proxy if nu not in {0.5,1.5,2.5}
        return sigma_f2 * np.exp(-0.5 * (h_over_ell**2))

def matern_semivariogram(h: np.ndarray, sigma_f2: float, ell: float, nugget: float, nu: float) -> np.ndarray:
    """γ(h) = σ_f^2 - C(h) + nugget at h=0 (nugget handled separately when fitting)."""
    C = matern_covariance(h, sigma_f2, ell, nu)
    gamma = sigma_f2 - C
    # Note: nugget appears as discontinuity at h=0 in empirical γ; we fit it as a separate parameter.
    return gamma

def _variogram_model_for_fit(h, sigma_f2, ell, nugget, nu):
    # For curve_fit: add nugget on γ(0); for h>0, γ(h) = σ_f^2 - C(h). We approximate nugget as an additive constant.
    return matern_semivariogram(h, sigma_f2, ell, 0.0, nu) + nugget

# ---------------------------------------
# 1) Build voxel-level targets + variances
# ---------------------------------------

def build_voxel_targets_and_noise(
    all_voxel_wise_dose_df: pd.DataFrame,
    patient_id: str,
    bx_index: int,
    target_stat: Literal["median","mean"] = "median",
    variance_clip_min: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
        X (n,): voxel centers in mm along core
        y (n,): per-voxel target dose (Gy)
        var_n (n,): per-voxel heteroscedastic noise variances from MC (Gy^2)
        df_per_voxel: the per-voxel table (for debugging/plotting)
    """
    df_bx = all_voxel_wise_dose_df.query("`Patient ID` == @patient_id and `Bx index` == @bx_index")

    # Position along the core: use voxel center from begin/end; fallback to Z if needed.
    if {"Voxel begin (Z)","Voxel end (Z)"}.issubset(df_bx.columns):
        pos = df_bx.groupby("Voxel index").agg(
            x_mm=("Voxel begin (Z)", "first"),  # we'll replace with center below
            x_mm_end=("Voxel end (Z)", "first"),
        )
        pos["x_mm"] = 0.5 * (pos["x_mm"] + pos["x_mm_end"])
    else:
        # If only 'Z (Bx frame)' available, use its mean per voxel
        pos = df_bx.groupby("Voxel index").agg(x_mm=("Z (Bx frame)", "mean"))

    # Targets and MC variance per voxel
    agg_fn = "median" if target_stat == "median" else "mean"
    stats = df_bx.groupby("Voxel index").agg(
        y=("Dose (Gy)", agg_fn),
        var_n=("Dose (Gy)", "var"),
        n_trials=("MC trial num", "nunique")
    )

    per_voxel = pos[["x_mm"]].join(stats, how="inner").sort_values("x_mm").reset_index()

    X = per_voxel["x_mm"].to_numpy(float)
    y = per_voxel["y"].to_numpy(float)
    var_n = np.clip(per_voxel["var_n"].fillna(0.0).to_numpy(float), variance_clip_min, None)

    return X, y, var_n, per_voxel

# --------------------------------------------
# 2) Fit Matérn hyperparams from semivariogram
# --------------------------------------------

def fit_variogram_matern(
    semivariogram_df: pd.DataFrame,
    patient_id: str,
    bx_index: int,
    nu: float = 1.5,
    bounds: Optional[Dict[str, Tuple[float,float]]] = None
) -> GPHyperparams:
    """
    Fit (sigma_f2, ell, nugget) by least squares to the biopsy's empirical semivariogram.
    Expects columns: ['h_mm','semivariance','Patient ID','Bx index']
    """
    sv = semivariogram_df.query("`Patient ID` == @patient_id and `Bx index` == @bx_index").copy()
    sv = sv.sort_values("h_mm")
    h = sv["h_mm"].to_numpy(float)
    gamma_hat = sv["semivariance"].to_numpy(float)

    # Reasonable defaults for bounds
    if bounds is None:
        # sigma_f2: [small, large], ell: [small_mm, large_mm], nugget: [0, large]
        bounds = {
            "sigma_f2": (1e-6, max(gamma_hat.max()*10, 1e-3)),
            "ell": (np.maximum(np.diff(np.unique(h)).min() if h.size>1 else 0.1, 0.05), max(h.max()*5, 1.0)),
            "nugget": (0.0, max(gamma_hat.max(), 1.0))
        }

    p0 = [
        max(gamma_hat.max()*0.7, 1e-3),  # sigma_f2 init ~ sill
        max(h.max()/3.0, 1.0),           # ell init ~ range/3
        max(gamma_hat.min(), 1e-6)       # nugget init ~ near-origin γ
    ]

    lower = [bounds["sigma_f2"][0], bounds["ell"][0], bounds["nugget"][0]]
    upper = [bounds["sigma_f2"][1], bounds["ell"][1], bounds["nugget"][1]]

    popt, _ = curve_fit(lambda hh, s2, L, ng: _variogram_model_for_fit(hh, s2, L, ng, nu),
                        h, gamma_hat, p0=p0, bounds=(lower, upper), maxfev=10000)
    sigma_f2, ell, nugget = float(popt[0]), float(popt[1]), float(popt[2])
    return GPHyperparams(sigma_f2=sigma_f2, ell=ell, nugget=nugget, nu=nu)

# -------------------------------
# 3) Build GP matrices and solve
# -------------------------------

def build_kernel_matrix(Xa: np.ndarray, Xb: np.ndarray, hyp: GPHyperparams) -> np.ndarray:
    """K[a,b] = k(|Xa-Xb|)."""
    Xa = np.asarray(Xa, float)[:, None]
    Xb = np.asarray(Xb, float)[None, :]
    h = np.abs(Xa - Xb)
    return matern_covariance(h, hyp.sigma_f2, hyp.ell, hyp.nu)

def gp_posterior(
    X: np.ndarray, y: np.ndarray, var_n: np.ndarray, hyp: GPHyperparams,
    X_star: np.ndarray, jitter: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian-noise GP with heteroscedastic noise: Var(eps_i)=var_n[i]; plus nugget.
    Returns posterior mean (m,) and std (m,) at X_star.
    """
    Kxx = build_kernel_matrix(X, X, hyp)
    Sigma_eff = np.diag(var_n) + (hyp.nugget + jitter) * np.eye(len(X))

    # Cholesky solve
    L, lower = cho_factor(Kxx + Sigma_eff, lower=True, check_finite=False)
    Ksx = build_kernel_matrix(X_star, X, hyp)
    alpha = cho_solve((L, lower), y, check_finite=False)      # (n,)
    mu_star = Ksx @ alpha                                     # (m,)

    v = cho_solve((L, lower), Ksx.T, check_finite=False)      # (n,m)
    Kss = build_kernel_matrix(X_star, X_star, hyp)
    Sigma_star = Kss - (Ksx @ v)                              # (m,m)
    # Numerical safety
    var_star = np.clip(np.diag(Sigma_star), 0.0, None)
    return mu_star, np.sqrt(var_star)

# ----------------------------------------------
# 4) One-call convenience for a single biopsy
# ----------------------------------------------

def run_gp_for_biopsy(
    all_voxel_wise_dose_df: pd.DataFrame,
    semivariogram_df: pd.DataFrame,
    patient_id: str,
    bx_index: int,
    target_stat: Literal["median","mean"]="median",
    nu: float = 1.5,
    grid_mm: Optional[np.ndarray] = None
) -> Dict[str, object]:
    """
    Orchestrates: targets+noise -> variogram fit -> GP posterior on grid.
    Returns a dict with inputs, hyperparams, and posterior for plotting/paper.
    """
    # Step 1: voxel targets and heteroscedastic noise
    X, y, var_n, per_voxel = build_voxel_targets_and_noise(
        all_voxel_wise_dose_df, patient_id, bx_index, target_stat=target_stat
    )

    # Step 2: variogram fit → kernel hyperparams
    hyp = fit_variogram_matern(semivariogram_df, patient_id, bx_index, nu=nu)

    # Step 3/4: posterior on grid
    if grid_mm is None:
        X_star = np.linspace(X.min(), X.max(), max(200, 3*len(X)))
    else:
        X_star = np.asarray(grid_mm, float)

    mu_star, sd_star = gp_posterior(X, y, var_n, hyp, X_star)

    # Step 3b: posterior AT TRAINING VOXELS for residuals/metrics
    mu_X, sd_X = gp_posterior(X, y, var_n, hyp, X_star=X)

    return dict(
        patient_id=patient_id,
        bx_index=bx_index,
        per_voxel=per_voxel,       # table of X,y,var_n for debugging/plots
        hyperparams=hyp,           # sigma_f2, ell, nugget, nu
        X=X, y=y, var_n=var_n,
        X_star=X_star, mu_star=mu_star, sd_star=sd_star,
        mu_X=mu_X, sd_X=sd_X   # <-- added keys
    )















##### Uncertainty assessment metrics


def _safe_spacing(X):
    """Return a robust estimate of voxel spacing along the core in mm."""
    if len(X) < 2:
        return 1.0
    diffs = np.diff(np.sort(np.asarray(X)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else 1.0

def _finite(arr):
    arr = np.asarray(arr)
    return arr[np.isfinite(arr)]

def compute_per_biopsy_metrics(pid, bx_idx, res, semivariogram_df):
    """
    Build one row of metrics for a single biopsy using your existing 'res' dict.
    Expected keys in res: X, y, var_n, sd_X, mu_X, X_star, mu_star, sd_star, hyperparams
    """
    X        = np.asarray(res["X"])
    y        = np.asarray(res["y"])
    var_n    = np.asarray(res["var_n"])
    gp_sd    = np.asarray(res["sd_X"])
    mu_X     = np.asarray(res["mu_X"])
    hyp      = res["hyperparams"]

    indep_sd = np.sqrt(np.maximum(var_n, 0))
    spacing  = _safe_spacing(X)  # mm

    # --- uncertainty reduction (voxelwise and integrated) ---
    # Ratio (independent SD / GP SD), protect against div-by-zero
    ratio = np.divide(indep_sd, gp_sd, out=np.full_like(indep_sd, np.nan), where=(gp_sd > 0))
    ratio_f = _finite(ratio)

    mean_ratio   = float(np.nanmean(ratio_f)) if len(ratio_f) else np.nan
    median_ratio = float(np.nanmedian(ratio_f)) if len(ratio_f) else np.nan
    iqr_ratio    = float(np.nanpercentile(ratio_f, 75) - np.nanpercentile(ratio_f, 25)) if len(ratio_f) else np.nan
    pct_vox_ge_20 = float(np.nanmean(ratio_f >= 1.25) * 100) if len(ratio_f) else np.nan  # ≥20% reduction ~ ratio>=1.25


    # Integrated uncertainty (sum of SD * spacing) — crude trapezoid = spacing * sum(SD)
    integ_indep_sd = float(np.nansum(indep_sd) * spacing)
    integ_gp_sd    = float(np.nansum(gp_sd)    * spacing)
    integ_ratio    = (integ_indep_sd / integ_gp_sd) if (integ_gp_sd > 0) else np.nan
    mean_indep_sd  = float(np.nanmean(indep_sd)) if np.isfinite(indep_sd).any() else np.nan
    mean_gp_sd     = float(np.nanmean(gp_sd))    if np.isfinite(gp_sd).any()    else np.nan


    # percent reductions (protect against divide-by-zero)
    pct_reduction_mean_sd  = 100.0 * (1.0 - mean_gp_sd  / mean_indep_sd)  if mean_indep_sd  > 0 else np.nan
    pct_reduction_integ_sd = 100.0 * (1.0 - integ_gp_sd / integ_indep_sd) if integ_indep_sd > 0 else np.nan

    # (optional) derive from ratio for consistency checks
    pct_reduction_from_ratio = 100.0 * (1.0 - 1.0 / mean_ratio) if (mean_ratio is not np.nan and mean_ratio > 0) else np.nan

    # --- residual diagnostics (at training X) ---
    resids = y - mu_X
    resids_std = np.divide(resids, np.maximum(gp_sd, 1e-12))
    resids_f   = _finite(resids)
    resstd_f   = _finite(resids_std)

    mae_resid   = float(np.nanmean(np.abs(resids_f))) if len(resids_f) else np.nan
    rmse_resid  = float(np.sqrt(np.nanmean(resids_f**2))) if len(resids_f) else np.nan
    mean_resstd = float(np.nanmean(resstd_f)) if len(resstd_f) else np.nan
    std_resstd  = float(np.nanstd(resstd_f, ddof=1)) if len(resstd_f) > 1 else np.nan

    # crude skew/kurtosis (no SciPy)
    def _moments(a):
        a = _finite(a)
        if len(a) < 3: return np.nan, np.nan
        m = np.nanmean(a)
        s = np.nanstd(a, ddof=1)
        if not (isfinite(m) and isfinite(s) and s > 0): return np.nan, np.nan
        z = (a - m) / s
        skew = float(np.nanmean(z**3))
        kurt = float(np.nanmean(z**4)) - 3.0
        return skew, kurt
    skew_resstd, kurt_resstd = _moments(resstd_f)

    # --- hyperparams + variogram goodness-of-fit ---
    ell       = float(getattr(hyp, "ell", np.nan))
    sigma_f2  = float(getattr(hyp, "sigma_f2", np.nan))
    nugget    = float(getattr(hyp, "nugget", np.nan))
    nu_param  = float(getattr(hyp, "nu", np.nan))

    # optional: empirical vs implied semivariogram RMSE on this biopsy
    sv = semivariogram_df.query("`Patient ID`==@pid and `Bx index`==@bx_idx").sort_values("h_mm")
    if len(sv):
        h = sv["h_mm"].to_numpy(float)
        gamma_hat = sv["semivariance"].to_numpy(float)
        gamma_model = (
            matern_semivariogram(
                h, sigma_f2, ell, 0.0, nu_param
            ) + nugget
        )
        sv_rmse = float(np.sqrt(np.nanmean((gamma_hat - gamma_model)**2)))
    else:
        sv_rmse = np.nan

    return dict(
        **{"Patient ID": pid, "Bx index": bx_idx, "n_voxels": int(len(X)), "spacing_mm": spacing},
        mean_indep_sd=mean_indep_sd,
        mean_gp_sd=mean_gp_sd,
        mean_ratio=mean_ratio,
        median_ratio=median_ratio,
        iqr_ratio=iqr_ratio,
        pct_vox_ge_20=pct_vox_ge_20,
        integ_indep_sd=integ_indep_sd,
        integ_gp_sd=integ_gp_sd,
        integ_ratio=integ_ratio,
        pct_reduction_mean_sd=pct_reduction_mean_sd,
        pct_reduction_integ_sd=pct_reduction_integ_sd,
        pct_reduction_from_ratio=pct_reduction_from_ratio,
        mae_resid=mae_resid,
        rmse_resid=rmse_resid,
        mean_resstd=mean_resstd,
        std_resstd=std_resstd,
        skew_resstd=skew_resstd,
        kurt_resstd=kurt_resstd,
        ell=ell, sigma_f2=sigma_f2, nugget=nugget, nu=nu_param,
        sv_rmse=sv_rmse
    )






# ---------------------------------------------------------------------
# 1) Fit function: OLS + Deming (λ=1); save & return a one-row DataFrame
# ---------------------------------------------------------------------
def fit_mean_sd_regressions(
    metrics_df: pd.DataFrame,
    save_csv_path: Optional[Path] = None,
    x_col: str = "mean_indep_sd",
    y_col: str = "mean_gp_sd",
) -> pd.DataFrame:
    """
    Fits OLS (y = a + b x) and Deming (λ=1) between per-biopsy mean SDs.
    Returns a one-row DataFrame with slopes, intercepts, CIs, R^2, etc.,
    plus extra fields needed to draw a 95% CI ribbon (no refit in the plot).
    """
    x = metrics_df[x_col].to_numpy(dtype=float)
    y = metrics_df[y_col].to_numpy(dtype=float)
    msk = np.isfinite(x) & np.isfinite(y)
    x, y = x[msk], y[msk]
    n = x.size

    out = {
        # original fields
        "n": n, "x_col": x_col, "y_col": y_col,
        "ols_slope": np.nan, "ols_intercept": np.nan,
        "ols_slope_ci_low": np.nan, "ols_slope_ci_high": np.nan,
        "ols_intercept_ci_low": np.nan, "ols_intercept_ci_high": np.nan,
        "ols_R2": np.nan, "ols_slope_eq1_t": np.nan, "ols_slope_eq1_p": np.nan,
        "deming_slope": np.nan, "deming_intercept": np.nan,
        "origin_slope": np.nan,
        # new: for CI ribbons (mean & prediction)
        "ols_sigma2": np.nan,   # residual variance
        "ols_xbar": np.nan,
        "ols_ybar": np.nan,
        "ols_Sxx": np.nan,
        "ols_df": np.nan,
        "ols_tcrit": np.nan,    # t_{0.975, df}
    }

    if n >= 3:
        xbar, ybar = x.mean(), y.mean()
        Sxx = np.sum((x - xbar)**2)
        Syy = np.sum((y - ybar)**2)
        Sxy = np.sum((x - xbar)*(y - ybar))

        if Sxx > 0:
            # -------- OLS --------
            b = Sxy / Sxx
            a = ybar - b * xbar
            yhat = a + b * x
            SSE  = np.sum((y - yhat)**2)
            SST  = np.sum((y - ybar)**2)
            R2   = 1.0 - SSE / SST if SST > 0 else np.nan

            df = n - 2
            sigma2 = SSE / df
            se_b = np.sqrt(sigma2 / Sxx)
            se_a = np.sqrt(sigma2 * (1.0/n + xbar**2 / Sxx))

            tcrit = stats.t.ppf(0.975, df=df)
            b_ci = (b - tcrit*se_b, b + tcrit*se_b)
            a_ci = (a - tcrit*se_a, a + tcrit*se_a)

            # test slope == 1
            t_eq1 = (b - 1.0) / se_b if se_b > 0 else np.nan
            p_eq1 = 2 * stats.t.sf(np.abs(t_eq1), df=df) if np.isfinite(t_eq1) else np.nan

            out.update({
                "ols_slope": b, "ols_intercept": a, "ols_R2": R2,
                "ols_slope_ci_low": b_ci[0], "ols_slope_ci_high": b_ci[1],
                "ols_intercept_ci_low": a_ci[0], "ols_intercept_ci_high": a_ci[1],
                "ols_slope_eq1_t": t_eq1, "ols_slope_eq1_p": p_eq1,
                # CI ribbon ingredients
                "ols_sigma2": sigma2, "ols_xbar": xbar, "ols_ybar": ybar,
                "ols_Sxx": Sxx, "ols_df": df, "ols_tcrit": tcrit,
            })

        # -------- Deming (orthogonal, λ=1) --------
        if Sxy != 0:
            Delta = Syy - Sxx
            b_dem = (Delta + np.sqrt(Delta**2 + 4*Sxy**2)) / (2*Sxy)
            a_dem = ybar - b_dem * xbar
            out.update({"deming_slope": b_dem, "deming_intercept": a_dem})

        # -------- Through-origin (optional ref) --------
        denom = np.dot(x, x)
        if denom > 0:
            out["origin_slope"] = float(np.dot(x, y) / denom)

    df = pd.DataFrame([out])
    if save_csv_path is not None:
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv_path, index=False)
    return df















