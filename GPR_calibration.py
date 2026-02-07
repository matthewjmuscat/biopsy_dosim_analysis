"""
Calibration utilities for per-biopsy GP residuals.

Computes calibration/normality diagnostics from standardized residuals and
optionally saves them to CSV for downstream plotting or reporting.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# SciPy is optional; fall back to NaN if unavailable
try:  # pragma: no cover - optional dependency
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover - tolerate missing SciPy
    stats = None

__all__ = [
    "compute_biopsy_calibration",
    "build_calibration_metrics",
    "save_calibration_metrics",
]


def _percent(mask: np.ndarray) -> float:
    """Return percentage (0-100) of True values in mask."""
    if mask.size == 0:
        return float("nan")
    return float(np.nanmean(mask) * 100.0)


def _get_array(obj, key_or_attr: str) -> np.ndarray:
    """Retrieve a field from dict or object; return as float ndarray."""
    if isinstance(obj, dict):
        val = obj.get(key_or_attr, [])
    else:
        val = getattr(obj, key_or_attr, [])
    return np.asarray(val, dtype=float)


def _safe_std_residuals(gp_res) -> np.ndarray:
    """Return finite standardized residuals."""
    y = _get_array(gp_res, "y")
    mu = _get_array(gp_res, "mu_X")
    sd = _get_array(gp_res, "sd_X")
    denom = np.maximum(sd, 1e-12)
    res_std = (y - mu) / denom
    res_std = res_std[np.isfinite(res_std)]
    return res_std


def compute_biopsy_calibration(
    patient_id,
    bx_index,
    gp_res,
    mean_bounds: Optional[Tuple[float, float]] = None,
    sd_bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Compute calibration metrics for a single biopsy using standardized residuals.
    Returns a flat dict suitable for DataFrame construction.
    """
    res_std = _safe_std_residuals(gp_res)
    n = res_std.size
    if n == 0:
        # Return NaNs for all metrics when data is missing
        return {
            "Patient ID": patient_id,
            "Bx index": bx_index,
            "n_resid": 0,
            "mean_resstd": float("nan"),
            "std_resstd": float("nan"),
            "skew_resstd": float("nan"),
            "kurt_resstd": float("nan"),
            "pct_abs_le1": float("nan"),
            "pct_abs_le2": float("nan"),
            "pct_abs_ge3": float("nan"),
            "pct_pos": float("nan"),
            "mean_abs_resstd": float("nan"),
            "median_abs_resstd": float("nan"),
            "ks_stat": float("nan"),
            "ks_pvalue": float("nan"),
            "ad_stat": float("nan"),
            "log_pdf_mean": float("nan"),
        }

    mean_resstd = float(np.nanmean(res_std))
    std_resstd = float(np.nanstd(res_std, ddof=1)) if n > 1 else float("nan")
    skew_resstd = float(
        np.nanmean(((res_std - mean_resstd) / (std_resstd or 1.0)) ** 3)
    ) if np.isfinite(std_resstd) and std_resstd > 0 else float("nan")
    kurt_resstd = float(
        np.nanmean(((res_std - mean_resstd) / (std_resstd or 1.0)) ** 4) - 3.0
    ) if np.isfinite(std_resstd) and std_resstd > 0 else float("nan")

    pct_abs_le1 = _percent(np.abs(res_std) <= 1.0)
    pct_abs_le2 = _percent(np.abs(res_std) <= 2.0)
    pct_abs_ge3 = _percent(np.abs(res_std) >= 3.0)
    pct_pos = _percent(res_std > 0)
    mean_abs_resstd = float(np.nanmean(np.abs(res_std)))
    median_abs_resstd = float(np.nanmedian(np.abs(res_std)))

    # KS / AD tests versus N(0,1) if SciPy is available
    ks_stat = ks_p = ad_stat = float("nan")
    if stats is not None and n > 0:
        try:
            ks_res = stats.kstest(res_std, "norm")
            ks_stat = float(ks_res.statistic)
            ks_p = float(ks_res.pvalue)
        except Exception:
            pass
        try:
            ad_res = stats.anderson(res_std, dist="norm")
            ad_stat = float(ad_res.statistic)
        except Exception:
            pass

    # Mean log predictive density (using normal with mu_X, sd_X)
    log_pdf_mean = float("nan")
    y = _get_array(gp_res, "y")
    mu = _get_array(gp_res, "mu_X")
    sd = _get_array(gp_res, "sd_X")
    mask = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    if mask.any():
        log_pdf = -0.5 * np.log(2 * np.pi * sd[mask] ** 2) - 0.5 * (
            (y[mask] - mu[mask]) ** 2 / (sd[mask] ** 2)
        )
        log_pdf_mean = float(np.nanmean(log_pdf))

    acceptable = float("nan")
    if mean_bounds is not None and sd_bounds is not None and np.isfinite(mean_resstd) and np.isfinite(std_resstd):
        acceptable = float(
            (mean_bounds[0] <= mean_resstd <= mean_bounds[1])
            and (sd_bounds[0] <= std_resstd <= sd_bounds[1])
        )

    return {
        "Patient ID": patient_id,
        "Bx index": bx_index,
        "n_resid": int(n),
        "mean_resstd": mean_resstd,
        "std_resstd": std_resstd,
        "skew_resstd": skew_resstd,
        "kurt_resstd": kurt_resstd,
        "pct_abs_le1": pct_abs_le1,
        "pct_abs_le2": pct_abs_le2,
        "pct_abs_ge3": pct_abs_ge3,
        "pct_pos": pct_pos,
        "mean_abs_resstd": mean_abs_resstd,
        "median_abs_resstd": median_abs_resstd,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
        "ad_stat": ad_stat,
        "log_pdf_mean": log_pdf_mean,
        "acceptable": acceptable,
    }


def build_calibration_metrics(
    results: Dict[Tuple, dict],
    mean_bounds: Optional[Tuple[float, float]] = None,
    sd_bounds: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Build a calibration metrics DataFrame for all biopsies in `results`,
    where results is a dict keyed by (patient_id, bx_index) -> gp_res.
    """
    rows = []
    for (pid, bx), res in results.items():
        rows.append(compute_biopsy_calibration(pid, bx, res, mean_bounds=mean_bounds, sd_bounds=sd_bounds))
    return pd.DataFrame(rows)


def save_calibration_metrics(calib_df: pd.DataFrame, output_dir: Path | str, filename: str = "gpr_calibration_metrics.csv") -> Path:
    """
    Save calibration DataFrame to CSV; returns the path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir.joinpath(filename)
    calib_df.to_csv(out_path, index=False)
    return out_path
