import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
from typing import Literal




def compute_semivariogram_regular(df, voxel_size_mm=1.0, max_lag_voxels=None):
    """
    Compute the empirical semivariogram for one already-filtered biopsy table.
    Assumes df contains only the trials/voxels you want (no in-function filtering).
    """
    # Pivot to matrix: rows = trials, cols = voxel index (ordered along the core)
    M = df.pivot_table(index='MC trial num', columns='Voxel index', values='Dose (Gy)', aggfunc='first')
    M = M.sort_index(axis=1)  # ensure voxel order
    D = M.values  # shape (T, N)

    T, N = D.shape
    if max_lag_voxels is None:
        max_lag_voxels = N - 1  # all possible lags

    # Pre-allocate
    lags = np.arange(1, max_lag_voxels + 1)
    gamma = np.empty_like(lags, dtype=float)
    npairs = np.empty_like(lags, dtype=int)

    # Compute semivariogram efficiently by shifting columns
    for k_idx, L in enumerate(lags):
        diffs = D[:, L:] - D[:, :-L]           # shape (T, N-L)
        gamma[k_idx] = 0.5 * np.mean(diffs**2) # average over trials and positions
        npairs[k_idx] = T * (N - L)

    out = pd.DataFrame({
        'lag_voxels': lags,
        'h_mm': lags * voxel_size_mm,
        'semivariance': gamma,
        'n_pairs': npairs
    })
    return out

# -------------------------------------------------------------
# semivariogram_by_biopsy:
#   - Groups the trial-wise voxel table by (Patient ID, Bx index).
#   - For each biopsy, computes the empirical semivariogram γ(h) via
#     compute_semivariogram_regular, then appends biopsy metadata and counts.
#   - Returns a single concatenated DataFrame for downstream GP fitting/plots.
#   - Assumes all_df has already been filtered upstream (e.g., by Simulated type).
# -------------------------------------------------------------
def semivariogram_by_biopsy(
    all_df,
    voxel_size_mm: float = 1.0,
    max_lag_voxels=None,
):
    """
    Compute semivariogram for each biopsy (grouped by Patient ID and Bx index).
    Assumes the input table is already filtered to the desired Simulated type(s)
    and trials; no filtering is performed here. Returns one concatenated
    DataFrame with metadata columns attached.

    Parameters
    ----------
    all_df : pd.DataFrame
        Trial-wise voxel table (already filtered).
    voxel_size_mm : float
        Physical size per voxel (used to convert lag_voxels -> h_mm).
    max_lag_voxels : int | None
        Maximum voxel lag to compute; None uses all available.
    """

    results = []
    group_cols = ['Patient ID', 'Bx index']

    for (patient_id, bx_index), g in all_df.groupby(group_cols):
        sv = compute_semivariogram_regular(
            g,
            voxel_size_mm=voxel_size_mm,
            max_lag_voxels=max_lag_voxels
        )

        # Metadata (handle mixed cases explicitly)
        meta = {}
        for col in ['Simulated bool', 'Simulated type', 'Bx ID']:
            if col in g.columns:
                vals = g[col].dropna().unique()
                if len(vals) == 1:
                    meta[col] = vals[0]
                elif len(vals) == 0:
                    meta[col] = np.nan
                else:
                    meta[col] = "Mixed"  # clearer than silently picking the first

        # Attach metadata + grouping keys
        sv['Patient ID'] = patient_id
        sv['Bx index'] = bx_index
        for k, v in meta.items():
            sv[k] = v

        # Helpful counts for weighting / QC
        try:
            n_trials = g['MC trial num'].nunique()
            n_voxels = g['Voxel index'].nunique()
            sv['n_trials'] = n_trials
            sv['n_voxels'] = n_voxels
        except Exception:
            pass

        results.append(sv)

    return pd.concat(results, ignore_index=True)


# -------------------------------------------------------------
# GP run + metrics aggregation (cohort-level helper)
# -------------------------------------------------------------
def run_gp_and_collect_metrics(
    all_voxel_wise_dose_df: pd.DataFrame,
    semivariogram_df: pd.DataFrame,
    output_dir,
    *,
    target_stat: str = "median",
    nu: float | None = None,
    kernel_spec=None,
    kernel_label: str | None = None,
    position_mode: Literal["center","begin"] = "center",
    save_csv: bool = True,
):
    """
    Runs the per-biopsy GP (posterior + hyperparams) on an already-filtered
    trial-wise voxel table, computes per-biopsy metrics, and writes cohort
    summary CSVs to output_dir. Returns (results_dict, metrics_df,
    cohort_summary_dict, by_patient_df).

    Assumes:
    - all_voxel_wise_dose_df and semivariogram_df are filtered to the same
      Simulated type subset upstream.
    - Grouping keys: Patient ID, Bx index.
    """
    # Local import to avoid circular dependency
    import GPR_analysis_pipeline_functions as gpr_pf

    # Validate Matérn ν availability when needed
    if kernel_spec is not None:
        if len(kernel_spec) != 2:
            raise ValueError("kernel_spec must be a 2-tuple: (kernel_name, param_or_None)")
        kname, kparam = kernel_spec
        if kname == "matern" and kparam is None and nu is None:
            raise ValueError("Matérn kernel requires ν; provide it via kernel_spec[1] or nu.")

    results = {}
    for (pid, bx_idx), _ in all_voxel_wise_dose_df.groupby(["Patient ID", "Bx index"]):
        res = gpr_pf.run_gp_for_biopsy(
            all_voxel_wise_dose_df,
            semivariogram_df,
            patient_id=pid,
            bx_index=bx_idx,
            target_stat=target_stat,
            nu=nu,
            kernel_spec=kernel_spec,
            position_mode=position_mode,
        )
        results[(pid, bx_idx)] = res
        print(f"Processed Patient ID: {pid}, Bx index: {bx_idx}")

    # Per-biopsy metrics table
    rows = []
    for (pid, bx_idx), res in results.items():
        row = gpr_pf.compute_per_biopsy_metrics(pid, bx_idx, res, semivariogram_df, kernel_label=kernel_label)
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    print("Per-biopsy metrics (head):")
    print(metrics_df.head())

    # Save metrics
    if save_csv:
        metrics_csv_path = output_dir.joinpath("gpr_per_biopsy_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved per-biopsy metrics to: {metrics_csv_path}")

    # Cohort summary numbers
    q = lambda s, p: float(s.quantile(p)) if not s.dropna().empty else float("nan")
    cohort_summary = {
        "n_biopsies": int(len(metrics_df)),
        "mean_uncertainty_ratio": float(metrics_df["mean_ratio"].mean()),
        "median_uncertainty_ratio": float(metrics_df["median_ratio"].median()),
        "mean_integrated_ratio": float(metrics_df["integ_ratio"].mean()),
        "uncertainty_ratio_q05": q(metrics_df["mean_ratio"], 0.05),
        "uncertainty_ratio_q25": q(metrics_df["mean_ratio"], 0.25),
        "uncertainty_ratio_q75": q(metrics_df["mean_ratio"], 0.75),
        "uncertainty_ratio_q95": q(metrics_df["mean_ratio"], 0.95),
        "uncertainty_ratio_iqr": q(metrics_df["mean_ratio"], 0.75) - q(metrics_df["mean_ratio"], 0.25),
        "pct_biopsies_ge20pct_reduction": float(
            (metrics_df["pct_vox_ge_20"] > 50).mean() * 100.0
        ),  # >50% of voxels get ≥20% reduction
        "pct_reduction_mean_sd_mean": float(metrics_df["pct_reduction_mean_sd"].mean()),
        "pct_reduction_mean_sd_std": float(metrics_df["pct_reduction_mean_sd"].std(ddof=1)),
        "pct_reduction_mean_sd_median": float(metrics_df["pct_reduction_mean_sd"].median()),
        "pct_reduction_mean_sd_iqr": float(
            metrics_df["pct_reduction_mean_sd"].quantile(0.75)
            - metrics_df["pct_reduction_mean_sd"].quantile(0.25)
        ),
        "pct_reduction_mean_sd_q05": q(metrics_df["pct_reduction_mean_sd"], 0.05),
        "pct_reduction_mean_sd_q25": q(metrics_df["pct_reduction_mean_sd"], 0.25),
        "pct_reduction_mean_sd_q75": q(metrics_df["pct_reduction_mean_sd"], 0.75),
        "pct_reduction_mean_sd_q95": q(metrics_df["pct_reduction_mean_sd"], 0.95),
        "pct_reduction_integ_sd_mean": float(metrics_df["pct_reduction_integ_sd"].mean()),
        "pct_reduction_integ_sd_std": float(metrics_df["pct_reduction_integ_sd"].std(ddof=1)),
        "pct_reduction_integ_sd_median": float(metrics_df["pct_reduction_integ_sd"].median()),
        "pct_reduction_integ_sd_iqr": float(
            metrics_df["pct_reduction_integ_sd"].quantile(0.75)
            - metrics_df["pct_reduction_integ_sd"].quantile(0.25)
        ),
        "pct_reduction_integ_sd_q05": q(metrics_df["pct_reduction_integ_sd"], 0.05),
        "pct_reduction_integ_sd_q25": q(metrics_df["pct_reduction_integ_sd"], 0.25),
        "pct_reduction_integ_sd_q75": q(metrics_df["pct_reduction_integ_sd"], 0.75),
        "pct_reduction_integ_sd_q95": q(metrics_df["pct_reduction_integ_sd"], 0.95),
        # MC SD across biopsies
        "mc_sd_mean": float(metrics_df["mean_indep_sd"].mean()),
        "mc_sd_median": float(metrics_df["mean_indep_sd"].median()),
        "mc_sd_q05": q(metrics_df["mean_indep_sd"], 0.05),
        "mc_sd_q25": q(metrics_df["mean_indep_sd"], 0.25),
        "mc_sd_q75": q(metrics_df["mean_indep_sd"], 0.75),
        "mc_sd_q95": q(metrics_df["mean_indep_sd"], 0.95),
        "mc_sd_iqr": q(metrics_df["mean_indep_sd"], 0.75) - q(metrics_df["mean_indep_sd"], 0.25),
        # GP SD across biopsies
        "gp_sd_mean": float(metrics_df["mean_gp_sd"].mean()),
        "gp_sd_median": float(metrics_df["mean_gp_sd"].median()),
        "gp_sd_q05": q(metrics_df["mean_gp_sd"], 0.05),
        "gp_sd_q25": q(metrics_df["mean_gp_sd"], 0.25),
        "gp_sd_q75": q(metrics_df["mean_gp_sd"], 0.75),
        "gp_sd_q95": q(metrics_df["mean_gp_sd"], 0.95),
        "gp_sd_iqr": q(metrics_df["mean_gp_sd"], 0.75) - q(metrics_df["mean_gp_sd"], 0.25),
        "median_length_scale_mm": float(metrics_df["ell"].median()),
        "median_nugget": float(metrics_df["nugget"].median()),
        "median_sv_rmse": float(metrics_df["sv_rmse"].median()),
    }
    if kernel_label:
        cohort_summary["kernel_label"] = kernel_label
    cohort_summary_df = pd.DataFrame([cohort_summary])
    if save_csv:
        cohort_summary_path = output_dir.joinpath("gpr_cohort_summary.csv")
        cohort_summary_df.to_csv(cohort_summary_path, index=False)
        print("Cohort summary:", cohort_summary)
        # Additional split CSVs for convenience (keep main summary unchanged)
        def _write_subset(filename: str, keys: list[str]):
            subset = {k: cohort_summary.get(k, float("nan")) for k in keys}
            pd.DataFrame([subset]).to_csv(output_dir.joinpath(filename), index=False)

        _write_subset(
            "gpr_cohort_summary_ratio.csv",
            [
                "n_biopsies",
                "mean_uncertainty_ratio",
                "median_uncertainty_ratio",
                "mean_integrated_ratio",
                "uncertainty_ratio_q05",
                "uncertainty_ratio_q25",
                "uncertainty_ratio_q75",
                "uncertainty_ratio_q95",
                "uncertainty_ratio_iqr",
                "pct_biopsies_ge20pct_reduction",
            ],
        )
        _write_subset(
            "gpr_cohort_summary_percent_reduction.csv",
            [
                "pct_reduction_mean_sd_mean",
                "pct_reduction_mean_sd_std",
                "pct_reduction_mean_sd_median",
                "pct_reduction_mean_sd_iqr",
                "pct_reduction_mean_sd_q05",
                "pct_reduction_mean_sd_q25",
                "pct_reduction_mean_sd_q75",
                "pct_reduction_mean_sd_q95",
                "pct_reduction_integ_sd_mean",
                "pct_reduction_integ_sd_std",
                "pct_reduction_integ_sd_median",
                "pct_reduction_integ_sd_iqr",
                "pct_reduction_integ_sd_q05",
                "pct_reduction_integ_sd_q25",
                "pct_reduction_integ_sd_q75",
                "pct_reduction_integ_sd_q95",
            ],
        )
        _write_subset(
            "gpr_cohort_summary_mc_sd.csv",
            [
                "mc_sd_mean",
                "mc_sd_median",
                "mc_sd_q05",
                "mc_sd_q25",
                "mc_sd_q75",
                "mc_sd_q95",
                "mc_sd_iqr",
            ],
        )
        _write_subset(
            "gpr_cohort_summary_gp_sd.csv",
            [
                "gp_sd_mean",
                "gp_sd_median",
                "gp_sd_q05",
                "gp_sd_q25",
                "gp_sd_q75",
                "gp_sd_q95",
                "gp_sd_iqr",
            ],
        )
        _write_subset(
            "gpr_cohort_summary_kernel_params.csv",
            [
                "median_length_scale_mm",
                "median_nugget",
                "median_sv_rmse",
            ],
        )

    # Patient-level rollups
    by_patient = (
        metrics_df.groupby("Patient ID")
        .agg(
            n_bx=("Bx index", "nunique"),
            mean_ratio_mean=("mean_ratio", "mean"),
            mean_ratio_sd=("mean_ratio", "std"),
            ell_median=("ell", "median"),
        )
        .reset_index()
    )
    if kernel_label:
        by_patient["kernel_label"] = kernel_label
    if save_csv:
        by_patient_path = output_dir.joinpath("gpr_by_patient_summary.csv")
        by_patient.to_csv(by_patient_path, index=False)

    return results, metrics_df, cohort_summary_df, by_patient



# ---------------------------------------
# Optional: median absolute-difference curve
# ---------------------------------------
def compute_median_absdiff_curve(
    df_one_biopsy: pd.DataFrame,
    voxel_size_mm: float = 1.0,
    max_lag_voxels: int | None = None,
    dose_col: str = 'Dose (Gy)',
    voxel_col: str = 'Voxel index',
    trial_col: str = 'MC trial num'
) -> pd.DataFrame:
    """
    For sanity checks vs your boxplots:
    median(|D(z) - D(z+h)|) and mean(|.|) at each lag h.
    """
    dose_matrix = df_one_biopsy.pivot_table(
        index=trial_col, columns=voxel_col, values=dose_col, aggfunc='first'
    ).sort_index(axis=1)
    dose_array = dose_matrix.values  # (T, N)
    n_trials, n_voxels = dose_array.shape
    if max_lag_voxels is None:
        max_lag_voxels = n_voxels - 1

    lags = np.arange(1, max_lag_voxels + 1)
    med_abs = np.empty_like(lags, dtype=float)
    mean_abs = np.empty_like(lags, dtype=float)

    for idx, L in enumerate(lags):
        diffs = dose_array[:, L:] - dose_array[:, :-L]     # (T, N-L)
        abs_diffs = np.abs(diffs).ravel()                  # flatten trials & positions
        med_abs[idx] = np.median(abs_diffs)
        mean_abs[idx] = np.mean(abs_diffs)

    out = pd.DataFrame({
        'lag_voxels': lags,
        'h_mm': lags * float(voxel_size_mm),
        'median_absdiff': med_abs,
        'mean_absdiff': mean_abs
    })
    # Copy metadata if present
    meta_cols = ['Patient ID', 'Bx index', 'Bx ID', 'Simulated bool', 'Simulated type']
    for c in meta_cols:
        if c in df_one_biopsy.columns:
            vals = df_one_biopsy[c].dropna().unique()
            out[c] = vals[0] if len(vals) else np.nan

    return out


# ---------------------------
# Seaborn plotting function
# ---------------------------
def plot_variogram_for_biopsy(
    all_df: pd.DataFrame,
    patient_id,
    bx_index,
    voxel_size_mm: float = 1.0,
    max_lag_voxels: int | None = None,
    overlay: str = 'none',  # 'none' | 'median_abs' | 'mean_abs' | 'both'
    include_title_meta: bool = True,
    save_path: str | None = None,   # NEW
    file_name: str | None = None    # NEW
):
    """
    Draw an empirical variogram for one biopsy (Patient ID + Bx index).
    Optionally save the figure to disk.

    Parameters
    ----------
    overlay : choose 'none', 'median_abs', 'mean_abs', or 'both'
    save_path : optional directory (or full path) to save the figure
    file_name : optional file name (default: auto-generated from Patient/Bx)
    """
    # filter to one biopsy
    mask = (all_df['Patient ID'] == patient_id) & (all_df['Bx index'] == bx_index)
    dfb = all_df.loc[mask].copy()
    if dfb.empty:
        raise ValueError(f"No rows found for Patient ID={patient_id}, Bx index={bx_index}")

    # compute curves
    sv = compute_semivariogram_regular(
        dfb, voxel_size_mm=voxel_size_mm, max_lag_voxels=max_lag_voxels
    )

    # optional overlays
    if overlay in ('median_abs', 'mean_abs', 'both'):
        abs_curve = compute_median_absdiff_curve(
            dfb, voxel_size_mm=voxel_size_mm, max_lag_voxels=max_lag_voxels
        )

    # ---- seaborn plot ----
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Variogram line
    sns.lineplot(
        data=sv, x='h_mm', y='semivariance', ax=ax, marker='o', linewidth=2, label='Semivariogram γ(h)'
    )

    # Optional overlays
    if overlay in ('median_abs', 'both'):
        sns.lineplot(
            data=abs_curve, x='h_mm', y='median_absdiff', ax=ax,
            marker='s', linewidth=1.8, linestyle='--', label='Median |Δdose|'
        )
    if overlay in ('mean_abs', 'both'):
        sns.lineplot(
            data=abs_curve, x='h_mm', y='mean_absdiff', ax=ax,
            marker='^', linewidth=1.8, linestyle=':', label='Mean |Δdose|'
        )

    # Title
    if include_title_meta:
        meta_bits = [f"Patient {patient_id}", f"Biopsy {bx_index}"]
        for c in ['Bx ID', 'Simulated type', 'Simulated bool']:
            if c in dfb.columns:
                vals = dfb[c].dropna().unique()
                if len(vals) == 1:
                    meta_bits.append(f"{c}: {vals[0]}")
        ax.set_title(" | ".join(meta_bits))

    ax.set_xlabel("Separation distance h (mm)")
    ax.set_ylabel("γ(h)  (semivariance)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()

    # ---- save or show ----
    if save_path:
        # build file path
        if file_name is None:
            file_name = f"variogram_patient{patient_id}_bx{bx_index}.png"
        elif not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            file_name += ".png"  # default extension
        full_path = save_path if save_path.endswith(file_name) else f"{save_path}/{file_name}"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {full_path}")
    else:
        plt.show()

    return sv


# NOTE: plot_variogram_from_df has been moved to GPR_production_plots.plot_variogram_from_df
# and should be imported/used directly from there.
