import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path




def compute_semivariogram_regular(df, voxel_size_mm=1.0, use_simulated=None, max_lag_voxels=None):
    """
    df: all_voxel_wise_dose_df filtered to a single biopsy (one Bx ID).
    Assumes every trial has same set/order of Voxel index.
    """
    # Optional filter: use only simulated trials or include nominal t=0 as well
    if use_simulated is True:
        df = df[df['Simulated bool'] == True]
    elif use_simulated is False:
        df = df[df['Simulated bool'] == False]
    # else: use all

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

def semivariogram_by_biopsy(all_df, voxel_size_mm=1.0, max_lag_voxels=None, use_simulated=None):
    """
    Compute semivariogram for each biopsy (grouped by Patient ID and Bx index).
    Returns one concatenated DataFrame with metadata columns attached.
    """
    results = []
    group_cols = ['Patient ID', 'Bx index']

    for (patient_id, bx_index), g in all_df.groupby(group_cols):
        sv = compute_semivariogram_regular(
            g,
            voxel_size_mm=voxel_size_mm,
            use_simulated=use_simulated,
            max_lag_voxels=max_lag_voxels
        )

        # Metadata (handle mixed cases explicitly)
        meta = {}
        for col in ['Simulated bool', 'Simulated type', 'Bx refnum', 'Bx ID']:
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
    meta_cols = ['Patient ID', 'Bx index', 'Bx refnum', 'Bx ID', 'Simulated bool', 'Simulated type']
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
        for c in ['Bx ID', 'Bx refnum', 'Simulated type', 'Simulated bool']:
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


def plot_variogram_from_df(
    semivariogram_df: pd.DataFrame,
    patient_id,
    bx_index,
    *,
    overlay_df: pd.DataFrame | None = None,  # optional precomputed overlay with columns ['h_mm', 'median_absdiff', 'mean_absdiff'] (any subset ok)
    include_title_meta: bool = True,
    save_path: str | Path | None = None,     # directory or full file path
    file_name: str | None = None,            # if dir provided, use this file name (ext optional)
    return_path: bool = False,               # return saved path for downstream use
    show: bool = False,                 # NEW: default to headless

) -> pd.DataFrame | tuple[pd.DataFrame, str | None]:
    """
    Plot an empirical variogram for one biopsy using an already computed semivariogram_df.

    Expects semivariogram_df to contain rows for many biopsies with at least:
      ['Patient ID', 'Bx index', 'h_mm', 'semivariance'].
    If overlay_df is provided, it should already be filtered to the same biopsy or be filterable.
    """
    if not show:
        plt.ioff()

    required_cols = {'Patient ID', 'Bx index', 'h_mm', 'semivariance'}
    missing = required_cols - set(semivariogram_df.columns)
    if missing:
        raise ValueError(f"semivariogram_df missing columns: {sorted(missing)}")

    # filter to the requested biopsy
    mask = (semivariogram_df['Patient ID'] == patient_id) & (semivariogram_df['Bx index'] == bx_index)
    sv = semivariogram_df.loc[mask].copy().sort_values('h_mm')
    if sv.empty:
        raise ValueError(f"No semivariogram rows for Patient ID={patient_id}, Bx index={bx_index}")

    # prepare overlay if provided
    ov = None
    if overlay_df is not None and not overlay_df.empty:
        # If overlay_df contains multiple biopsies, filter similarly (ignore if cols absent)
        ov = overlay_df.copy()
        if 'Patient ID' in ov.columns and 'Bx index' in ov.columns:
            ov = ov[(ov['Patient ID'] == patient_id) & (ov['Bx index'] == bx_index)]
        if not ov.empty and 'h_mm' in ov.columns:
            ov = ov.sort_values('h_mm')

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    sns.lineplot(data=sv, x='h_mm', y='semivariance', ax=ax,
                    marker='o', linewidth=2, label='Semivariogram γ(h)')

    if ov is not None and not ov.empty:
        if 'median_absdiff' in ov.columns:
            sns.lineplot(data=ov, x='h_mm', y='median_absdiff', ax=ax,
                            marker='s', linewidth=1.8, linestyle='--', label='Median |Δdose|')
        if 'mean_absdiff' in ov.columns:
            sns.lineplot(data=ov, x='h_mm', y='mean_absdiff', ax=ax,
                            marker='^', linewidth=1.8, linestyle=':', label='Mean |Δdose|')

    if include_title_meta:
        bits = [f"Patient {patient_id}", f"Biopsy {bx_index}"]
        for c in ['Bx ID', 'Bx refnum', 'Simulated type', 'Simulated bool']:
            if c in sv.columns:
                vals = sv[c].dropna().unique()
                if len(vals) == 1:
                    bits.append(f"{c}: {vals[0]}")
        ax.set_title(" | ".join(bits))

    ax.set_xlabel("Separation distance h (mm)")
    ax.set_ylabel("γ(h)  (semivariance)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()

    saved_path = None
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix:
            out_path = save_path
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            if file_name is None:
                file_name = f"variogram_patient{patient_id}_bx{bx_index}.png"
            if Path(file_name).suffix == "":
                file_name += ".png"
            out_path = save_path / file_name
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        saved_path = str(out_path)

    if show:
        plt.show()

    plt.close(fig)

    return (sv, saved_path) if return_path else sv


