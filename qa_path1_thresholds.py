# qa_path1_thresholds.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, List

import pandas as pd


import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

def compute_nominal_core_averages_from_voxels(
    voxel_df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("Patient ID", "Bx index", "Bx ID"),
    trial_col: str = "MC trial num",
    dose_col: str = "Dose (Gy)",
    grad_col: str = "Dose grad (Gy/mm)",
) -> pd.DataFrame:
    """
    Returns one row per biopsy with core-averaged nominal predictors:
      - nominal_core_mean_dose_gy
      - nominal_core_mean_grad_gy_per_mm
    """
    nominal = voxel_df.loc[voxel_df[trial_col] == 0, list(group_cols) + [dose_col, grad_col]].copy()

    out = (
        nominal
        .groupby(list(group_cols), as_index=False)
        .agg(
            nominal_core_mean_dose_gy=(dose_col, "mean"),
            nominal_core_mean_grad_gy_per_mm=(grad_col, "mean"),
        )
    )
    return out


def attach_core_nominal_predictors(
    path1_df: pd.DataFrame,
    core_nominal_df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("Patient ID", "Bx index", "Bx ID"),
) -> pd.DataFrame:
    """Left-join predictors onto Path-1 rows (keeps path1_df row count unchanged)."""
    return path1_df.merge(core_nominal_df, on=list(group_cols), how="left")



@dataclass(frozen=True)
class ThresholdConfig:
    """
    Configuration for a single metric / threshold pair.

    metric_col : name of the DVH column, e.g. 'D_98% (Gy)'
    threshold  : numeric threshold in the same units as metric_col
    comparison : one-sided comparison, currently:
                 'ge'  -> metric >= threshold (default, most QA use case)
                 'le'  -> metric <= threshold
    label      : optional label that will show up in the output; if None,
                 a default like 'D_98% (Gy) ≥ 13.5' is used.
    """
    metric_col: str
    threshold: float
    comparison: str = "ge"
    label: str | None = None


def _apply_comparison(values: pd.Series, threshold: float, comparison: str) -> pd.Series:
    """Return a boolean Series indicating whether the pass criterion is met."""
    if comparison == "ge":
        return values >= threshold
    elif comparison == "le":
        return values <= threshold
    else:
        raise ValueError(f"Unsupported comparison '{comparison}', use 'ge' or 'le'.")


def compute_biopsy_threshold_probabilities(
    dvh_per_trial_df: pd.DataFrame,
    configs: Iterable[ThresholdConfig],
    group_cols: Tuple[str, ...] = ("Patient ID", "Bx index", "Bx ID"),
    trial_col: str = "MC trial num",
    prob_pass_cutoffs: Tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:

    # Work on a copy so we never mutate the original df
    dvh = dvh_per_trial_df.copy()

    # Build a stable group key (tuple per row)
    dvh["_group_key"] = list(zip(*[dvh[col] for col in group_cols]))

    # Split nominal vs MC trials
    nominal_mask = dvh[trial_col] == 0
    mc_mask = ~nominal_mask

    # Helper indexers
    nominal_by_key = dvh.loc[nominal_mask].set_index("_group_key")
    mc_df = dvh.loc[mc_mask]

    low_cut, high_cut = prob_pass_cutoffs
    rows: List[dict] = []

    for cfg in configs:
        metric = cfg.metric_col
        thr = cfg.threshold
        comp = cfg.comparison

        if metric not in dvh.columns:
            raise KeyError(f"Metric column '{metric}' not found in DVH dataframe.")

        label = cfg.label
        if label is None:
            label = f"{metric} ≥ {thr:g}" if comp == "ge" else f"{metric} ≤ {thr:g}"

        # Boolean "pass" per row for MC trials
        pass_mask = _apply_comparison(mc_df[metric], thr, comp)

        # Group by biopsy and compute probability & n_trials
        grouped_pass = pass_mask.groupby(mc_df["_group_key"], sort=False)
        p_pass = grouped_pass.mean()
        n_trials = grouped_pass.size()
        n_pass   = grouped_pass.sum()   # <-- add this


        for key, p in p_pass.items():
            # FIX #1: robust tuple-key lookup (works for Index-of-tuples and MultiIndex cases)
            n = int(n_trials.loc[[key]].iloc[0])
            k = int(n_pass.loc[[key]].iloc[0])

            # Get the representative group columns (using nominal row)
            try:
                nom_row = nominal_by_key.loc[[key]].iloc[0]   # <- key fix
            except KeyError:
                continue


            # FIX #2: if duplicates exist, .loc can return a DataFrame
            if isinstance(nom_row, pd.DataFrame):
                nom_row = nom_row.iloc[0]

            nominal_value = float(nom_row[metric])

            if comp == "ge":
                nominal_pass = nominal_value >= thr
                distance = nominal_value - thr
            else:  # 'le'
                nominal_pass = nominal_value <= thr
                distance = thr - nominal_value

            # Uncertainty-aware QA classification based on p_pass
            if p >= high_cut:
                qa_class = "confident pass"
            elif p <= low_cut:
                qa_class = "confident fail"
            else:
                qa_class = "borderline"

            # Misclassification flag (ignore borderline)
            if qa_class == "borderline":
                misclassified = False
                misclass_type = "borderline"
            else:
                if qa_class == "confident pass" and not nominal_pass:
                    misclassified = True
                    misclass_type = "nominal_underestimates"
                elif qa_class == "confident fail" and nominal_pass:
                    misclassified = True
                    misclass_type = "nominal_overestimates"
                else:
                    misclassified = False
                    misclass_type = "agree"

            row = {
                **{col: nom_row[col] for col in group_cols},
                "metric": metric,
                "threshold": thr,
                "comparison": comp,
                "label": label,
                "p_pass": float(p),
                "n_pass": k,
                "n_trials": n,
                "qa_class": qa_class,
                "nominal_value": nominal_value,
                "nominal_pass": nominal_pass,
                "distance_from_threshold_nominal": distance,
                "misclassified": misclassified,
                "misclassification_type": misclass_type,
            }
            rows.append(row)

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        result_df = result_df.sort_values(
            by=["metric", "threshold"] + list(group_cols)
        ).reset_index(drop=True)

    return result_df











def summarize_path1_by_threshold(
    path1_df: pd.DataFrame,
    label_col: str = "label",
    p_col: str = "p_pass",
    class_col: str = "qa_class",
    mis_col: str = "misclassified",
    margin_col: str = "distance_from_threshold_nominal",
) -> pd.DataFrame:
    """
    One row per threshold label with counts + basic distribution summaries.
    """
    def _summ(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_rows": len(g),
            "n_conf_pass": int((g[class_col] == "confident pass").sum()),
            "n_borderline": int((g[class_col] == "borderline").sum()),
            "n_conf_fail": int((g[class_col] == "confident fail").sum()),
            "misclass_rate_conf_only": float(
                g.loc[g[class_col] != "borderline", mis_col].mean()
            ) if (g[class_col] != "borderline").any() else np.nan,
            "p_pass_mean": float(g[p_col].mean()),
            "p_pass_median": float(g[p_col].median()),
            "margin_median": float(g[margin_col].median()),
        })

    return path1_df.groupby(label_col, as_index=False).apply(_summ).reset_index(drop=True)



def summarize_path1_by_threshold_v2(
    path1_df: pd.DataFrame,
    label_col: str = "label",
    p_col: str = "p_pass",
    class_col: str = "qa_class",
    mis_col: str = "misclassified",
    mis_type_col: str = "misclassification_type",
    margin_col: str = "distance_from_threshold_nominal",
    extra_numeric_cols: Tuple[str, ...] = (
        "nominal_value",
        "nominal_core_mean_dose_gy",
        "nominal_core_mean_grad_gy_per_mm",
    ),
) -> pd.DataFrame:
    """
    One row per threshold label with:
      - counts (conf pass / borderline / conf fail)
      - misclassification stats (conf-only)
      - rich distribution summaries for p_pass + nominal margin
        (mean, std, min/max, Q05/Q25/Q50/Q75/Q95, IQR, IPR90)
      - optional numeric columns (if present) summarized the same way
    """

    def _col_stats(s: pd.Series, prefix: str) -> Dict[str, float]:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_mean": np.nan,
                f"{prefix}_std": np.nan,
                f"{prefix}_min": np.nan,
                f"{prefix}_q05": np.nan,
                f"{prefix}_q25": np.nan,
                f"{prefix}_median": np.nan,
                f"{prefix}_q75": np.nan,
                f"{prefix}_q95": np.nan,
                f"{prefix}_max": np.nan,
                f"{prefix}_iqr": np.nan,
                f"{prefix}_ipr90": np.nan,
            }

        qs = s.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        q0, q05, q25, q50, q75, q95, q100 = (qs.loc[x] for x in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])

        return {
            f"{prefix}_n": int(len(s)),
            f"{prefix}_mean": float(s.mean()),
            f"{prefix}_std": float(s.std(ddof=1)),
            f"{prefix}_min": float(q0),
            f"{prefix}_q05": float(q05),
            f"{prefix}_q25": float(q25),
            f"{prefix}_median": float(q50),
            f"{prefix}_q75": float(q75),
            f"{prefix}_q95": float(q95),
            f"{prefix}_max": float(q100),
            f"{prefix}_iqr": float(q75 - q25),
            f"{prefix}_ipr90": float(q95 - q05),
        }

    rows: List[dict] = []

    for label, g in path1_df.groupby(label_col, sort=False):
        g = g.copy()
        conf_mask = g[class_col] != "borderline"

        n_rows = int(len(g))
        n_conf = int(conf_mask.sum())

        n_conf_pass = int((g[class_col] == "confident pass").sum())
        n_borderline = int((g[class_col] == "borderline").sum())
        n_conf_fail = int((g[class_col] == "confident fail").sum())

        # conf-only misclassification rate + breakdown
        if n_conf > 0:
            misclass_rate = float(g.loc[conf_mask, mis_col].mean())
            n_mis_conf = int(g.loc[conf_mask, mis_col].sum())
            n_over = int((g.loc[conf_mask, mis_type_col] == "nominal_overestimates").sum())
            n_under = int((g.loc[conf_mask, mis_type_col] == "nominal_underestimates").sum())
        else:
            misclass_rate = np.nan
            n_mis_conf = 0
            n_over = 0
            n_under = 0

        row = {
            "label": label,
            "n_rows": n_rows,
            "n_conf_pass": n_conf_pass,
            "n_borderline": n_borderline,
            "n_conf_fail": n_conf_fail,
            "n_conf_only": n_conf,
            "n_misclassified_conf_only": n_mis_conf,
            "misclass_rate_conf_only": misclass_rate,
            "n_nominal_overestimates_conf_only": n_over,
            "n_nominal_underestimates_conf_only": n_under,
            **_col_stats(g[p_col], "p_pass"),
            **_col_stats(g[margin_col], "margin"),
        }

        # optional columns (only if they exist)
        for c in extra_numeric_cols:
            if c in g.columns:
                row.update(_col_stats(g[c], c))

        rows.append(row)

    return pd.DataFrame(rows)



import numpy as np
import pandas as pd
from typing import Sequence, Tuple

def fit_path1_logit_per_threshold(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    metric_col: str = "metric",
    threshold_col: str = "threshold",
    p_pass_col: str = "p_pass",
    n_pass_col: str = "n_pass",
    n_trials_col: str = "n_trials",
    qa_class_col: str = "qa_class",
    predictors: Sequence[str] = ("distance_from_threshold_nominal",),
    margin_col: str = "distance_from_threshold_nominal",
    grad_col: str = "nominal_core_mean_grad_gy_per_mm",
    target_ps: Tuple[float, ...] = (0.5, 0.95),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit logistic models of p_pass vs predictors separately for each threshold label.

    The GLM is fit using statsmodels with a Binomial family and `freq_weights`
    equal to the Monte Carlo number of trials, but *reporting* metrics
    (AIC, Brier, RMSE, LR) are all scaled to an effective sample size equal
    to the number of biopsies so that their magnitude does not depend on the
    Monte Carlo trial count (e.g. 10,000).
    """
    import statsmodels.api as sm  # local import so the module still imports if SM is missing

    df = df.copy()

    required_cols = (
        {label_col, metric_col, threshold_col,
         p_pass_col, n_pass_col, n_trials_col}
        | set(predictors)
    )
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"fit_path1_logit_per_threshold: df is missing required columns: {sorted(missing)}"
        )

    # Consistent label ordering
    label_order = (
        df[[label_col, metric_col, threshold_col]]
        .drop_duplicates()
        .sort_values([metric_col, threshold_col])[label_col]
        .tolist()
    )

    coef_rows: list[dict] = []
    pred_rows: list[dict] = []

    for lbl in label_order:
        g = df[df[label_col] == lbl].copy()
        if g.empty:
            continue

        n_biopsies = int(len(g))

        # Response: Monte Carlo proportion, modelled as Binomial with freq_weights
        y = g[p_pass_col].to_numpy(dtype=float)
        y = np.clip(y, 1e-6, 1 - 1e-6)  # avoid exact 0/1

        w = g[n_trials_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError(f"Non-positive or non-finite {n_trials_col} for label={lbl!r}")

        # Weight scaling: MLEs are invariant to multiplying all weights by a constant,
        # but log-likelihood and AIC scale with that constant.
        w_mean = float(np.mean(w))

        X = g[list(predictors)].to_numpy(dtype=float)
        X_design = sm.add_constant(X, has_constant="add")

        # Full model
        model_full = sm.GLM(
            y,
            X_design,
            family=sm.families.Binomial(),
            freq_weights=w,
        )
        res_full = model_full.fit()

        # Null (intercept-only) model
        X0 = np.ones((len(g), 1))
        model_null = sm.GLM(
            y,
            X0,
            family=sm.families.Binomial(),
            freq_weights=w,
        )
        res_null = model_null.fit()

        # Effective (per-biopsy) log-likelihoods
        ll_full_raw = float(res_full.llf)
        ll_null_raw = float(res_null.llf)
        ll_eff = ll_full_raw / w_mean
        ll_null_eff = ll_null_raw / w_mean

        k_params = int(res_full.df_model) + 1  # intercept + slopes
        aic = -2.0 * ll_eff + 2.0 * k_params

        # McFadden R² (scale-invariant)
        mcfadden_r2 = 1.0 - (ll_full_raw / ll_null_raw)

        # Predictions at the observed biopsies
        p_hat = np.asarray(res_full.predict(X_design), dtype=float)

        # Per-biopsy predictive scores (not depending on n_trials)
        err2 = (y - p_hat) ** 2
        brier_w = float(np.mean(err2))
        rmse_prob_w = float(np.sqrt(brier_w))

        # Base coefficient row
        row: dict[str, object] = {
            label_col: lbl,
            metric_col: g[metric_col].iloc[0],
            threshold_col: g[threshold_col].iloc[0],
            "predictors": list(predictors),
            "n_biopsies": n_biopsies,
            "n_trials_mean": w_mean,
            "ll_eff": ll_eff,
            "ll_null_eff": ll_null_eff,
            "aic": aic,
            "mcfadden_r2": mcfadden_r2,
            "brier_w": brier_w,
            "rmse_prob_w": rmse_prob_w,
            "k_params": k_params,
        }

        # Coefficient names: b_const, b_<predictor>
        params = np.asarray(res_full.params, dtype=float)
        row["b_const"] = float(params[0])
        for j, name in enumerate(predictors, start=1):
            row[f"b_{name}"] = float(params[j])

        # Solve for δ̂_p at reference values of non-margin predictors
        if margin_col in predictors:
            other_preds = [p for p in predictors if p != margin_col]
            const_other = 0.0
            ref_values: dict[str, float] = {}

            for name in other_preds:
                beta_j = float(row[f"b_{name}"])
                x_ref = float(g[name].median())
                const_other += beta_j * x_ref
                ref_values[name] = x_ref

            beta_delta = float(row[f"b_{margin_col}"])

            # --- NEW: SD of margin and OR per 1 SD ---
            # Unweighted SD over biopsies (weights are MC trials, which are ~constant anyway)
            delta_vals = g[margin_col].to_numpy(dtype=float)
            if delta_vals.size > 1:
                delta_sd = float(np.std(delta_vals, ddof=1))
            else:
                delta_sd = np.nan

            row["margin_sd"] = delta_sd  # same units as margin (Gy or %-points)

            if (
                np.isfinite(delta_sd)
                and delta_sd > 0.0
                and abs(beta_delta) > 1e-12
            ):
                # OR corresponding to a 1-SD increase in nominal margin
                row["or_per_1sd_margin"] = float(np.exp(beta_delta * delta_sd))
            else:
                row["or_per_1sd_margin"] = np.nan
            # --- END NEW ---

            
            if abs(beta_delta) > 1e-12:
                for p_target in target_ps:
                    logit_p = float(np.log(p_target / (1.0 - p_target)))
                    delta_hat = (logit_p - row["b_const"] - const_other) / beta_delta
                    key = f"margin_at_p{int(round(p_target * 100))}"
                    if other_preds:
                        key += "_ref"
                    row[key] = float(delta_hat)

            for name, val in ref_values.items():
                row[f"{name}_ref_median"] = val

        coef_rows.append(row)

        # Build prediction rows for pred_df
        base_cols = [
            label_col,
            metric_col,
            threshold_col,
            p_pass_col,
            n_pass_col,
            n_trials_col,
        ] + list(predictors)

        if qa_class_col in g.columns:
            base_cols.append(qa_class_col)

        for idx_row, (_, r) in enumerate(g[base_cols].iterrows()):
            out = r.to_dict()
            out["p_hat_model"] = float(p_hat[idx_row])
            pred_rows.append(out)

    coef_df = pd.DataFrame(coef_rows)
    pred_df = pd.DataFrame(pred_rows)
    return coef_df, pred_df




import numpy as np
import pandas as pd

def compare_path1_logit_models(
    coef1_df: pd.DataFrame,
    coef2_df: pd.DataFrame,
    *,
    label_col: str = "label",
    metric_col: str = "metric",
    threshold_col: str = "threshold",
    suffix1: str = "_1d",
    suffix2: str = "_2d",
) -> pd.DataFrame:
    """
    Compare two sets of logistic models (e.g. 1D margin-only vs 2D margin+grad).

    Both inputs must be outputs from `fit_path1_logit_per_threshold`, and thus
    use per-biopsy log-likelihoods (ll_eff) for AIC and LR.
    """
    merged = pd.merge(
        coef1_df,
        coef2_df,
        on=[label_col],
        suffixes=(suffix1, suffix2),
        how="inner",
    )
    if merged.empty:
        raise ValueError("compare_path1_logit_models: no overlapping labels between models.")

    rows: list[dict] = []

    try:
        from scipy.stats import chi2  # type: ignore
        have_scipy = True
    except Exception:
        chi2 = None
        have_scipy = False

    for _, r in merged.iterrows():
        lbl = r[label_col]

        ll1 = float(r["ll_eff" + suffix1])
        ll2 = float(r["ll_eff" + suffix2])
        aic1 = float(r["aic" + suffix1])
        aic2 = float(r["aic" + suffix2])
        brier1 = float(r["brier_w" + suffix1])
        brier2 = float(r["brier_w" + suffix2])
        rmse1 = float(r["rmse_prob_w" + suffix1])
        rmse2 = float(r["rmse_prob_w" + suffix2])
        k1 = int(r["k_params" + suffix1])
        k2 = int(r["k_params" + suffix2])
        n_bx = int(r["n_biopsies" + suffix1])

        lr_df = max(k2 - k1, 1)
        lr_stat = 2.0 * (ll2 - ll1)

        if have_scipy and (lr_stat >= 0.0):
            lr_p = float(1.0 - chi2.cdf(lr_stat, lr_df))
        else:
            lr_p = float("nan")

        rows.append(
            {
                label_col: lbl,
                metric_col: r.get(metric_col + suffix1, np.nan),
                threshold_col: r.get(threshold_col + suffix1, np.nan),
                "n_biopsies": n_bx,
                "aic_model1": aic1,
                "aic_model2": aic2,
                "delta_aic": aic2 - aic1,          # 2D − 1D
                "brier_model1": brier1,
                "brier_model2": brier2,
                "delta_brier_w": brier2 - brier1,  # 2D − 1D
                "rmse_model1": rmse1,
                "rmse_model2": rmse2,
                "delta_rmse_prob_w": rmse2 - rmse1,
                "lr_stat": lr_stat,
                "lr_df": lr_df,
                "lr_pvalue": lr_p,
            }
        )

    return pd.DataFrame(rows)



def attach_margin_z_scores_from_trials(
    path1_df: pd.DataFrame,
    dvh_per_trial_df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("Patient ID", "Bx index", "Bx ID"),
    trial_col: str = "MC trial num",
    metric_col_in_path: str = "metric",
    z_col: str = "z_margin",
    std_col: str = "metric_std",
) -> pd.DataFrame:
    """
    Attach a dimensionless margin z-score to the Path 1 QA table.

    z = distance_from_threshold_nominal / STD(metric across MC trials)

    Parameters
    ----------
    path1_df
        Output of `compute_biopsy_threshold_probabilities(...)`.
        Must contain columns:
            - group_cols
            - `metric` (metric name, e.g. "D_98% (Gy)")
            - `distance_from_threshold_nominal`
    dvh_per_trial_df
        Output of `compute_dvh_metrics_per_trial_vectorized(...)`.
        Must contain:
            - group_cols
            - trial_col
            - the DVH metric columns used in Path 1 (e.g. "D_98% (Gy)", etc.)
    group_cols
        Columns defining a biopsy.
    trial_col
        Column indicating nominal vs MC trial number (nominal usually 0).
    metric_col_in_path
        Column in `path1_df` that holds the metric name (typically "metric").
    z_col
        Name of the output z-score column.
    std_col
        Name of the attached STD column.

    Returns
    -------
    DataFrame
        A *copy* of path1_df with two new columns:
            - std_col: STD of the DVH metric across MC trials
            - z_col: distance_from_threshold_nominal / STD
    """
    out = path1_df.copy()

    # Use only MC trials for variability estimate
    mc = dvh_per_trial_df.loc[dvh_per_trial_df[trial_col] != 0].copy()

    # Metrics actually present in the Path 1 table
    metrics = out[metric_col_in_path].unique().tolist()

    records = []
    # Loop over metrics (small, e.g. 4 thresholds) → OK
    for m in metrics:
        if m not in mc.columns:
            raise KeyError(
                f"Metric column '{m}' from Path 1 QA results "
                f"not found in dvh_per_trial_df."
            )

        grouped = (
            mc.groupby(list(group_cols), observed=False)[m]
              .agg(lambda x: float(np.nanstd(x.to_numpy(dtype=float), ddof=0)))
              .rename(std_col)
        )

        tmp = grouped.reset_index()
        tmp[metric_col_in_path] = m
        records.append(tmp)

    if not records:
        raise ValueError("No metrics found in path1_df to compute z-scores for.")

    sigma_df = pd.concat(records, ignore_index=True)

    # Merge STD back onto Path 1 rows (one STD per biopsy+metric)
    out = out.merge(
        sigma_df,
        on=[*group_cols, metric_col_in_path],
        how="left",
        validate="m:1",
    )

    # Compute z = delta / sigma, guarding against zero / NaN sigma
    denom = out[std_col].to_numpy()
    num = out["distance_from_threshold_nominal"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = num / denom

    # If STD <= 0 or NaN, set z to NaN
    bad = ~np.isfinite(denom) | (denom <= 0)
    z = np.where(bad, np.nan, z)

    out[z_col] = z

    return out




def compute_margin_correlations_by_threshold(
    design_df: pd.DataFrame,
    predictor_cols: list[str],
    target_col: str,
    label_col: str,
    min_n: int = 8,
) -> pd.DataFrame:
    """
    For each DVH rule (label), compute correlation between the margin
    (e.g. distance_from_threshold_nominal) and a set of predictor columns.

    One output row per (label, predictor), with Pearson and Spearman
    coefficients + p-values.

    The result is sorted within each rule by |Pearson r| (strongest first).
    """
    rows: list[dict] = []

    # Make sure the columns exist
    missing = [c for c in [target_col, label_col] if c not in design_df.columns]
    if missing:
        raise KeyError(f"design_df is missing required columns: {missing}")

    # Only keep predictors that actually exist in the dataframe
    available_predictors = [c for c in predictor_cols if c in design_df.columns]
    if not available_predictors:
        raise ValueError("None of the predictor_cols are present in design_df.")

    for rule_value, sub in design_df.groupby(label_col):
        # Drop rows with NaN in target once per rule
        sub = sub.dropna(subset=[target_col])

        for pred in available_predictors:
            # pairwise complete cases for this predictor
            temp = sub[[target_col, pred]].dropna()
            n = len(temp)
            if n < min_n:
                continue

            x = pd.to_numeric(temp[target_col], errors="coerce")
            y = pd.to_numeric(temp[pred], errors="coerce")
            valid = (~x.isna()) & (~y.isna())
            x = x[valid]
            y = y[valid]
            n = len(x)
            if n < min_n:
                continue

            # If no variance, correlation is undefined
            if x.nunique() <= 1 or y.nunique() <= 1:
                continue

            try:
                r_pearson, p_pearson = pearsonr(x, y)
            except Exception:
                r_pearson, p_pearson = np.nan, np.nan

            try:
                r_spear, p_spear = spearmanr(x, y)
            except Exception:
                r_spear, p_spear = np.nan, np.nan

            rows.append(
                {
                    label_col: rule_value,
                    "predictor": pred,
                    "N": n,
                    "pearson_r": r_pearson,
                    "pearson_p": p_pearson,
                    "spearman_rho": r_spear,
                    "spearman_p": p_spear,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                label_col,
                "predictor",
                "N",
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
            ]
        )

    out = pd.DataFrame(rows)

    # Add absolute value columns for sorting / inspection
    out["abs_pearson_r"] = out["pearson_r"].abs()
    out["abs_spearman_rho"] = out["spearman_rho"].abs()

    # *** This is the bit you were asking for ***
    # Sort *within each DVH rule* by |Pearson r| descending
    out = (
        out.sort_values(
            by=[label_col, "abs_pearson_r"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    return out






def summarize_margin_by_categorical_predictors(
    design_df: pd.DataFrame,
    categorical_cols: Sequence[str],
    margin_col: str = "distance_from_threshold_nominal",
    rule_col: str = "label",
) -> pd.DataFrame:
    """
    For each DVH threshold rule and each categorical predictor, compute
    the distribution of margin across categories.

    One row in the output = one (rule, predictor, category level).

    Parameters
    ----------
    design_df
        The big design matrix, e.g.
        `margin_predictors_with_radiomics_and_distances_df`.
    categorical_cols
        List of categorical predictor column names to summarise.
        (e.g. LR/AP/SI sextants, biopsy position categories)
    margin_col
        Name of the margin quantity; currently we use
        'distance_from_threshold_nominal'.
    rule_col
        Column that identifies the DVH rule / threshold label
        (e.g. "D98 ≥ 20 Gy").

    Returns
    -------
    pd.DataFrame
        Tall summary table with columns such as:
          ['Rule', 'Predictor', 'Level', 'N',
           'Margin mean', 'Margin std',
           'Margin median', 'Margin Q25', 'Margin Q75']
    """

    if margin_col not in design_df.columns:
        raise KeyError(f"{margin_col!r} not found in design_df")

    if rule_col not in design_df.columns:
        raise KeyError(f"{rule_col!r} not found in design_df")

    rows: list[dict] = []

    for rule, df_rule in design_df.groupby(rule_col):
        # For each DVH rule
        for cat_col in categorical_cols:
            if cat_col not in df_rule.columns:
                # Silently skip missing columns so the caller
                # can pass a unified list across variants.
                continue

            # Drop rows where this categorical variable is NaN
            df_cat = df_rule[[cat_col, margin_col]].dropna(subset=[cat_col])

            if df_cat.empty:
                continue

            for level, df_level in df_cat.groupby(cat_col):
                vals = df_level[margin_col].dropna()
                if vals.empty:
                    continue

                rows.append(
                    {
                        "Rule": rule,
                        "Predictor": cat_col,
                        "Level": level,
                        "N": len(vals),
                        "Margin mean": float(vals.mean()),
                        "Margin std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                        "Margin median": float(vals.median()),
                        "Margin Q25": float(vals.quantile(0.25)),
                        "Margin Q75": float(vals.quantile(0.75)),
                    }
                )

    return pd.DataFrame(rows)





import numpy as np
import pandas as pd
from math import log


def compute_delta95_vs_best_secondary_per_threshold(
    path1_results_df: pd.DataFrame,
    design_cc: pd.DataFrame,
    coef2_all_df: pd.DataFrame,
    best_per_rule: pd.DataFrame,
    *,
    metric_col: str = "metric",
    threshold_col: str = "threshold",
    label_col: str = "label",
    metric_std_col: str = "metric_std",
    secondary_col: str = "secondary_predictor",
    p_target: float = 0.95,
    low_q: float = 0.10,
    high_q: float = 0.90,
) -> pd.DataFrame:
    """
    For each DVH rule (metric / threshold / label), and its *best* secondary
    predictor (from best_per_rule), compute how the 95%-pass margin δ̂_0.95
    changes between a low and high quantile of that secondary predictor.

    Returns a tidy DataFrame with both:
      - absolute change in δ̂_0.95 in Gy (delta95_diff_high_minus_low)
      - the same change normalised by the median trial-wise STD of the metric
        across biopsies (delta95_diff_over_sigma_metric).
    """

    # --- cohort-level σ_T per DVH metric (from path1_results_df) ---
    metric_std_summary = (
        path1_results_df
        .copy()
        .dropna(subset=[metric_std_col])
        .groupby(metric_col, observed=False)[metric_std_col]
        .median()
        .reset_index()
        .rename(columns={metric_std_col: "metric_std_median"})
    )
    metric_std_lookup = dict(
        zip(metric_std_summary[metric_col], metric_std_summary["metric_std_median"])
    )

    def _delta_for_p(
        b0: float,
        b_delta: float,
        b_sec: float,
        sec_val: float,
        p: float,
    ) -> float:
        """Solve for δ such that logit(p) = b0 + b_delta*δ + b_sec*g."""
        if not np.isfinite(b_delta) or abs(b_delta) < 1e-9:
            return np.nan
        eta_target = log(p / (1.0 - p))
        return (eta_target - b0 - b_sec * sec_val) / b_delta

    delta95_effect_rows: list[dict] = []

    for _, row in best_per_rule.iterrows():
        metric = row[metric_col]
        thr = row[threshold_col]
        lbl = row[label_col]
        sec_col = row.get(secondary_col, None)

        # Skip if no secondary predictor recorded
        if not isinstance(sec_col, str) or not sec_col:
            continue

        # σ_T for this metric
        sigma_T = metric_std_lookup.get(metric, np.nan)

        # Restrict design_cc to this rule to get the secondary predictor distribution
        rule_mask = (
            (design_cc[metric_col] == metric)
            & (design_cc[threshold_col] == thr)
            & (design_cc[label_col] == lbl)
        )
        design_rule = design_cc.loc[rule_mask].copy()
        if design_rule.empty or sec_col not in design_rule.columns:
            continue

        sec_vals = pd.to_numeric(design_rule[sec_col], errors="coerce").dropna()
        if sec_vals.empty:
            continue

        sec_low, sec_high = sec_vals.quantile([low_q, high_q]).tolist()

        # Coefficients for the 2D model (margin + this secondary predictor)
        c2 = coef2_all_df[
            (coef2_all_df[metric_col] == metric)
            & (coef2_all_df[threshold_col] == thr)
            & (coef2_all_df[label_col] == lbl)
            & (coef2_all_df[secondary_col] == sec_col)
        ]
        if c2.empty:
            continue
        c2 = c2.iloc[0]

        try:
            b0 = float(c2["b_const"])
            b_delta = float(c2["b_distance_from_threshold_nominal"])
            b_sec = float(c2[f"b_{sec_col}"])
        except KeyError:
            # Missing expected coefficient column – skip this rule
            continue

        delta95_low = _delta_for_p(b0, b_delta, b_sec, sec_low, p=p_target)
        delta95_high = _delta_for_p(b0, b_delta, b_sec, sec_high, p=p_target)

        if not (np.isfinite(delta95_low) and np.isfinite(delta95_high)):
            continue

        delta95_diff = float(delta95_high - delta95_low)

        if np.isfinite(sigma_T) and sigma_T > 0:
            delta95_diff_over_sigma = delta95_diff / sigma_T
        else:
            delta95_diff_over_sigma = np.nan

        delta95_effect_rows.append(
            {
                metric_col: metric,
                threshold_col: thr,
                label_col: lbl,
                secondary_col: sec_col,
                "sec_q_low": low_q,
                "sec_q_high": high_q,
                "sec_val_low": float(sec_low),
                "sec_val_high": float(sec_high),
                "p_target": p_target,
                "delta95_at_low": float(delta95_low),
                "delta95_at_high": float(delta95_high),
                "delta95_diff_high_minus_low": delta95_diff,
                "metric_std_median": float(sigma_T) if np.isfinite(sigma_T) else np.nan,
                "delta95_diff_over_sigma_metric": delta95_diff_over_sigma,
            }
        )

    if not delta95_effect_rows:
        return pd.DataFrame(
            columns=[
                metric_col,
                threshold_col,
                label_col,
                secondary_col,
                "sec_q_low",
                "sec_q_high",
                "sec_val_low",
                "sec_val_high",
                "p_target",
                "delta95_at_low",
                "delta95_at_high",
                "delta95_diff_high_minus_low",
                "metric_std_median",
                "delta95_diff_over_sigma_metric",
            ]
        )

    return pd.DataFrame(delta95_effect_rows)




def drop_influential_d2_biopsy_by_cooks(path1_results_df: pd.DataFrame):
    """
    Identify a single influential biopsy for the D2% rule using
    Cook's distance from a margin-only logistic GLM, and drop it.

    Returns
    -------
    cleaned_df : DataFrame
        Copy of path1_results_df with the influential D2% biopsy removed
        (if any exceeds the Cook's D threshold).

    outlier_info : dict or None
        If an influential point was dropped, a dict with:
          - "index" : original index in path1_results_df
          - "Patient ID"
          - "Bx index"
          - "Bx ID"
          - "p_pass"
          - "distance_from_threshold_nominal"
        Otherwise None.

    cooks_max : float
        Maximum Cook's distance value for D2% biopsies.

    cooks_threshold : float
        Heuristic threshold used (4 / n_biopsies).
    """
    df = path1_results_df.copy()

    # Subset to the D2% rule only
    mask = df["metric"] == "D_2% (Gy)"
    g = df.loc[mask].copy()
    n = len(g)
    if n == 0:
        return df, None, np.nan, np.nan

    # Response: Monte Carlo pass probability, clipped away from 0 and 1
    y = np.clip(g["p_pass"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)

    # Predictor: nominal margin
    X = g["distance_from_threshold_nominal"].to_numpy(dtype=float)
    X_design = sm.add_constant(X, has_constant="add")

    # Weights: number of MC trials
    w = g["n_trials"].to_numpy(dtype=float)

    model = sm.GLM(
        y,
        X_design,
        family=sm.families.Binomial(),
        freq_weights=w,
    )
    res = model.fit()

    infl = res.get_influence(observed=True)
    cooks_d, _ = infl.cooks_distance

    cooks_max = float(np.max(cooks_d))
    cooks_threshold = 4.0 / n   # common rule-of-thumb for "large" influence

    if cooks_max > cooks_threshold:
        i_max = int(np.argmax(cooks_d))
        row = g.iloc[i_max]
        outlier_idx = g.index[i_max]

        outlier_info = {
            "index": outlier_idx,
            "Patient ID": row.get("Patient ID", None),
            "Bx index": row.get("Bx index", None),
            "Bx ID": row.get("Bx ID", None),
            "p_pass": float(row["p_pass"]),
            "distance_from_threshold_nominal": float(row["distance_from_threshold_nominal"]),
        }

        cleaned_df = df.drop(index=outlier_idx)
        return cleaned_df, outlier_info, cooks_max, cooks_threshold
    else:
        # No clearly influential biopsy
        return df, None, cooks_max, cooks_threshold
