# GPR CSV Data Dictionary

This document defines the CSV outputs produced by `main_pipe_GPR_analysis.py`,
including:

- where each file is written,
- when it is written (toggle/condition),
- every column in each CSV schema,
- how columns relate to each other.

The goal is auditability and traceability for production runs.

## 1) Output file inventory

## 1.1 Main output directory

Base directory: `output_data_GPR_analysis/`

Always or conditionally produced in main pipeline:

- `voxel_stats_cross_check_summary.csv`
- `voxel_stats_cross_check_mismatches_sample.csv` (only if mismatches exist)
- `semivariogram_method_parity/semivariogram_method_parity_summary.csv`
- `semivariogram_method_parity/semivariogram_method_parity_differences.csv`
- `gpr_per_biopsy_metrics.csv`
- `gpr_cohort_summary.csv`
- `gpr_by_patient_summary.csv`
- `cohort_mean_sd_regression_stats.csv` (only when `run_cohort_plots=True`)

Optional split summary CSVs (controlled by `write_split_main_cohort_summary_csvs`):

- `gpr_cohort_summary_ratio.csv`
- `gpr_cohort_summary_percent_reduction.csv`
- `gpr_cohort_summary_mc_sd.csv`
- `gpr_cohort_summary_gp_sd.csv`
- `gpr_cohort_summary_kernel_params.csv`

## 1.2 Kernel sensitivity output directory

Base directory: `output_data_GPR_analysis/kernel_sensitivity/csv/`

Always produced when `run_kernel_sensitivity_and_calibtration_flag=True`:

- `metrics_kernel_all.csv`
- `patient_rollup_all.csv`
- `cohort_summary_all.csv`
- `calibration_metrics_all.csv`
- `calibration_metrics_summary_all.csv`

Optional per-kernel CSVs:

- Controlled by `write_sensitivity_per_kernel_metrics_csvs`:
  - `metrics_kernel_<kernel>.csv`
  - `patient_rollup_<kernel>.csv`
  - `cohort_summary_<kernel>.csv`
- Controlled by `write_sensitivity_per_kernel_calibration_csvs`:
  - `calibration_metrics_<kernel>.csv`
  - `calibration_metrics_summary_<kernel>.csv`

If sensitivity is disabled, baseline calibration-only path writes:

- `kernel_sensitivity/csv/calibration_metrics_<BASE_KERNEL_LABEL>.csv`

## 1.3 blocked_CV output directory

Base directory: `output_data_GPR_analysis/blocked_CV/csv/`

Phase 3B currently produces:

- `blocked_cv_fold_map.csv`
- `blocked_cv_fold_summary.csv`

Phase 3D all-kernel fit/predict path produces:

- `blocked_cv_point_predictions_all.csv`
- `blocked_cv_fold_fit_status_all.csv`
- `blocked_cv_point_predictions_variance_compare_all.csv` (when variance-mode comparison is enabled)
- `blocked_cv_variance_mode_summary_all.csv` (when variance-mode comparison is enabled)

Optional per-kernel slices (gated by blocked_CV booleans in main):

- `blocked_cv_point_predictions_<kernel>.csv`
- `blocked_cv_fold_fit_status_<kernel>.csv`
- `blocked_cv_point_predictions_variance_compare_<kernel>.csv`
- `blocked_cv_variance_mode_summary_<kernel>.csv`

Notes:

- These files are generated only when `run_blocked_cv=True`.
- At Phase 3B, these are split-definition artifacts only (no CV model fitting).
- Phase 3D adds strict train-only fold fit/predict outputs for all configured kernels.
- Optional per-kernel CSV slices can be enabled from main; they are subsets of
  the centralized `_all` files.

## 1.4 Redundancy relationship

For sensitivity outputs, every per-kernel CSV is a subset of its `_all`
counterpart filtered by `kernel_label`.

---

## 2) Canonical schema: per-biopsy metrics CSV

File(s):

- `gpr_per_biopsy_metrics.csv`
- `metrics_kernel_all.csv`
- `metrics_kernel_<kernel>.csv` (optional)

Row granularity: one row per biopsy (`Patient ID`, `Bx index`) per kernel run.

## 2.1 Identifier and run-context columns

- `Patient ID`: biopsy patient id.
- `Bx index`: biopsy index within patient.
- `kernel_label` (optional in some contexts): kernel identifier
  (`matern_nu_1_5`, `matern_nu_2_5`, `rbf`, `exp`, etc.).
- `n_voxels`: number of voxel positions used in this biopsy GP fit.
- `spacing_mm`: robust voxel spacing estimate used for integrated SD metrics.
- `gp_mean_mode`: GP mean handling mode (`zero` or `ordinary`).
- `target_stat`: voxel target summary statistic (`median` or `mean`).
- `position_mode`: voxel position convention (`begin` or `center`).

## 2.2 SD and ratio columns

- `mean_indep_sd`: mean voxel MC SD (Gy), where `indep_sd = sqrt(var_n)`.
- `mean_gp_sd`: mean voxel GP posterior SD at training voxels (Gy).
- `mean_sd_mc`: alias of `mean_indep_sd`.
- `mean_sd_gp`: alias of `mean_gp_sd`.
- `mean_sd_ratio`: manuscript biopsy-level shrinkage ratio
  `mean_indep_sd / mean_gp_sd` (this corresponds to `\overline{R}_b` in the paper).
- `mean_ratio`: mean of voxelwise ratio `indep_sd / gp_sd`.
- `median_ratio`: median of voxelwise ratio `indep_sd / gp_sd`.
- `iqr_ratio`: IQR of voxelwise ratio `indep_sd / gp_sd`.
- `pct_vox_ratio_ge_1p25`: percent of voxels with ratio >= 1.25
  (>=20% SD reduction); canonical threshold column.
- `pct_vox_ratio_ge_1p50`: percent of voxels with ratio >= 1.5
  (>=33.3% SD reduction); canonical threshold column.
- `pct_vox_ge_20`: legacy alias of `pct_vox_ratio_ge_1p25`.
- `pct_vox_ge_50`: legacy alias of `pct_vox_ratio_ge_1p50`.

## 2.3 Integrated SD columns

- `integ_indep_sd`: `sum(indep_sd) * spacing_mm`.
- `integ_gp_sd`: `sum(gp_sd) * spacing_mm`.
- `int_sd_mc`: alias of `integ_indep_sd`.
- `int_sd_gp`: alias of `integ_gp_sd`.
- `integ_ratio`: `integ_indep_sd / integ_gp_sd`.

## 2.4 Percent-reduction columns

- `pct_reduction_mean_sd`: `100 * (1 - mean_gp_sd / mean_indep_sd)`.
- `pct_reduction_integ_sd`: `100 * (1 - integ_gp_sd / integ_indep_sd)`.
- `delta_mean_percent`: alias of `pct_reduction_mean_sd`.
- `delta_int_percent`: alias of `pct_reduction_integ_sd`.
- `pct_reduction_from_ratio`: `100 * (1 - 1 / mean_ratio)` using the mean of
  voxelwise ratios; this is not the manuscript `\overline{R}_b` definition.

## 2.5 Residual and calibration-support columns

- `mae_resid`: mean absolute residual `mean(|y - mu_X|)` (Gy).
- `rmse_resid`: root mean squared residual `sqrt(mean((y - mu_X)^2))` (Gy).
- `mean_y_gy`: mean of training targets `y` (Gy).
- `mean_muX_gy`: mean GP posterior mean at training voxels (Gy).
- `mean_residual_gy`: mean residual `mean(y - mu_X)` (Gy).
- `mean_resstd`: mean standardized residual
  `mean((y - mu_X) / max(sd_X, 1e-12))`.
- `std_resstd`: sample SD of standardized residuals.
- `skew_resstd`: skewness-like moment of standardized residuals.
- `kurt_resstd`: excess kurtosis-like moment of standardized residuals.

## 2.6 Hyperparameter and variogram-fit columns

- `ell`: fitted length scale (mm).
- `sigma_f2`: fitted signal variance.
- `nugget`: fitted nugget variance.
- `tau2`: alias of `nugget`.
- `nu`: Matérn smoothness parameter (or kernel-equivalent parameter).
- `nugget_fraction`: `nugget / (nugget + sigma_f2)`.
- `sv_rmse`: RMSE between empirical semivariogram and fitted model.

## 2.7 Mean-mode audit columns

- `m_b_hat_gy`: estimated constant GLS mean for ordinary kriging mode.
  `NaN` for zero-mean mode.
- `ones_cinv_ones`: denominator term `1^T C^{-1} 1` from GLS estimate.
- `solver_jitter`: jitter used in the GP solve.

---

## 3) Cohort summary schema

File(s):

- `gpr_cohort_summary.csv`
- `cohort_summary_all.csv`
- `cohort_summary_<kernel>.csv` (optional)

Row granularity:

- `gpr_cohort_summary.csv`: one row for the main baseline kernel run.
- `cohort_summary_all.csv`: one row per kernel (concatenated).

Columns:

- `n_biopsies`
- `mean_uncertainty_ratio`: cohort mean of `mean_sd_ratio`
- `median_uncertainty_ratio`: cohort median of `mean_sd_ratio`
- `mean_integrated_ratio`
- `uncertainty_ratio_q05`
- `uncertainty_ratio_q25`
- `uncertainty_ratio_q75`
- `uncertainty_ratio_q95`
- `uncertainty_ratio_iqr`
- `mean_sd_ratio_mean`: explicit alias of the cohort mean manuscript ratio
- `mean_sd_ratio_median`: explicit alias of the cohort median manuscript ratio
- `mean_sd_ratio_q05`
- `mean_sd_ratio_q25`
- `mean_sd_ratio_q75`
- `mean_sd_ratio_q95`
- `mean_sd_ratio_iqr`
- `voxelwise_mean_ratio_mean`: cohort mean of biopsy-level `mean_ratio`
- `voxelwise_mean_ratio_median`: cohort median of biopsy-level `mean_ratio`
- `voxelwise_median_ratio_mean`: cohort mean of biopsy-level `median_ratio`
- `voxelwise_median_ratio_median`: cohort median of biopsy-level `median_ratio`
- `pct_biopsies_ge20pct_reduction`
- `pct_biopsies_majority_vox_ratio_ge_1p25`
- `pct_reduction_mean_sd_mean`
- `pct_reduction_mean_sd_std`
- `pct_reduction_mean_sd_median`
- `pct_reduction_mean_sd_iqr`
- `pct_reduction_mean_sd_q05`
- `pct_reduction_mean_sd_q25`
- `pct_reduction_mean_sd_q75`
- `pct_reduction_mean_sd_q95`
- `pct_reduction_integ_sd_mean`
- `pct_reduction_integ_sd_std`
- `pct_reduction_integ_sd_median`
- `pct_reduction_integ_sd_iqr`
- `pct_reduction_integ_sd_q05`
- `pct_reduction_integ_sd_q25`
- `pct_reduction_integ_sd_q75`
- `pct_reduction_integ_sd_q95`
- `mc_sd_mean`
- `mc_sd_median`
- `mc_sd_q05`
- `mc_sd_q25`
- `mc_sd_q75`
- `mc_sd_q95`
- `mc_sd_iqr`
- `gp_sd_mean`
- `gp_sd_median`
- `gp_sd_q05`
- `gp_sd_q25`
- `gp_sd_q75`
- `gp_sd_q95`
- `gp_sd_iqr`
- `median_length_scale_mm`
- `median_nugget`
- `median_sv_rmse`
- `gp_mean_mode`
- `target_stat`
- `position_mode`
- `kernel_label` (present when kernel label is provided by caller)

---

## 4) Split cohort summary CSV schemas (optional)

These files are strict subsets of `gpr_cohort_summary.csv`.

- `gpr_cohort_summary_ratio.csv`:
  - `n_biopsies`
  - `mean_uncertainty_ratio`
  - `median_uncertainty_ratio`
  - `mean_integrated_ratio`
  - `uncertainty_ratio_q05`
  - `uncertainty_ratio_q25`
  - `uncertainty_ratio_q75`
  - `uncertainty_ratio_q95`
  - `uncertainty_ratio_iqr`
  - `pct_biopsies_ge20pct_reduction`

- `gpr_cohort_summary_percent_reduction.csv`:
  - `pct_reduction_mean_sd_mean`
  - `pct_reduction_mean_sd_std`
  - `pct_reduction_mean_sd_median`
  - `pct_reduction_mean_sd_iqr`
  - `pct_reduction_mean_sd_q05`
  - `pct_reduction_mean_sd_q25`
  - `pct_reduction_mean_sd_q75`
  - `pct_reduction_mean_sd_q95`
  - `pct_reduction_integ_sd_mean`
  - `pct_reduction_integ_sd_std`
  - `pct_reduction_integ_sd_median`
  - `pct_reduction_integ_sd_iqr`
  - `pct_reduction_integ_sd_q05`
  - `pct_reduction_integ_sd_q25`
  - `pct_reduction_integ_sd_q75`
  - `pct_reduction_integ_sd_q95`

- `gpr_cohort_summary_mc_sd.csv`:
  - `mc_sd_mean`, `mc_sd_median`, `mc_sd_q05`, `mc_sd_q25`,
    `mc_sd_q75`, `mc_sd_q95`, `mc_sd_iqr`

- `gpr_cohort_summary_gp_sd.csv`:
  - `gp_sd_mean`, `gp_sd_median`, `gp_sd_q05`, `gp_sd_q25`,
    `gp_sd_q75`, `gp_sd_q95`, `gp_sd_iqr`

- `gpr_cohort_summary_kernel_params.csv`:
  - `median_length_scale_mm`, `median_nugget`, `median_sv_rmse`

---

## 5) Patient rollup schema

File(s):

- `gpr_by_patient_summary.csv`
- `patient_rollup_all.csv`
- `patient_rollup_<kernel>.csv` (optional)

Row granularity: one row per patient (and per kernel for sensitivity runs).

Columns:

- `Patient ID`
- `n_bx`: number of biopsies for that patient.
- `mean_sd_ratio_mean`: patient mean of biopsy-level `mean_sd_ratio`.
- `mean_sd_ratio_sd`: patient SD of biopsy-level `mean_sd_ratio`.
- `mean_ratio_mean`: patient mean of biopsy-level `mean_ratio`.
- `mean_ratio_sd`: patient SD of biopsy-level `mean_ratio`.
- `ell_median`: patient median of biopsy-level `ell`.
- `kernel_label` (present when kernel label is provided by caller)

---

## 6) Calibration metrics schema

File(s):

- `calibration_metrics_all.csv`
- `calibration_metrics_<kernel>.csv` (optional; or baseline-only mode)

Row granularity: one row per biopsy (and per kernel when applicable).

Base columns:

- `Patient ID`
- `Bx index`
- `n_resid`: number of finite standardized residuals used.
- `mean_resstd`
- `std_resstd`
- `skew_resstd`
- `kurt_resstd`
- `pct_abs_le1`: percent with `|r_std| <= 1`.
- `pct_abs_le2`: percent with `|r_std| <= 2`.
- `pct_abs_ge3`: percent with `|r_std| >= 3`.
- `pct_pos`: percent with `r_std > 0`.
- `mean_abs_resstd`
- `median_abs_resstd`
- `ks_stat`: KS statistic vs N(0,1).
- `ks_pvalue`: KS p-value vs N(0,1).
- `ad_stat`: Anderson-Darling statistic vs normal.
- `log_pdf_mean`: mean log predictive density using `(y, mu_X, sd_X)`.
- `acceptable`: 1.0 if both mean/std are within configured bounds, else 0.0.

Run-context columns (added from result dict):

- `gp_mean_mode`
- `target_stat`
- `position_mode`

Sensitivity-only kernel column:

- `kernel_label` (added during kernel sensitivity aggregation path)

---

## 7) Calibration summary schema

File(s):

- `calibration_metrics_summary_all.csv`
- `calibration_metrics_summary_<kernel>.csv` (optional)

Row granularity: one row per calibration metric per kernel.

Columns:

- `kernel_label`
- `metric`: calibration column name summarized.
- `mean`
- `median`
- `iqr`
- `q05`
- `q25`
- `q75`
- `q95`

---

## 8) Mean SD regression stats schema

File:

- `cohort_mean_sd_regression_stats.csv`

Row granularity: single-row regression summary for mean SD scatter.

Columns:

- `n`
- `x_col`
- `y_col`
- `ols_slope`
- `ols_intercept`
- `ols_slope_ci_low`
- `ols_slope_ci_high`
- `ols_intercept_ci_low`
- `ols_intercept_ci_high`
- `ols_R2`
- `ols_slope_eq1_t`
- `ols_slope_eq1_p`
- `deming_slope`
- `deming_intercept`
- `origin_slope`
- `ols_sigma2`
- `ols_xbar`
- `ols_ybar`
- `ols_Sxx`
- `ols_df`
- `ols_tcrit`

---

## 9) Voxelwise cross-check CSV schemas

Files:

- `voxel_stats_cross_check_summary.csv`
- `voxel_stats_cross_check_mismatches_sample.csv` (if mismatches exist)

## 9.1 Summary file columns

- `metric`: flattened metric-stat key, e.g. `Dose (Gy)__mean`.
- `max_abs_diff`
- `mean_abs_diff`
- `n_mismatched`

## 9.2 Mismatch sample columns

Dynamic schema; each row corresponds to a metric that exceeded tolerance.
Columns include:

- `metric`
- key columns (typically): `Patient ID`, `Bx index`, `Bx ID`, `Voxel index`
- `<metric>`: recomputed value from raw MC data
- `<metric>__ref`: reference value from cohort summary table
- `abs_diff`

---

## 10) Semivariogram method parity CSV schemas

Files (under subfolder `semivariogram_method_parity/`):

- `semivariogram_method_parity_summary.csv`
- `semivariogram_method_parity_differences.csv`

## 10.1 Summary file columns

- `Patient ID`
- `Bx index`
- `n_lags_compared`: count of lag bins with finite semivariance from both methods.
- `max_abs_diff_semivariance`
- `mean_abs_diff_semivariance`
- `median_abs_diff_semivariance`
- `max_abs_diff_n_pairs`
- `mean_abs_diff_n_pairs`

## 10.2 Differences file columns

Per-lag comparison columns:

- `Patient ID`
- `Bx index`
- `lag_voxels`
- `h_mm`
- `semivariance_shift`
- `n_pairs_shift`
- `semivariance_pairwise`
- `n_pairs_pairwise`
- `abs_diff_semivariance`
- `abs_diff_n_pairs`

## 10.3 Method options and defaults

These controls are defined near the top of `main_pipe_GPR_analysis.py`:

- `semivariogram_method`
  - `"shift"`: legacy contiguous-lag semivariogram (current default for baseline path).
  - `"pairwise"`: gap-safe semivariogram based on pairwise physical lag distances.

- `semivariogram_pairwise_position_mode` (used when `semivariogram_method="pairwise"`)
  - `"begin"`: use `Voxel begin (Z)` as axial voxel position.
  - `"center"`: use midpoint of `Voxel begin (Z)` and `Voxel end (Z)`.

- `semivariogram_pairwise_lag_bin_width_mm` (used when `semivariogram_method="pairwise"`)
  - float in mm: explicit lag-bin width around each lag center.
  - `None`: defaults to `voxel_size_mm` (currently `1.0` mm in main).

---

## 11) blocked_CV fold map schemas (Phase 3B)

Files (under `blocked_CV/csv/`):

- `blocked_cv_fold_map.csv`
- `blocked_cv_fold_summary.csv`
- `blocked_cv_point_predictions_all.csv` (Phase 3D)
- `blocked_cv_fold_fit_status_all.csv` (Phase 3D)
- `blocked_cv_point_predictions_variance_compare_all.csv` (Phase 3D; optional)
- `blocked_cv_variance_mode_summary_all.csv` (Phase 3D; optional)

## 11.1 `blocked_cv_fold_map.csv` columns

One row per `(Patient ID, Bx index, fold_id, voxel)`:

- `Patient ID`
- `Bx index`
- `fold_id`
- `Voxel index`
- `x_mm`
- `is_test` (True if voxel is in held-out block for this fold)
- `n_train`
- `n_test`
- `effective_n_folds`
- `merged_tail_fold` (True when tiny tail fold merge was applied for this biopsy)
- `test_z_min_mm`
- `test_z_max_mm`
- `block_mode` (`equal_voxels` or `fixed_mm`)
- `position_mode` (`begin` or `center`)

## 11.2 `blocked_cv_fold_summary.csv` columns

One row per `(Patient ID, Bx index, fold_id)`:

- `Patient ID`
- `Bx index`
- `fold_id`
- `block_mode`
- `position_mode`
- `n_voxels`
- `n_train`
- `n_test`
- `effective_n_folds`
- `merged_tail_fold`
- `test_z_min_mm`
- `test_z_max_mm`
- `contiguous_test_block`

## 11.3 `blocked_cv_point_predictions_all.csv` columns

One row per held-out voxel prediction from strict train-only fold fit:

- `Patient ID`
- `Bx index`
- `fold_id`
- `kernel_label`
- `kernel_name`
- `kernel_param`
- `Voxel index`
- `x_mm`
- `y_test` (held-out MC voxel target)
- `mu_test` (GP predictive mean at held-out voxel)
- `sd_test_latent` (latent GP predictive SD)
- `var_obs_test` (held-out MC observation variance)
- `var_pred_used` (variance used for standardization/NLPD mode)
- `sd_pred_used`
- `residual` (`y_test - mu_test`)
- `rstd` (`residual / sd_pred_used`)
- `abs_res_over_sd_latent` (`|residual| / sd_test_latent`)
- `abs_res_over_sd_used` (`|residual| / sd_pred_used`)
- `gp_mean_mode`
- `target_stat`
- `predictive_variance_mode` (primary mode selected in main)
- `variance_modes_scored` (pipe-separated list of modes scored in this run)
- `n_train_voxels`
- `n_test_voxels`
- `ell`
- `sigma_f2`
- `nugget`
- `nu`

## 11.4 `blocked_cv_fold_fit_status_all.csv` columns

One row per `(Patient ID, Bx index, fold_id)` attempt:

- `Patient ID`
- `Bx index`
- `fold_id`
- `kernel_label`
- `kernel_name`
- `kernel_param`
- `primary_predictive_variance_mode`
- `variance_modes_scored`
- `status` (`ok`, `skipped`, or `error`)
- `message`
- `n_train_voxels` (when status is `ok`)
- `n_test_voxels` (when status is `ok`)

## 11.5 `blocked_cv_point_predictions_variance_compare_all.csv` columns

One row per held-out voxel *per variance mode* (same folds/predictions):

- `Patient ID`
- `Bx index`
- `fold_id`
- `kernel_label`
- `kernel_name`
- `kernel_param`
- `Voxel index`
- `x_mm`
- `y_test`
- `mu_test`
- `sd_test_latent`
- `var_obs_test`
- `variance_mode` (`latent`, `observed_mc`, `observed_mc_plus_nugget`)
- `var_pred_used`
- `sd_pred_used`
- `residual`
- `rstd`
- `abs_res_over_sd_latent`
- `abs_res_over_sd_used`
- `gp_mean_mode`
- `target_stat`
- `n_train_voxels`
- `n_test_voxels`
- `ell`
- `sigma_f2`
- `nugget`
- `nu`

## 11.6 `blocked_cv_variance_mode_summary_all.csv` columns

One row per `variance_mode` aggregated over all held-out points:

- `kernel_label`
- `kernel_name`
- `kernel_param`
- `variance_mode`
- `n_points`
- `mean_rstd`
- `sd_rstd`
- `pct_abs_le1`
- `pct_abs_le2`
- `median_abs_res_over_sd_used`

---

## 12) Notes on aliases and equivalences

These columns intentionally duplicate the same values for compatibility:

- `mean_indep_sd == mean_sd_mc`
- `mean_gp_sd == mean_sd_gp`
- `integ_indep_sd == int_sd_mc`
- `integ_gp_sd == int_sd_gp`
- `pct_reduction_mean_sd == delta_mean_percent`
- `pct_reduction_integ_sd == delta_int_percent`

If needed, downstream analysis can safely collapse these aliases.
