from __future__ import annotations

import io
import warnings
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

import helper_funcs
import load_files
import misc_funcs
from pipeline_shared_config import SharedPipelineConfig, resolve_existing_output_dir


@dataclass(frozen=True)
class PipelineDataPaths:
    main_output_path: Path
    csv_directory: Path
    cohort_csvs_directory: Path
    csv_directory_for_distances: Path
    cohort_csvs_directory_for_distances: Path
    mc_sim_results_path: Path


@dataclass
class CommonLoadedData:
    config: SharedPipelineConfig
    paths: PipelineDataPaths
    master_structure_info_dict_results: Any | None
    cohort_3d_radiomic_features_all_oar_dil_df: pd.DataFrame
    cohort_biopsy_basic_spatial_features_df: pd.DataFrame
    cohort_biopsy_level_distances_statistics_filtered_df: pd.DataFrame
    cohort_voxel_level_distances_statistics_filtered_df: pd.DataFrame
    cohort_voxel_level_double_sextant_positions_filtered_df: pd.DataFrame
    cohort_global_dosimetry_df: pd.DataFrame
    cohort_global_dosimetry_by_voxel_df: pd.DataFrame
    all_point_wise_dose_df: pd.DataFrame
    all_voxel_wise_dose_df: pd.DataFrame
    all_mc_structure_transformation_df: pd.DataFrame
    all_differential_dvh_by_mc_trial_number_df: pd.DataFrame | None
    all_cumulative_dvh_by_mc_trial_number_df: pd.DataFrame | None
    uncertainties_df: pd.DataFrame
    valid_bx_keys: pd.DataFrame
    unique_patient_ids_all: list[str]
    unique_patient_ids_f1_prioritized: list[str]
    unique_patient_ids_f1: list[str]
    unique_patient_ids_f2: list[str]


def resolve_pipeline_paths(config: SharedPipelineConfig) -> PipelineDataPaths:
    resolved_main_output_path = resolve_existing_output_dir(config.main_output_path)
    resolved_distances_output_path = resolve_existing_output_dir(config.resolved_distances_output_path)
    csv_directory = resolved_main_output_path / "Output CSVs"
    csv_directory_for_distances = resolved_distances_output_path / "Output CSVs"
    return PipelineDataPaths(
        main_output_path=resolved_main_output_path,
        csv_directory=csv_directory,
        cohort_csvs_directory=csv_directory / "Cohort",
        csv_directory_for_distances=csv_directory_for_distances,
        cohort_csvs_directory_for_distances=csv_directory_for_distances / "Cohort",
        mc_sim_results_path=csv_directory / "MC simulation",
    )


def _add_iqr_ipr90_columns(df: pd.DataFrame, top_level_names: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in top_level_names:
        q25_key = (col, "quantile_25")
        q75_key = (col, "quantile_75")
        q05_key = (col, "quantile_05")
        q95_key = (col, "quantile_95")
        if q25_key in out.columns and q75_key in out.columns:
            out[(col, "IQR")] = out[q75_key] - out[q25_key]
        if q05_key in out.columns and q95_key in out.columns:
            out[(col, "IPR90")] = out[q95_key] - out[q05_key]
    return out


def _drop_bx_refnum_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    if isinstance(df.columns, pd.MultiIndex):
        to_drop = [
            col
            for col in df.columns
            if any(str(level) == "Bx refnum" for level in (col if isinstance(col, tuple) else (col,)))
        ]
        if to_drop:
            return df.drop(columns=to_drop)
        return df
    if "Bx refnum" in df.columns:
        return df.drop(columns=["Bx refnum"])
    return df


def _filter_by_valid_bx_pairs(
    df: pd.DataFrame,
    valid_bx_keys: pd.DataFrame,
    *,
    patient_col: Any,
    bx_index_col: Any,
) -> pd.DataFrame:
    valid_index = valid_bx_keys.set_index(["Patient ID", "Bx index"]).index
    dist_index = df.set_index([patient_col, bx_index_col]).index
    mask = dist_index.isin(valid_index)
    return df.loc[mask].copy()


def _filter_frame_by_simulation(df: pd.DataFrame, name: str, config: SharedPipelineConfig) -> pd.DataFrame:
    return helper_funcs.filter_df_by_sim_flags(
        df,
        name=name,
        sim_type_filter=config.sim_filter.simulated_type_filter,
        sim_bool_filter=config.sim_filter.simulated_bool_filter,
    )


def _load_and_concat_tables(
    root: Path,
    suffixes: Sequence[str],
    *,
    parquet: bool,
    verbose: bool,
) -> pd.DataFrame:
    paths = load_files.find_csv_files(root, list(suffixes))
    if not paths:
        raise FileNotFoundError(f"No files found under {root} matching suffixes {list(suffixes)}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        stdout_context = nullcontext() if verbose else redirect_stdout(io.StringIO())
        with stdout_context:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
                frame = (
                    load_files.load_parquet_as_dataframe(path)
                    if parquet
                    else load_files.load_csv_as_dataframe(path)
                )
        if frame is None:
            raise ValueError(f"Failed to load table from {path}")
        frames.append(frame)
    print(f"[BULK_LOAD] loaded {len(paths)} files matching {list(suffixes)} from {root}")
    return pd.concat(frames, ignore_index=True)


def _load_uncertainties_frame(main_output_path: Path) -> pd.DataFrame:
    csv_files = list(main_output_path.glob("uncertainties*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No uncertainties CSV found under {main_output_path}")
    df = load_files.load_csv_as_dataframe(csv_files[0])
    if df is None:
        raise ValueError(f"Failed to load uncertainties CSV {csv_files[0]}")
    return df


def build_dvh_metric_tables(
    voxel_wise_dose_df: pd.DataFrame,
    *,
    reference_dose_gy: float = 13.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_trial = helper_funcs.compute_dvh_metrics_per_trial_vectorized(
        voxel_wise_dose_df,
        d_perc_list=[2, 50, 98],
        v_perc_list=[100, 125, 150, 175, 200, 300],
        ref_dose_gy=reference_dose_gy,
        ref_dose_col=None,
        ref_dose_map=None,
        output_dir=None,
        csv_name=None,
    )
    per_biopsy = helper_funcs.build_dvh_summary_one_row_per_biopsy(per_trial)
    return per_trial, per_biopsy


def build_cumulative_dvh_table(voxel_wise_dose_df: pd.DataFrame) -> pd.DataFrame:
    return helper_funcs.build_cumulative_dvh_by_mc_trial_number_df(voxel_wise_dose_df)


def build_dataframe_inventory(
    common_data: CommonLoadedData,
    *,
    extra_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for field_name, value in common_data.__dict__.items():
        if isinstance(value, pd.DataFrame):
            rows.append(
                {
                    "table_name": field_name,
                    "n_rows": int(len(value)),
                    "n_columns": int(len(value.columns)),
                }
            )
    if extra_frames:
        for name, frame in extra_frames.items():
            rows.append(
                {
                    "table_name": name,
                    "n_rows": int(len(frame)),
                    "n_columns": int(len(frame.columns)),
                }
            )
    return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)


def load_common_data(
    config: SharedPipelineConfig,
    *,
    load_master_structure_dict: bool = False,
    load_differential_dvh: bool = True,
    load_cumulative_dvh_from_disk: bool = False,
) -> CommonLoadedData:
    paths = resolve_pipeline_paths(config)

    master_structure_info_dict_results = None
    if load_master_structure_dict:
        master_structure_info_dict_results = load_files.load_master_dict(
            config.main_output_path,
            "master_structure_info_dict_results",
        )

    cohort_3d_radiomic_features_all_oar_dil_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory / "Cohort: 3D radiomic features all OAR and DIL structures.csv"
    )
    cohort_biopsy_basic_spatial_features_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory / "Cohort: Biopsy basic spatial features dataframe.csv"
    )
    cohort_biopsy_basic_spatial_features_df = _filter_frame_by_simulation(
        cohort_biopsy_basic_spatial_features_df,
        "cohort_biopsy_basic_spatial_features_df",
        config,
    )
    cohort_biopsy_basic_spatial_features_df = _drop_bx_refnum_columns(
        cohort_biopsy_basic_spatial_features_df
    )

    valid_bx_keys = (
        cohort_biopsy_basic_spatial_features_df[["Patient ID", "Bx index"]]
        .drop_duplicates()
        .copy()
    )

    cohort_biopsy_level_distances_statistics_df = load_files.load_multiindex_csv(
        paths.cohort_csvs_directory_for_distances / "Cohort: Tissue class - distances global results.csv",
        header_rows=[0, 1],
    )
    cohort_biopsy_level_distances_statistics_filtered_df = _filter_by_valid_bx_pairs(
        cohort_biopsy_level_distances_statistics_df,
        valid_bx_keys,
        patient_col=("Patient ID", ""),
        bx_index_col=("Bx index", ""),
    )

    cohort_voxel_level_distances_statistics_df = load_files.load_multiindex_csv(
        paths.cohort_csvs_directory_for_distances / "Cohort: Tissue class - distances voxel-wise results.csv",
        header_rows=[0, 1],
    )
    cohort_voxel_level_distances_statistics_filtered_df = _filter_by_valid_bx_pairs(
        cohort_voxel_level_distances_statistics_df,
        valid_bx_keys,
        patient_col=("Patient ID", ""),
        bx_index_col=("Bx index", ""),
    )

    cohort_voxel_level_double_sextant_positions_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory_for_distances / "Cohort: Per voxel prostate double sextant classification.csv"
    )
    cohort_voxel_level_double_sextant_positions_df = _filter_frame_by_simulation(
        cohort_voxel_level_double_sextant_positions_df,
        "cohort_voxel_level_double_sextant_positions_df",
        config,
    )
    cohort_voxel_level_double_sextant_positions_filtered_df = cohort_voxel_level_double_sextant_positions_df.merge(
        valid_bx_keys,
        on=["Patient ID", "Bx index"],
        how="inner",
    )
    cohort_voxel_level_double_sextant_positions_filtered_df = _drop_bx_refnum_columns(
        cohort_voxel_level_double_sextant_positions_filtered_df
    )

    cohort_global_dosimetry_df = load_files.load_multiindex_csv(
        paths.cohort_csvs_directory / "Cohort: Global dosimetry (NEW).csv",
        header_rows=[0, 1],
    )
    cohort_global_dosimetry_df = _add_iqr_ipr90_columns(
        cohort_global_dosimetry_df,
        ["Dose (Gy)", "Dose grad (Gy/mm)"],
    )
    cohort_global_dosimetry_df = _filter_frame_by_simulation(
        cohort_global_dosimetry_df,
        "cohort_global_dosimetry_df",
        config,
    )
    cohort_global_dosimetry_df = _drop_bx_refnum_columns(cohort_global_dosimetry_df)

    cohort_global_dosimetry_by_voxel_df = load_files.load_multiindex_csv(
        paths.cohort_csvs_directory / "Cohort: Global dosimetry by voxel.csv",
        header_rows=[0, 1],
    )
    cohort_global_dosimetry_by_voxel_df = _add_iqr_ipr90_columns(
        cohort_global_dosimetry_by_voxel_df,
        ["Dose (Gy)", "Dose grad (Gy/mm)"],
    )
    cohort_global_dosimetry_by_voxel_df = _filter_frame_by_simulation(
        cohort_global_dosimetry_by_voxel_df,
        "cohort_global_dosimetry_by_voxel_df",
        config,
    )
    cohort_global_dosimetry_by_voxel_df = _drop_bx_refnum_columns(cohort_global_dosimetry_by_voxel_df)

    all_point_wise_dose_df = _load_and_concat_tables(
        paths.mc_sim_results_path,
        ["Point-wise dose output by MC trial number.parquet"],
        parquet=True,
        verbose=config.verbose_bulk_file_loading,
    )
    all_point_wise_dose_df = _filter_frame_by_simulation(all_point_wise_dose_df, "all_point_wise_dose_df", config)
    all_point_wise_dose_df = _drop_bx_refnum_columns(all_point_wise_dose_df)

    all_voxel_wise_dose_df = _load_and_concat_tables(
        paths.mc_sim_results_path,
        ["Voxel-wise dose output by MC trial number.parquet"],
        parquet=True,
        verbose=config.verbose_bulk_file_loading,
    )
    all_voxel_wise_dose_df = _filter_frame_by_simulation(all_voxel_wise_dose_df, "all_voxel_wise_dose_df", config)
    all_voxel_wise_dose_df = _drop_bx_refnum_columns(all_voxel_wise_dose_df)

    all_mc_structure_transformation_df = _load_and_concat_tables(
        paths.mc_sim_results_path,
        ["All MC structure transformation values.csv"],
        parquet=False,
        verbose=config.verbose_bulk_file_loading,
    )
    all_mc_structure_transformation_df = _filter_frame_by_simulation(
        all_mc_structure_transformation_df,
        "all_mc_structure_transformation_df",
        config,
    )

    all_differential_dvh_by_mc_trial_number_df: pd.DataFrame | None = None
    if load_differential_dvh:
        all_differential_dvh_by_mc_trial_number_df = _load_and_concat_tables(
            paths.mc_sim_results_path,
            ["Differential DVH by MC trial.parquet"],
            parquet=True,
            verbose=config.verbose_bulk_file_loading,
        )
        all_differential_dvh_by_mc_trial_number_df = _filter_frame_by_simulation(
            all_differential_dvh_by_mc_trial_number_df,
            "all_differential_dvh_by_mc_trial_number_df",
            config,
        )

    all_cumulative_dvh_by_mc_trial_number_df: pd.DataFrame | None = None
    if load_cumulative_dvh_from_disk:
        all_cumulative_dvh_by_mc_trial_number_df = _load_and_concat_tables(
            paths.mc_sim_results_path,
            ["Cumulative DVH by MC trial.parquet"],
            parquet=True,
            verbose=config.verbose_bulk_file_loading,
        )
        all_cumulative_dvh_by_mc_trial_number_df = _filter_frame_by_simulation(
            all_cumulative_dvh_by_mc_trial_number_df,
            "all_cumulative_dvh_by_mc_trial_number_df",
            config,
        )

    uncertainties_df = _load_uncertainties_frame(paths.main_output_path)

    unique_patient_ids_all = cohort_biopsy_basic_spatial_features_df["Patient ID"].unique().tolist()
    unique_patient_ids_f1_prioritized = misc_funcs.get_unique_patient_ids_fraction_prioritize(
        cohort_biopsy_basic_spatial_features_df,
        patient_id_col="Patient ID",
        priority_fraction="F1",
    )
    unique_patient_ids_f1 = misc_funcs.get_unique_patient_ids_fraction_specific(
        cohort_biopsy_basic_spatial_features_df,
        patient_id_col="Patient ID",
        fraction="F1",
    )
    unique_patient_ids_f2 = misc_funcs.get_unique_patient_ids_fraction_specific(
        cohort_biopsy_basic_spatial_features_df,
        patient_id_col="Patient ID",
        fraction="F2",
    )

    return CommonLoadedData(
        config=config,
        paths=paths,
        master_structure_info_dict_results=master_structure_info_dict_results,
        cohort_3d_radiomic_features_all_oar_dil_df=cohort_3d_radiomic_features_all_oar_dil_df,
        cohort_biopsy_basic_spatial_features_df=cohort_biopsy_basic_spatial_features_df,
        cohort_biopsy_level_distances_statistics_filtered_df=cohort_biopsy_level_distances_statistics_filtered_df,
        cohort_voxel_level_distances_statistics_filtered_df=cohort_voxel_level_distances_statistics_filtered_df,
        cohort_voxel_level_double_sextant_positions_filtered_df=cohort_voxel_level_double_sextant_positions_filtered_df,
        cohort_global_dosimetry_df=cohort_global_dosimetry_df,
        cohort_global_dosimetry_by_voxel_df=cohort_global_dosimetry_by_voxel_df,
        all_point_wise_dose_df=all_point_wise_dose_df,
        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
        all_mc_structure_transformation_df=all_mc_structure_transformation_df,
        all_differential_dvh_by_mc_trial_number_df=all_differential_dvh_by_mc_trial_number_df,
        all_cumulative_dvh_by_mc_trial_number_df=all_cumulative_dvh_by_mc_trial_number_df,
        uncertainties_df=uncertainties_df,
        valid_bx_keys=valid_bx_keys,
        unique_patient_ids_all=unique_patient_ids_all,
        unique_patient_ids_f1_prioritized=unique_patient_ids_f1_prioritized,
        unique_patient_ids_f1=unique_patient_ids_f1,
        unique_patient_ids_f2=unique_patient_ids_f2,
    )
