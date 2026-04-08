from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from load_data_shared import (
    CommonLoadedData,
    build_cumulative_dvh_table,
    build_dvh_metric_tables,
    load_common_data,
)
from pipeline_shared_config import ExemplarSelectionConfig, SharedPipelineConfig


@dataclass(frozen=True)
class SelectedExemplar:
    patient_id: str
    bx_index: int
    bx_id: str
    display_label: str


@dataclass
class ExemplarLoadedData:
    common: CommonLoadedData
    calculated_dvh_metrics_per_trial_df: pd.DataFrame | None
    cohort_global_dosimetry_dvh_metrics_df: pd.DataFrame | None
    all_cumulative_dvh_by_mc_trial_number_df: pd.DataFrame | None
    selected_exemplars: list[SelectedExemplar]


def _resolve_selected_exemplars(
    biopsy_basic_df: pd.DataFrame,
    selection_config: ExemplarSelectionConfig,
) -> list[SelectedExemplar]:
    selected: list[SelectedExemplar] = []
    for patient_id, bx_index in selection_config.biopsy_pairs:
        match = biopsy_basic_df[
            (biopsy_basic_df["Patient ID"] == patient_id)
            & (biopsy_basic_df["Bx index"].astype(int) == int(bx_index))
        ]
        if match.empty:
            raise ValueError(f"Selected exemplar ({patient_id}, {bx_index}) not found in biopsy basic table.")
        bx_id = str(match.iloc[0]["Bx ID"])
        display_label = str(selection_config.display_label_map.get((patient_id, int(bx_index)), bx_id))
        selected.append(
            SelectedExemplar(
                patient_id=str(patient_id),
                bx_index=int(bx_index),
                bx_id=bx_id,
                display_label=display_label,
            )
        )
    return selected


def load_exemplar_data(
    config: SharedPipelineConfig,
    selection_config: ExemplarSelectionConfig,
    *,
    load_master_structure_dict: bool = False,
    load_differential_dvh: bool = False,
    build_supporting_dvh_tables: bool = False,
    build_cumulative_dvh_table_from_voxels: bool = False,
) -> ExemplarLoadedData:
    common = load_common_data(
        config,
        load_master_structure_dict=load_master_structure_dict,
        load_differential_dvh=load_differential_dvh,
        load_cumulative_dvh_from_disk=False,
    )
    calculated_dvh_metrics_per_trial_df: pd.DataFrame | None = None
    cohort_global_dosimetry_dvh_metrics_df: pd.DataFrame | None = None
    if build_supporting_dvh_tables:
        (
            calculated_dvh_metrics_per_trial_df,
            cohort_global_dosimetry_dvh_metrics_df,
        ) = build_dvh_metric_tables(
            common.all_voxel_wise_dose_df,
            reference_dose_gy=config.cohort_reference_dose_gy,
        )

    all_cumulative_dvh_by_mc_trial_number_df: pd.DataFrame | None = None
    if build_cumulative_dvh_table_from_voxels:
        all_cumulative_dvh_by_mc_trial_number_df = build_cumulative_dvh_table(common.all_voxel_wise_dose_df)
    selected_exemplars = _resolve_selected_exemplars(
        common.cohort_biopsy_basic_spatial_features_df,
        selection_config,
    )
    return ExemplarLoadedData(
        common=common,
        calculated_dvh_metrics_per_trial_df=calculated_dvh_metrics_per_trial_df,
        cohort_global_dosimetry_dvh_metrics_df=cohort_global_dosimetry_dvh_metrics_df,
        all_cumulative_dvh_by_mc_trial_number_df=all_cumulative_dvh_by_mc_trial_number_df,
        selected_exemplars=selected_exemplars,
    )
