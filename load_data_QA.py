from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from load_data_shared import CommonLoadedData, build_dvh_metric_tables, load_common_data
from pipeline_shared_config import SharedPipelineConfig


@dataclass
class QALoadedData:
    common: CommonLoadedData
    calculated_dvh_metrics_per_trial_df: pd.DataFrame
    cohort_global_dosimetry_dvh_metrics_df: pd.DataFrame


def load_qa_data(
    config: SharedPipelineConfig,
    *,
    load_master_structure_dict: bool = False,
    load_differential_dvh: bool = False,
) -> QALoadedData:
    common = load_common_data(
        config,
        load_master_structure_dict=load_master_structure_dict,
        load_differential_dvh=load_differential_dvh,
        load_cumulative_dvh_from_disk=False,
    )
    calculated_dvh_metrics_per_trial_df, cohort_global_dosimetry_dvh_metrics_df = build_dvh_metric_tables(
        common.all_voxel_wise_dose_df,
        reference_dose_gy=config.cohort_reference_dose_gy,
    )
    return QALoadedData(
        common=common,
        calculated_dvh_metrics_per_trial_df=calculated_dvh_metrics_per_trial_df,
        cohort_global_dosimetry_dvh_metrics_df=cohort_global_dosimetry_dvh_metrics_df,
    )
