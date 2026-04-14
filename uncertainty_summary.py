from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from load_data_shared import (
    _filter_frame_by_simulation,
    _load_and_concat_tables,
    _load_uncertainties_frame,
    resolve_pipeline_paths,
)
from load_data_shared import CommonLoadedData
from pipeline_shared_config import SharedPipelineConfig


SUMMARY_PERCENTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class UncertaintySummaryOutputs:
    configured_biopsy_parameter_summary_df: pd.DataFrame
    configured_by_structure_summary_df: pd.DataFrame
    realized_biopsy_parameter_summary_df: pd.DataFrame
    realized_per_biopsy_summary_df: pd.DataFrame
    paper_relevant_summary_df: pd.DataFrame
    markdown_summary: str


CONFIGURED_PARAMETER_SPECS: tuple[dict[str, str], ...] = (
    {
        "column": "mu (X)",
        "source_family": "rigid_translation",
        "parameter": "translation_x",
        "parameter_kind": "configured_mean",
        "axis": "X",
        "unit": "mm",
    },
    {
        "column": "mu (Y)",
        "source_family": "rigid_translation",
        "parameter": "translation_y",
        "parameter_kind": "configured_mean",
        "axis": "Y",
        "unit": "mm",
    },
    {
        "column": "mu (Z)",
        "source_family": "rigid_translation",
        "parameter": "translation_z",
        "parameter_kind": "configured_mean",
        "axis": "Z",
        "unit": "mm",
    },
    {
        "column": "sigma (X)",
        "source_family": "rigid_translation",
        "parameter": "translation_x",
        "parameter_kind": "configured_sigma",
        "axis": "X",
        "unit": "mm",
    },
    {
        "column": "sigma (Y)",
        "source_family": "rigid_translation",
        "parameter": "translation_y",
        "parameter_kind": "configured_sigma",
        "axis": "Y",
        "unit": "mm",
    },
    {
        "column": "sigma (Z)",
        "source_family": "rigid_translation",
        "parameter": "translation_z",
        "parameter_kind": "configured_sigma",
        "axis": "Z",
        "unit": "mm",
    },
    {
        "column": "Dilations mu (XY)",
        "source_family": "dilation",
        "parameter": "dilation_xy",
        "parameter_kind": "configured_mean",
        "axis": "XY",
        "unit": "native_scale_units",
    },
    {
        "column": "Dilations mu (Z)",
        "source_family": "dilation",
        "parameter": "dilation_z",
        "parameter_kind": "configured_mean",
        "axis": "Z",
        "unit": "native_scale_units",
    },
    {
        "column": "Dilations sigma (XY)",
        "source_family": "dilation",
        "parameter": "dilation_xy",
        "parameter_kind": "configured_sigma",
        "axis": "XY",
        "unit": "native_scale_units",
    },
    {
        "column": "Dilations sigma (Z)",
        "source_family": "dilation",
        "parameter": "dilation_z",
        "parameter_kind": "configured_sigma",
        "axis": "Z",
        "unit": "native_scale_units",
    },
    {
        "column": "Rotations mu (X)",
        "source_family": "rotation",
        "parameter": "rotation_x",
        "parameter_kind": "configured_mean",
        "axis": "X",
        "unit": "native_rotation_units",
    },
    {
        "column": "Rotations mu (Y)",
        "source_family": "rotation",
        "parameter": "rotation_y",
        "parameter_kind": "configured_mean",
        "axis": "Y",
        "unit": "native_rotation_units",
    },
    {
        "column": "Rotations mu (Z)",
        "source_family": "rotation",
        "parameter": "rotation_z",
        "parameter_kind": "configured_mean",
        "axis": "Z",
        "unit": "native_rotation_units",
    },
    {
        "column": "Rotations sigma (X)",
        "source_family": "rotation",
        "parameter": "rotation_x",
        "parameter_kind": "configured_sigma",
        "axis": "X",
        "unit": "native_rotation_units",
    },
    {
        "column": "Rotations sigma (Y)",
        "source_family": "rotation",
        "parameter": "rotation_y",
        "parameter_kind": "configured_sigma",
        "axis": "Y",
        "unit": "native_rotation_units",
    },
    {
        "column": "Rotations sigma (Z)",
        "source_family": "rotation",
        "parameter": "rotation_z",
        "parameter_kind": "configured_sigma",
        "axis": "Z",
        "unit": "native_rotation_units",
    },
)


REALIZED_PARAMETER_SPECS: tuple[dict[str, str], ...] = (
    {
        "column": "Shift (X)",
        "source_family": "rigid_translation",
        "parameter": "translation_x",
        "parameter_kind": "realized_sample",
        "axis": "X",
        "unit": "mm",
    },
    {
        "column": "Shift (Y)",
        "source_family": "rigid_translation",
        "parameter": "translation_y",
        "parameter_kind": "realized_sample",
        "axis": "Y",
        "unit": "mm",
    },
    {
        "column": "Shift (Z)",
        "source_family": "rigid_translation",
        "parameter": "translation_z",
        "parameter_kind": "realized_sample",
        "axis": "Z",
        "unit": "mm",
    },
    {
        "column": "Shift (z_needle)",
        "source_family": "axial_tissue_deficit_offset",
        "parameter": "axial_tissue_deficit_offset",
        "parameter_kind": "realized_sample",
        "axis": "z_needle",
        "unit": "mm",
    },
    {
        "column": "Dilation (XY)",
        "source_family": "dilation",
        "parameter": "dilation_xy",
        "parameter_kind": "realized_sample",
        "axis": "XY",
        "unit": "native_scale_units",
    },
    {
        "column": "Dilation (Z)",
        "source_family": "dilation",
        "parameter": "dilation_z",
        "parameter_kind": "realized_sample",
        "axis": "Z",
        "unit": "native_scale_units",
    },
    {
        "column": "Rotation (X)",
        "source_family": "rotation",
        "parameter": "rotation_x",
        "parameter_kind": "realized_sample",
        "axis": "X",
        "unit": "native_rotation_units",
    },
    {
        "column": "Rotation (Y)",
        "source_family": "rotation",
        "parameter": "rotation_y",
        "parameter_kind": "realized_sample",
        "axis": "Y",
        "unit": "native_rotation_units",
    },
    {
        "column": "Rotation (Z)",
        "source_family": "rotation",
        "parameter": "rotation_z",
        "parameter_kind": "realized_sample",
        "axis": "Z",
        "unit": "native_rotation_units",
    },
)


def _normalise_group_key(group_key) -> tuple:
    if isinstance(group_key, tuple):
        return group_key
    return (group_key,)


def _series_summary(series: pd.Series) -> dict[str, float | int | bool]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "max": np.nan,
            "mean_abs": np.nan,
            "max_abs": np.nan,
            "is_all_zero": False,
        }
    qs = values.quantile(list(SUMMARY_PERCENTILES))
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "q05": float(qs.loc[0.05]),
        "q25": float(qs.loc[0.25]),
        "median": float(qs.loc[0.50]),
        "q75": float(qs.loc[0.75]),
        "q95": float(qs.loc[0.95]),
        "max": float(values.max()),
        "mean_abs": float(values.abs().mean()),
        "max_abs": float(values.abs().max()),
        "is_all_zero": bool(np.allclose(values.to_numpy(), 0.0)),
    }


def _summarize_parameter_specs(
    df: pd.DataFrame,
    specs: Sequence[Mapping[str, str]],
    *,
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    group_cols = list(group_cols or [])
    rows: list[dict[str, object]] = []
    available_specs = [spec for spec in specs if spec["column"] in df.columns]
    if not available_specs:
        return pd.DataFrame()

    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        grouped = [((), df)]

    for group_key, group_df in grouped:
        group_key = _normalise_group_key(group_key)
        group_meta = {col: group_key[idx] for idx, col in enumerate(group_cols)}
        for spec in available_specs:
            row = dict(group_meta)
            row.update(spec)
            row.update(_series_summary(group_df[spec["column"]]))
            rows.append(row)
    return pd.DataFrame(rows)


def build_uncertainty_summary_outputs(
    common: CommonLoadedData,
    *,
    biopsy_structure_type: str = "Bx ref",
) -> UncertaintySummaryOutputs:
    return build_uncertainty_summary_outputs_from_frames(
        uncertainties_df=common.uncertainties_df,
        transform_df=common.all_mc_structure_transformation_df,
        biopsy_structure_type=biopsy_structure_type,
    )


def build_uncertainty_summary_outputs_from_frames(
    *,
    uncertainties_df: pd.DataFrame,
    transform_df: pd.DataFrame,
    biopsy_structure_type: str = "Bx ref",
) -> UncertaintySummaryOutputs:
    uncertainties_df = uncertainties_df.copy()
    transform_df = transform_df.copy()

    configured_by_structure_summary_df = _summarize_parameter_specs(
        uncertainties_df,
        CONFIGURED_PARAMETER_SPECS,
        group_cols=["Structure type", "Frame of reference"],
    )

    configured_biopsy_df = uncertainties_df.loc[
        uncertainties_df["Structure type"].astype(str) == biopsy_structure_type
    ].copy()
    realized_biopsy_df = transform_df.loc[
        transform_df["Structure type"].astype(str) == biopsy_structure_type
    ].copy()
    if (
        not configured_biopsy_df.empty
        and not realized_biopsy_df.empty
        and {"Patient UID", "Structure ID"}.issubset(configured_biopsy_df.columns)
        and {"Patient ID", "Structure ID"}.issubset(realized_biopsy_df.columns)
    ):
        valid_pairs = set(
            zip(
                realized_biopsy_df["Patient ID"].astype(str),
                realized_biopsy_df["Structure ID"].astype(str),
            )
        )
        configured_biopsy_df = configured_biopsy_df.loc[
            configured_biopsy_df.apply(
                lambda row: (str(row["Patient UID"]), str(row["Structure ID"])) in valid_pairs,
                axis=1,
            )
        ].copy()
    configured_biopsy_parameter_summary_df = _summarize_parameter_specs(
        configured_biopsy_df,
        CONFIGURED_PARAMETER_SPECS,
    )
    realized_biopsy_parameter_summary_df = _summarize_parameter_specs(
        realized_biopsy_df,
        REALIZED_PARAMETER_SPECS,
    )
    realized_per_biopsy_summary_df = _summarize_parameter_specs(
        realized_biopsy_df,
        REALIZED_PARAMETER_SPECS,
        group_cols=["Patient ID", "Structure ID", "Structure index"],
    )

    paper_relevant_columns = ["translation_x", "translation_y", "translation_z", "axial_tissue_deficit_offset"]
    paper_relevant_summary_df = pd.concat(
        [
            configured_biopsy_parameter_summary_df.loc[
                configured_biopsy_parameter_summary_df["parameter"].isin(paper_relevant_columns)
            ].assign(summary_scope="configured_biopsy_distribution_parameter"),
            realized_biopsy_parameter_summary_df.loc[
                realized_biopsy_parameter_summary_df["parameter"].isin(paper_relevant_columns)
            ].assign(summary_scope="realized_biopsy_trial_samples"),
        ],
        ignore_index=True,
    )

    def _fmt_line(df: pd.DataFrame, parameter: str, parameter_kind: str) -> str | None:
        sub = df.loc[
            (df["parameter"] == parameter) & (df["parameter_kind"] == parameter_kind)
        ]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return (
            f"- `{parameter}` ({parameter_kind}, {row['unit']}): "
            f"mean={row['mean']:.4g}, median={row['median']:.4g}, "
            f"q05={row['q05']:.4g}, q95={row['q95']:.4g}, max={row['max']:.4g}, "
            f"mean_abs={row['mean_abs']:.4g}, max_abs={row['max_abs']:.4g}, "
            f"all_zero={bool(row['is_all_zero'])}"
        )

    md_lines = [
        "# Uncertainty Source Summary",
        "",
        f"- Biopsy structure filter: `{biopsy_structure_type}`",
        "",
        "## Configured Biopsy Uncertainty Parameters",
    ]
    for parameter in ("translation_x", "translation_y", "translation_z"):
        line = _fmt_line(configured_biopsy_parameter_summary_df, parameter, "configured_sigma")
        if line:
            md_lines.append(line)
    for parameter in ("rotation_x", "rotation_y", "rotation_z", "dilation_xy", "dilation_z"):
        line = _fmt_line(configured_biopsy_parameter_summary_df, parameter, "configured_sigma")
        if line:
            md_lines.append(line)
    md_lines.extend(
        [
            "",
            "## Realized Biopsy Trial Samples",
        ]
    )
    for parameter in (
        "translation_x",
        "translation_y",
        "translation_z",
        "axial_tissue_deficit_offset",
        "rotation_x",
        "rotation_y",
        "rotation_z",
        "dilation_xy",
        "dilation_z",
    ):
        line = _fmt_line(realized_biopsy_parameter_summary_df, parameter, "realized_sample")
        if line:
            md_lines.append(line)

    return UncertaintySummaryOutputs(
        configured_biopsy_parameter_summary_df=configured_biopsy_parameter_summary_df,
        configured_by_structure_summary_df=configured_by_structure_summary_df,
        realized_biopsy_parameter_summary_df=realized_biopsy_parameter_summary_df,
        realized_per_biopsy_summary_df=realized_per_biopsy_summary_df,
        paper_relevant_summary_df=paper_relevant_summary_df,
        markdown_summary="\n".join(md_lines) + "\n",
    )


def load_uncertainty_source_frames(
    config: SharedPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = resolve_pipeline_paths(config)
    transform_df = _load_and_concat_tables(
        paths.mc_sim_results_path,
        ["All MC structure transformation values.csv"],
        parquet=False,
        verbose=config.verbose_bulk_file_loading,
    )
    transform_df = _filter_frame_by_simulation(
        transform_df,
        "all_mc_structure_transformation_df",
        config,
    )
    uncertainties_df = _load_uncertainties_frame(paths.main_output_path)
    return uncertainties_df, transform_df


def write_uncertainty_summary_outputs(
    csv_root: str | Path,
    manifest_root: str | Path,
    outputs: UncertaintySummaryOutputs,
) -> dict[str, Path]:
    csv_root = Path(csv_root)
    manifest_root = Path(manifest_root)
    csv_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "configured_biopsy": csv_root / "configured_biopsy_uncertainty_parameter_summary.csv",
        "configured_by_structure": csv_root / "configured_uncertainty_parameter_summary_by_structure.csv",
        "realized_biopsy": csv_root / "realized_biopsy_uncertainty_trial_summary.csv",
        "realized_per_biopsy": csv_root / "realized_biopsy_uncertainty_trial_summary_per_biopsy.csv",
        "paper_relevant": csv_root / "paper_relevant_uncertainty_summary.csv",
        "markdown_summary": manifest_root / "uncertainty_source_summary.md",
    }

    outputs.configured_biopsy_parameter_summary_df.to_csv(paths["configured_biopsy"], index=False)
    outputs.configured_by_structure_summary_df.to_csv(paths["configured_by_structure"], index=False)
    outputs.realized_biopsy_parameter_summary_df.to_csv(paths["realized_biopsy"], index=False)
    outputs.realized_per_biopsy_summary_df.to_csv(paths["realized_per_biopsy"], index=False)
    outputs.paper_relevant_summary_df.to_csv(paths["paper_relevant"], index=False)
    paths["markdown_summary"].write_text(outputs.markdown_summary, encoding="utf-8")
    return paths


def load_and_write_uncertainty_summary_outputs(
    config: SharedPipelineConfig,
    *,
    csv_root: str | Path,
    manifest_root: str | Path,
    biopsy_structure_type: str = "Bx ref",
) -> dict[str, Path]:
    uncertainties_df, transform_df = load_uncertainty_source_frames(config)
    outputs = build_uncertainty_summary_outputs_from_frames(
        uncertainties_df=uncertainties_df,
        transform_df=transform_df,
        biopsy_structure_type=biopsy_structure_type,
    )
    return write_uncertainty_summary_outputs(
        csv_root=csv_root,
        manifest_root=manifest_root,
        outputs=outputs,
    )
