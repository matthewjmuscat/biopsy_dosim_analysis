from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
LEGACY_DVH_DIR = REPO_ROOT / "output_data" / "dvh_metrics"
REPORT_DIR = REPO_ROOT / "output_data_validation"


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    new_path: Path
    legacy_path: Path


@dataclass
class ComparisonResult:
    label: str
    new_path: str
    legacy_path: str
    status: str
    new_rows: int | None = None
    legacy_rows: int | None = None
    new_columns: int | None = None
    legacy_columns: int | None = None
    same_column_order: bool | None = None
    exact_equal: bool | None = None
    object_columns_equal: bool | None = None
    numeric_max_abs_diff: float | None = None
    numeric_worst_column: str | None = None
    mismatch_rows: int | None = None
    notes: str = ""


SPECS = [
    ComparisonSpec(
        label="QA per-trial DVH metrics vs legacy",
        new_path=REPO_ROOT / "output_data_QA" / "csv" / "qa_dvh_metrics_per_trial.csv",
        legacy_path=LEGACY_DVH_DIR / "Cohort: DVH metrics per trial.csv",
    ),
    ComparisonSpec(
        label="QA per-biopsy DVH summary vs legacy",
        new_path=REPO_ROOT / "output_data_QA" / "csv" / "qa_dvh_metrics_per_biopsy_summary.csv",
        legacy_path=LEGACY_DVH_DIR / "Cohort_DVH_metrics_stats_per_biopsy.csv",
    ),
    ComparisonSpec(
        label="Exemplars per-trial DVH metrics vs legacy",
        new_path=REPO_ROOT / "output_data_exemplars" / "csv" / "exemplar_dvh_metrics_per_trial.csv",
        legacy_path=LEGACY_DVH_DIR / "Cohort: DVH metrics per trial.csv",
    ),
    ComparisonSpec(
        label="Exemplars per-biopsy DVH summary vs legacy",
        new_path=REPO_ROOT / "output_data_exemplars" / "csv" / "exemplar_dvh_metrics_per_biopsy_summary.csv",
        legacy_path=LEGACY_DVH_DIR / "Cohort_DVH_metrics_stats_per_biopsy.csv",
    ),
]


def _compare_frames(spec: ComparisonSpec) -> ComparisonResult:
    result = ComparisonResult(
        label=spec.label,
        new_path=str(spec.new_path),
        legacy_path=str(spec.legacy_path),
        status="ok",
    )

    if not spec.new_path.exists():
        result.status = "missing_new"
        result.notes = "New/refactor CSV is missing."
        return result
    if not spec.legacy_path.exists():
        result.status = "missing_legacy"
        result.notes = "Legacy CSV is missing."
        return result

    new_df = pd.read_csv(spec.new_path)
    legacy_df = pd.read_csv(spec.legacy_path)
    result.new_rows = int(len(new_df))
    result.legacy_rows = int(len(legacy_df))
    result.new_columns = int(len(new_df.columns))
    result.legacy_columns = int(len(legacy_df.columns))
    result.same_column_order = list(new_df.columns) == list(legacy_df.columns)

    if set(new_df.columns) != set(legacy_df.columns):
        result.status = "column_mismatch"
        result.notes = (
            "Column sets differ. "
            f"only_new={sorted(set(new_df.columns) - set(legacy_df.columns))}; "
            f"only_legacy={sorted(set(legacy_df.columns) - set(new_df.columns))}"
        )
        return result

    ordered_columns = list(new_df.columns)
    sort_columns = [col for col in ordered_columns if new_df[col].dtype != object] + [
        col for col in ordered_columns if new_df[col].dtype == object
    ]

    new_sorted = new_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    legacy_sorted = legacy_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    result.exact_equal = bool(new_sorted.equals(legacy_sorted))
    if result.exact_equal:
        return result

    numeric_columns = [
        col
        for col in ordered_columns
        if pd.api.types.is_numeric_dtype(new_sorted[col]) and pd.api.types.is_numeric_dtype(legacy_sorted[col])
    ]
    object_columns = [col for col in ordered_columns if col not in numeric_columns]
    result.object_columns_equal = all(
        new_sorted[col].astype(str).equals(legacy_sorted[col].astype(str)) for col in object_columns
    )

    max_abs_diff = 0.0
    worst_column = None
    for col in numeric_columns:
        left = pd.to_numeric(new_sorted[col], errors="coerce")
        right = pd.to_numeric(legacy_sorted[col], errors="coerce")
        diff = (left - right).abs()
        col_max = diff.max(skipna=True)
        if pd.notna(col_max) and float(col_max) >= max_abs_diff:
            max_abs_diff = float(col_max)
            worst_column = col

    result.numeric_max_abs_diff = max_abs_diff
    result.numeric_worst_column = worst_column
    mismatch_rows = (new_sorted.astype(str) != legacy_sorted.astype(str)).any(axis=1)
    result.mismatch_rows = int(mismatch_rows.sum())
    result.status = "different"
    result.notes = "Dataframes differ after sorting."
    return result


def _write_report(results: list[ComparisonResult]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "refactor_csv_validation_summary.csv"
    md_path = REPORT_DIR / "refactor_csv_validation_summary.md"

    pd.DataFrame([asdict(result) for result in results]).to_csv(csv_path, index=False)

    lines = [
        "# Refactor CSV Validation Summary",
        "",
        f"Generated from `{Path(__file__).name}`.",
        "",
        "| Check | Status | Exact equal | Rows (new / legacy) | Cols (new / legacy) | Max abs diff | Notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.label,
                    result.status,
                    str(result.exact_equal),
                    f"{result.new_rows} / {result.legacy_rows}",
                    f"{result.new_columns} / {result.legacy_columns}",
                    "" if result.numeric_max_abs_diff is None else f"{result.numeric_max_abs_diff:g}",
                    result.notes.replace("|", "/"),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    results = [_compare_frames(spec) for spec in SPECS]
    csv_path, md_path = _write_report(results)

    for result in results:
        print(f"[{result.status}] {result.label}")
        print(f"  new={result.new_path}")
        print(f"  legacy={result.legacy_path}")
        if result.new_rows is not None:
            print(
                "  rows="
                f"{result.new_rows}/{result.legacy_rows}, "
                f"cols={result.new_columns}/{result.legacy_columns}, "
                f"exact_equal={result.exact_equal}"
            )
        if result.numeric_max_abs_diff is not None:
            print(
                "  numeric_max_abs_diff="
                f"{result.numeric_max_abs_diff:g} ({result.numeric_worst_column})"
            )
        if result.notes:
            print(f"  notes={result.notes}")
    print(f"[report] wrote {csv_path}")
    print(f"[report] wrote {md_path}")

    if any(result.status not in {"ok"} for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
