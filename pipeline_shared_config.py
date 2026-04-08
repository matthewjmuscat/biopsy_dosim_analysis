from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_MAIN_OUTPUT_PATH = Path(
    "/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Jan-04-2026 Time-11,55,49"
)


def resolve_existing_output_dir(requested_path: Path) -> Path:
    """
    Resolve a user-facing output path to an on-disk directory.

    This supports the common case where a historical hard-coded path later gained
    a descriptive suffix but kept the same leading timestamped folder stem.
    """
    if requested_path.is_dir():
        return requested_path

    parent = requested_path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    candidates = sorted(
        path
        for path in parent.glob(f"{requested_path.name}*")
        if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve output directory '{requested_path}'. No matching folder prefix was found."
        )
    if len(candidates) > 1:
        raise FileNotFoundError(
            "Ambiguous output directory prefix "
            f"'{requested_path}'. Matching candidates: {[str(path) for path in candidates]}"
        )
    return candidates[0]


@dataclass(frozen=True)
class SimulationFilterConfig:
    simulated_type_filter: str | Sequence[str] | None = "Real"
    simulated_bool_filter: bool | None = None


@dataclass(frozen=True)
class FigureExportConfig:
    save_formats: tuple[str, ...] = ("pdf", "svg")
    dpi: int = 300
    axes_label_fontsize: int = 16
    tick_label_fontsize: int = 14
    legend_fontsize: int = 12
    annotation_fontsize: int = 12
    title_fontsize: int = 17


@dataclass(frozen=True)
class SharedPipelineConfig:
    main_output_path: Path = DEFAULT_MAIN_OUTPUT_PATH
    distances_output_path: Path | None = None
    output_root: Path = REPO_ROOT / "output_data_refactor"
    bx_ref: str = "Bx ref"
    cohort_reference_dose_gy: float = 13.5
    verbose_bulk_file_loading: bool = False
    sim_filter: SimulationFilterConfig = field(default_factory=SimulationFilterConfig)

    @property
    def resolved_distances_output_path(self) -> Path:
        return self.main_output_path if self.distances_output_path is None else self.distances_output_path


@dataclass(frozen=True)
class ExemplarSelectionConfig:
    biopsy_pairs: tuple[tuple[str, int], ...] = (("184 (F2)", 1), ("184 (F2)", 2))
    display_label_map: Mapping[tuple[str, int], str] = field(
        default_factory=lambda: {
            ("184 (F2)", 1): "Biopsy A",
            ("184 (F2)", 2): "Biopsy B",
        }
    )


@dataclass(frozen=True)
class QAOutputConfig:
    output_root: Path = REPO_ROOT / "output_data_QA"
    figures_subdir: str = "figures"
    csv_subdir: str = "csv"
    manifest_subdir: str = "manifests"


@dataclass(frozen=True)
class ExemplarsOutputConfig:
    output_root: Path = REPO_ROOT / "output_data_exemplars"
    figures_subdir: str = "figures"
    csv_subdir: str = "csv"
    manifest_subdir: str = "manifests"
