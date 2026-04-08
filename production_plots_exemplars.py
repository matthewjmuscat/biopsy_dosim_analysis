from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

import production_plots
from pipeline_shared_config import FigureExportConfig


def _alpha_code(index: int) -> str:
    if index < 0:
        raise ValueError("index must be >= 0")
    chars: list[str] = []
    n = int(index)
    while True:
        n, rem = divmod(n, 26)
        chars.append(chr(ord("A") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(chars))


def build_biopsy_heading_map(
    biopsy_pairs: Sequence[tuple[str, int]],
    *,
    explicit_map: Mapping[tuple[str, int], str] | None = None,
    prefix: str = "Biopsy",
) -> dict[tuple[str, int], str]:
    out = {k: str(v) for k, v in dict(explicit_map or {}).items()}
    used = set(out.values())
    next_idx = 0
    for pair in biopsy_pairs:
        if pair in out:
            continue
        while True:
            candidate = f"{prefix} {_alpha_code(next_idx)}"
            next_idx += 1
            if candidate not in used:
                break
        out[pair] = candidate
        used.add(candidate)
    return out


@contextmanager
def _font_rc(export_config: FigureExportConfig):
    with mpl.rc_context(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "axes.labelsize": export_config.axes_label_fontsize,
            "xtick.labelsize": export_config.tick_label_fontsize,
            "ytick.labelsize": export_config.tick_label_fontsize,
            "legend.fontsize": export_config.legend_fontsize,
            "axes.titlesize": export_config.title_fontsize,
        }
    ):
        yield


def _existing_export_paths(
    save_dir: str | Path,
    file_stem: str,
    export_config: FigureExportConfig,
    *,
    fallback_formats: Sequence[str] = (),
) -> list[Path]:
    candidate_formats = list(export_config.save_formats) + [fmt for fmt in fallback_formats if fmt not in export_config.save_formats]
    paths: list[Path] = []
    for fmt in candidate_formats:
        path = Path(save_dir) / f"{file_stem}.{str(fmt).lstrip('.')}"
        if path.exists():
            paths.append(path)
    return paths


def _matching_export_paths(
    save_dir: str | Path,
    file_stem_fragment: str,
    export_config: FigureExportConfig,
) -> list[Path]:
    paths: list[Path] = []
    for fmt in export_config.save_formats:
        pattern = f"*{file_stem_fragment}.{str(fmt).lstrip('.')}"
        paths.extend(sorted(Path(save_dir).glob(pattern)))
    return paths


def plot_exemplar_delta_lines(
    deltas_df,
    biopsies: Sequence[tuple[str, int]],
    save_dir: str | Path,
    fig_name: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    **kwargs: Any,
):
    kwargs.setdefault("axes_label_fontsize", export_config.axes_label_fontsize)
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize)
    if biopsy_label_map:
        ordered_labels = [biopsy_label_map.get(pair, f"{pair[0]}, Bx {pair[1]}") for pair in biopsies]
        kwargs.setdefault("title", " vs ".join(ordered_labels))
    with _font_rc(export_config):
        return production_plots.plot_biopsy_deltas_line_multi(
            deltas_df,
            biopsies=biopsies,
            save_dir=save_dir,
            fig_name=fig_name,
            save_formats=export_config.save_formats,
            **kwargs,
        )


def plot_exemplar_voxel_dualboxes(
    deltas_df,
    biopsies: Sequence[tuple[str, int]],
    output_dir: str | Path,
    plot_name_base: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    biopsy_label_map: Mapping[tuple[str, int], str] | None = None,
    **kwargs: Any,
):
    kwargs.setdefault("axes_label_fontsize", export_config.axes_label_fontsize)
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("biopsy_label_map", biopsy_label_map)
    with _font_rc(export_config):
        return production_plots.plot_voxel_dualboxes_by_biopsy_lanes(
            deltas_df,
            biopsies=biopsies,
            output_dir=output_dir,
            plot_name_base=plot_name_base,
            save_formats=export_config.save_formats,
            **kwargs,
        )


def plot_exemplar_length_scale_boxes(
    df,
    save_dir: str | Path,
    file_name: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    **kwargs: Any,
):
    kwargs.setdefault("dpi", export_config.dpi)
    kwargs.setdefault("x_col", "length_scale")
    kwargs.setdefault("y_col", "dose_diff_abs")
    kwargs.setdefault("axis_label_font_size", export_config.axes_label_fontsize)
    kwargs.setdefault("tick_label_font_size", export_config.tick_label_fontsize)
    with _font_rc(export_config):
        result = production_plots.plot_dose_vs_length_with_summary_mutlibox(
            df,
            save_dir=save_dir,
            file_name=file_name,
            save_formats=export_config.save_formats,
            **kwargs,
        )
    if result is None:
        return _existing_export_paths(save_dir, file_name, export_config, fallback_formats=("png", "svg"))
    return result


def plot_exemplar_voxel_pair_heatmap(
    upper_df,
    lower_df,
    save_dir: str | Path,
    save_name_base: str,
    *,
    export_config: FigureExportConfig = FigureExportConfig(),
    **kwargs: Any,
):
    kwargs.setdefault("patient_id_col", "Patient ID")
    kwargs.setdefault("bx_index_col", "Bx index")
    kwargs.setdefault("bx_id_col", "Bx ID")
    kwargs.setdefault("tick_label_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("axis_label_fontsize", export_config.axes_label_fontsize)
    kwargs.setdefault("cbar_tick_fontsize", export_config.tick_label_fontsize)
    kwargs.setdefault("cbar_label_fontsize", export_config.legend_fontsize)
    with _font_rc(export_config):
        result = production_plots.plot_diff_stats_heatmap_upper_lower(
            upper_df=upper_df,
            lower_df=lower_df,
            save_dir=save_dir,
            save_name_base=save_name_base,
            save_formats=export_config.save_formats,
            **kwargs,
        )
    if result is None:
        return _matching_export_paths(save_dir, save_name_base, export_config)
    return result
