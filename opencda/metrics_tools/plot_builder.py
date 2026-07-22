"""Plot generation for metric reports."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.report_models import EntityReport, GroupReport, MetricReport, ModuleReport

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.plot_builder")

_PYPLOT: Any | None = None
_PYPLOT_LOAD_ATTEMPTED = False


@dataclass(frozen=True, slots=True)
class MetricPlotStyle:
    """Centralized visual style for metric time-series plots."""

    image_format: str = "png"
    dpi: int = 200
    figsize: tuple[float, float] = (15.0, 6.0)

    # --- Main line ---
    line_color: str = "#2D6A4F"  # deep green
    line_width: float = 2.0
    marker: str | None = None

    # --- Typography ---
    font_family: str = "serif"
    title_fontsize: int = 24
    label_fontsize: int = 20
    tick_fontsize: int = 17
    axis_text_color: str = "#1B5E20"

    # --- Grid ---
    grid_alpha: float = 0.55
    grid_linestyle: str = "--"
    grid_color: str = "#B7E4C7"  # light green

    # --- Background ---
    background_color: str = "#FFFFFF"
    axes_facecolor: str = "#F4FBF3"  # barely-there green tint

    # --- Spines ---
    hide_top_spine: bool = True
    hide_right_spine: bool = True

    # --- Minor ticks ---
    minor_ticks: bool = True

    # --- Summary statistics ---
    show_mean: bool = True
    show_min_max: bool = True
    stats_box_fontsize: int = 17
    stats_box_text_color: str = "#1B5E20"
    stats_box_text_alignment: str = "left"
    stats_box_x: float = 0.5
    stats_box_y: float = 0.035
    stats_bottom_margin: float = 0.12
    stats_box_facecolor: str = "#FFFFFF"
    stats_box_edgecolor: str = "#2E8B6E"
    stats_box_alpha: float = 0.92
    stats_box_linewidth: float = 1.0
    stats_box_style: str = "round,pad=0.45"

    value_precision: int = 3


DEFAULT_METRIC_PLOT_STYLE = MetricPlotStyle()


class MetricPlotBuilder:
    """Build simple time-series plots for metric report series."""

    def __init__(self, style: MetricPlotStyle = DEFAULT_METRIC_PLOT_STYLE) -> None:
        self.style = style

    def build_module_plots(self, module_report: ModuleReport, output_dir: str | Path) -> tuple[Path, ...]:
        """Build plots for every entity in a module report."""
        module_dir = Path(output_dir) / _safe_path_part(module_report.module)
        output_paths: list[Path] = []

        for entity_report in module_report.entities:
            entity_dir = module_dir / _safe_path_part(str(entity_report.info.entity_id))
            output_paths.extend(self._build_entity_plots(entity_report, entity_dir))

        return tuple(output_paths)

    def build_group_plots(self, group_report: GroupReport, output_dir: str | Path, module: str) -> tuple[Path, ...]:  # noqa: DC04
        """Build plots for every entity in a grouped report."""
        group_dir = Path(output_dir) / _safe_path_part(module) / _safe_path_part(str(group_report.group_id))
        output_paths: list[Path] = []

        for entity_report in group_report.entities:
            entity_dir = group_dir / _safe_path_part(str(entity_report.info.entity_id))
            output_paths.extend(self._build_entity_plots(entity_report, entity_dir))

        return tuple(output_paths)

    def _build_entity_plots(self, entity_report: EntityReport, output_dir: Path) -> tuple[Path, ...]:
        output_paths: list[Path] = []
        for metric_report in entity_report.metrics:
            output_paths.extend(self._build_metric_plots(metric_report, entity_report, output_dir))
        return tuple(output_paths)

    def _build_metric_plots(self, metric_report: MetricReport, entity_report: EntityReport, output_dir: Path) -> tuple[Path, ...]:
        output_paths: list[Path] = []
        for series in metric_report.series:
            if not series.samples:
                continue

            filename = f"{_safe_path_part(metric_report.metric_name)}__{_safe_path_part(series.name)}.{self.style.image_format}"
            title = f"{entity_report.info.module}/{entity_report.info.entity_id}/{series.name}"
            output_path = output_dir / filename
            if self._build_series_plot(series, output_path, title):
                output_paths.append(output_path)

        return tuple(output_paths)

    def _build_series_plot(self, series: MetricSeries, output_path: Path, title: str) -> bool:
        pyplot = _load_pyplot()
        if pyplot is None:
            return False

        ticks = [sample.tick for sample in series.samples]
        values = [sample.value for sample in series.samples]

        import matplotlib as mpl

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply font globally for this figure
        with mpl.rc_context({"font.family": self.style.font_family}):
            figure, axis = pyplot.subplots(figsize=self.style.figsize)
            figure.patch.set_facecolor(self.style.background_color)
            axis.set_facecolor(self.style.axes_facecolor)

            # Spines
            axis.spines["top"].set_visible(not self.style.hide_top_spine)
            axis.spines["right"].set_visible(not self.style.hide_right_spine)
            axis.spines["left"].set_color(self.style.axis_text_color)
            axis.spines["bottom"].set_color(self.style.axis_text_color)
            axis.spines["left"].set_linewidth(1.1)
            axis.spines["bottom"].set_linewidth(1.1)

            axis.plot(
                ticks,
                values,
                color=self.style.line_color,
                linewidth=self.style.line_width,
                marker=self.style.marker,
            )

            axis.set_title(title, fontsize=self.style.title_fontsize, color=self.style.axis_text_color, pad=14)
            axis.set_xlabel("Tick", fontsize=self.style.label_fontsize, color=self.style.axis_text_color)
            axis.set_ylabel(series.name, fontsize=self.style.label_fontsize, color=self.style.axis_text_color)

            # Ticks
            axis.tick_params(
                axis="both",
                labelsize=self.style.tick_fontsize,
                colors=self.style.axis_text_color,
                direction="out",
                length=5,
                width=1.0,
            )
            if self.style.minor_ticks:
                axis.minorticks_on()
                axis.tick_params(axis="both", which="minor", length=3, width=0.7, colors=self.style.axis_text_color)

            axis.grid(
                True,
                which="major",
                alpha=self.style.grid_alpha,
                linestyle=self.style.grid_linestyle,
                color=self.style.grid_color,
            )
            axis.set_axisbelow(True)

            stats_text = self._build_summary_statistics_text(values)
            figure.tight_layout(rect=(0.0, self.style.stats_bottom_margin if stats_text else 0.0, 1.0, 1.0))
            if stats_text:
                self._draw_summary_statistics(figure, stats_text)
            figure.savefig(output_path, dpi=self.style.dpi)
            pyplot.close(figure)

        return True

    def _build_summary_statistics_text(self, values: list[float]) -> str | None:
        if not values:
            return None

        summary_parts: list[str] = []

        if self.style.show_min_max:
            summary_parts.append(f"max: {_format_value(max(values), self.style.value_precision)}")
            summary_parts.append(f"min: {_format_value(min(values), self.style.value_precision)}")

        if self.style.show_mean:
            mean_value = sum(values) / len(values)
            summary_parts.append(f"mean: {_format_value(mean_value, self.style.value_precision)}")

        if not summary_parts:
            return None

        return "    ".join(summary_parts)

    def _draw_summary_statistics(self, figure: Any, stats_text: str) -> None:
        figure.text(
            self.style.stats_box_x,
            self.style.stats_box_y,
            stats_text,
            ha="center",
            va="bottom",
            fontsize=self.style.stats_box_fontsize,
            color=self.style.stats_box_text_color,
            multialignment=self.style.stats_box_text_alignment,
            bbox={
                "boxstyle": self.style.stats_box_style,
                "facecolor": self.style.stats_box_facecolor,
                "edgecolor": self.style.stats_box_edgecolor,
                "alpha": self.style.stats_box_alpha,
                "linewidth": self.style.stats_box_linewidth,
            },
        )


def _load_pyplot() -> Any | None:
    global _PYPLOT, _PYPLOT_LOAD_ATTEMPTED

    if _PYPLOT_LOAD_ATTEMPTED:
        return _PYPLOT

    _PYPLOT_LOAD_ATTEMPTED = True
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as pyplot
    except Exception as exc:
        logger.warning("Metric plot generation skipped: matplotlib is not available or failed to initialize: %s", exc)
        return None

    _PYPLOT = pyplot
    return _PYPLOT


def _safe_path_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "unknown"


def _format_value(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"
