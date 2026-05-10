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
    line_color: str = "#004F54"   # deep teal
    line_width: float = 2.0
    marker: str | None = None

    # --- Typography ---
    font_family: str = "serif"
    title_fontsize: int = 18
    label_fontsize: int = 15
    tick_fontsize: int = 13

    # --- Grid ---
    grid_alpha: float = 0.55
    grid_linestyle: str = "--"
    grid_color: str = "#B2DDD9"   # light teal

    # --- Background ---
    background_color: str = "#FFFFFF"
    axes_facecolor: str = "#F2FDFB"   # barely-there teal tint

    # --- Spines ---
    hide_top_spine: bool = True
    hide_right_spine: bool = True

    # --- Minor ticks ---
    minor_ticks: bool = True

    # --- Mean line ---
    show_mean: bool = True
    mean_line_color: str = "#5CDB95"   # bright green accent
    mean_line_width: float = 1.6
    mean_line_style: str = "--"
    mean_label_fontsize: int = 13

    # --- Min / max markers ---
    show_min_max: bool = True
    extreme_marker_color: str = "#2E8B6E"   # mid-green, readable on light bg
    extreme_marker_size: float = 70.0
    extreme_annotation_fontsize: int = 13

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

    def build_group_plots(self, group_report: GroupReport, output_dir: str | Path, module: str) -> tuple[Path, ...]:
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
            axis.spines["left"].set_color("#004F54")
            axis.spines["bottom"].set_color("#004F54")
            axis.spines["left"].set_linewidth(1.1)
            axis.spines["bottom"].set_linewidth(1.1)

            axis.plot(
                ticks,
                values,
                color=self.style.line_color,
                linewidth=self.style.line_width,
                marker=self.style.marker,
            )
            self._draw_summary_statistics(axis, ticks, values)

            axis.set_title(title, fontsize=self.style.title_fontsize, color="#004F54", pad=14)
            axis.set_xlabel("Tick", fontsize=self.style.label_fontsize, color="#004F54")
            axis.set_ylabel(series.name, fontsize=self.style.label_fontsize, color="#004F54")

            # Ticks
            axis.tick_params(
                axis="both",
                labelsize=self.style.tick_fontsize,
                colors="#004F54",
                direction="out",
                length=5,
                width=1.0,
            )
            if self.style.minor_ticks:
                axis.minorticks_on()
                axis.tick_params(axis="both", which="minor", length=3, width=0.7, colors="#004F54")

            axis.grid(
                True,
                which="major",
                alpha=self.style.grid_alpha,
                linestyle=self.style.grid_linestyle,
                color=self.style.grid_color,
            )
            axis.set_axisbelow(True)

            figure.tight_layout()
            figure.savefig(output_path, dpi=self.style.dpi)
            pyplot.close(figure)

        return True

    def _draw_summary_statistics(self, axis: Any, ticks: list[int], values: list[float]) -> None:
        if not values:
            return

        if self.style.show_mean:
            mean_value = sum(values) / len(values)
            axis.axhline(
                mean_value,
                color=self.style.mean_line_color,
                linewidth=self.style.mean_line_width,
                linestyle=self.style.mean_line_style,
            )
            axis.annotate(
                f"mean={_format_value(mean_value, self.style.value_precision)}",
                xy=(ticks[-1], mean_value),
                xytext=(-8, 8),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=self.style.mean_label_fontsize,
                color=self.style.mean_line_color,
            )

        if self.style.show_min_max:
            self._annotate_extreme(axis, ticks, values, max(values), "max")
            min_value = min(values)
            if min_value != max(values):
                self._annotate_extreme(axis, ticks, values, min_value, "min")

    def _annotate_extreme(self, axis: Any, ticks: list[int], values: list[float], extreme_value: float, label: str) -> None:
        index = values.index(extreme_value)
        tick = ticks[index]
        axis.scatter(
            [tick],
            [extreme_value],
            color=self.style.extreme_marker_color,
            s=self.style.extreme_marker_size,
            zorder=3,
        )
        axis.annotate(
            f"{label}={_format_value(extreme_value, self.style.value_precision)}",
            xy=(tick, extreme_value),
            xytext=(8, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=self.style.extreme_annotation_fontsize,
            color=self.style.extreme_marker_color,
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