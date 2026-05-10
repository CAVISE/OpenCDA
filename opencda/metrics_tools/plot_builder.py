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
    dpi: int = 140
    figsize: tuple[float, float] = (10.0, 4.0)
    line_color: str = "#004F54"
    line_width: float = 1.8
    marker: str | None = None
    title_fontsize: int = 12
    label_fontsize: int = 10
    tick_fontsize: int = 9
    grid_alpha: float = 0.25
    grid_linestyle: str = "--"
    background_color: str = "#EDFFF6"
    axes_facecolor: str = "#11222A"


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

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = pyplot.subplots(figsize=self.style.figsize)
        figure.patch.set_facecolor(self.style.background_color)
        axis.set_facecolor(self.style.axes_facecolor)
        axis.plot(
            ticks,
            values,
            color=self.style.line_color,
            linewidth=self.style.line_width,
            marker=self.style.marker,
        )
        axis.set_title(title, fontsize=self.style.title_fontsize)
        axis.set_xlabel("Tick", fontsize=self.style.label_fontsize)
        axis.set_ylabel(series.name, fontsize=self.style.label_fontsize)
        axis.tick_params(axis="both", labelsize=self.style.tick_fontsize)
        axis.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.style.dpi)
        pyplot.close(figure)
        return True


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
