"""
Evaluation manager.
"""

import os
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from opencda.core.common.cav_world import CavWorld
from opencda.metrics_tools.plot_builder import MetricPlotBuilder
from opencda.metrics_tools.report_models import EntityReport, GroupReport, ModuleReport
from opencda.metrics_tools.report_builder import UniversalReportBuilder

if TYPE_CHECKING:
    from opencda.core.common.coperception_model_manager import CoperceptionModelManager
    from opencda.metrics_tools.metric_collector import MetricCollector

logger = logging.getLogger("cavise.opencda.opencda.scenario_testing.evaluations.evaluate_manager")


class EvaluationManager(object):
    """
    Evaluation manager to manage the analysis of the
    results for different modules.

    Parameters
    ----------
    cav_world : opencda object
        The CavWorld object that contains all CAVs' information.

    script_name : str
        The current scenario testing name. E.g, single_town06_carla

    current_time : str
        Current timestamp, used to name the output folder.

    Attributes
    ----------
    eval_save_path : str
        The final output folder name.

    """

    def __init__(self, cav_world: CavWorld, script_name: str, current_time: str) -> None:
        self.cav_world = cav_world

        current_path = os.path.dirname(os.path.realpath(__file__))

        self.eval_save_path = os.path.join(current_path, "../../../simulation_output/evaluation_outputs", f"{script_name}_{current_time}")
        if not os.path.exists(self.eval_save_path):
            os.makedirs(self.eval_save_path)

    def evaluate(
        self,
        coperception_model_manager: "CoperceptionModelManager | None" = None,
        scenario_metrics_collector: "MetricCollector | None" = None,
    ) -> None:
        """
        Evaluate performance of all modules and persist structured outputs.
        """
        localization_report = self.localization_eval()
        logger.info("Localization Evaluation Done")

        planning_report = self.kinematics_eval()
        logger.info("Kinematics Evaluation Done")

        platooning_reports = self.platooning_eval()
        logger.info("Platooning Evaluation Done")

        coperception_report = self.coperception_eval(coperception_model_manager)
        logger.info("Cooperative perception evaluation done")

        scenario_report = self.scenario_eval(scenario_metrics_collector)
        logger.info("Scenario evaluation done")

        json_save_path = os.path.join(self.eval_save_path, "report.json")
        with open(json_save_path, "w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "scenario": scenario_report.to_dict(),
                    "planning": planning_report.to_dict(),
                    "localization": localization_report.to_dict(),
                    "platooning": [report.to_dict() for report in platooning_reports],
                    "coperception": coperception_report.to_dict(),
                },
                output_file,
                indent=2,
            )

        logger.info("Evaluation JSON report saved to: %s", json_save_path)
        self._build_metric_plots(
            module_reports=(planning_report, localization_report, coperception_report, scenario_report),
            platooning_reports=platooning_reports,
        )

    def _build_metric_plots(
        self,
        module_reports: tuple[ModuleReport, ...],
        platooning_reports: tuple[GroupReport, ...],
    ) -> None:
        plot_builder = MetricPlotBuilder()
        plots_dir = os.path.join(self.eval_save_path, "plots")
        output_paths: list[Path] = []

        for module_report in module_reports:
            output_paths.extend(plot_builder.build_module_plots(module_report, plots_dir))

        for platooning_report in platooning_reports:
            output_paths.extend(plot_builder.build_group_plots(platooning_report, plots_dir, module="platooning"))

        logger.info("Evaluation metric plots saved to: %s (%d files)", plots_dir, len(output_paths))

    def kinematics_eval(self) -> ModuleReport:
        """
        vehicle kinematics related evaluation.
        """
        report_builder = UniversalReportBuilder()
        kinematics_reports: list[EntityReport] = []

        for _, vm in self.cav_world.get_vehicle_managers().items():
            raw_data = vm.agent.metrics_collector.get_raw()
            kinematics_reports.append(report_builder.build_entity_report(raw_data))

        return report_builder.build_module_report("planning", kinematics_reports)

    def localization_eval(self) -> ModuleReport:
        """
        Localization module evaluation.
        """
        report_builder = UniversalReportBuilder()
        localization_reports: list[EntityReport] = []
        for _, vm in self.cav_world.get_vehicle_managers().items():
            metrics_collector = getattr(vm.localizer, "metrics_collector", None)
            if metrics_collector is None:
                continue
            raw_data = metrics_collector.get_raw()
            localization_reports.append(report_builder.build_entity_report(raw_data))

        return report_builder.build_module_report("localization", localization_reports)

    def platooning_eval(self) -> tuple[GroupReport, ...]:
        """
        Platooning evaluation.
        """
        report_builder = UniversalReportBuilder()
        platooning_reports: list[GroupReport] = []

        for pmid, pm in self.cav_world.get_platoon_dict().items():
            member_metrics = pm.get_metric_collections()
            platooning_reports.append(report_builder.build_group_report(pmid, member_metrics, module="platooning"))

        return tuple(platooning_reports)

    def coperception_eval(self, coperception_model_manager: "CoperceptionModelManager | None" = None) -> ModuleReport:
        """
        Cooperative perception module evaluation.
        """
        report_builder = UniversalReportBuilder()
        if coperception_model_manager is None:
            return report_builder.build_module_report("coperception", ())

        raw_data = coperception_model_manager.get_metric_collection()
        coperception_report = report_builder.build_entity_report(raw_data)
        return report_builder.build_module_report("coperception", (coperception_report,))

    def scenario_eval(self, scenario_metrics_collector: "MetricCollector | None" = None) -> ModuleReport:
        """
        Scenario-level evaluation.
        """
        report_builder = UniversalReportBuilder()
        if scenario_metrics_collector is None:
            return report_builder.build_module_report("scenario", ())

        scenario_report = report_builder.build_entity_report(scenario_metrics_collector.get_raw())
        return report_builder.build_module_report("scenario", (scenario_report,))
