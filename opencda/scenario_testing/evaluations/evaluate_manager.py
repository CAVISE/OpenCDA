"""
Evaluation manager.
"""

import os
import json
import logging

from opencda.metrics_tools.report_models import EntityReport, GroupReport, ModuleReport
from opencda.metrics_tools.report_builder import UniversalReportBuilder

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

    def __init__(self, cav_world, script_name, current_time):
        self.cav_world = cav_world

        current_path = os.path.dirname(os.path.realpath(__file__))

        self.eval_save_path = os.path.join(current_path, "../../../simulation_output/evaluation_outputs", f"{script_name}_{current_time}")
        if not os.path.exists(self.eval_save_path):
            os.makedirs(self.eval_save_path)

    def evaluate(self):
        """
        Evaluate performance of all modules and persist structured outputs.
        """
        localization_report = self.localization_eval()
        logger.info("Localization Evaluation Done")

        planning_report = self.kinematics_eval()
        logger.info("Kinematics Evaluation Done")

        platooning_reports = self.platooning_eval()
        logger.info("Platooning Evaluation Done")

        json_save_path = os.path.join(self.eval_save_path, "report.json")
        with open(json_save_path, "w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "planning": planning_report.to_dict(),
                    "localization": localization_report.to_dict(),
                    "platooning": [report.to_dict() for report in platooning_reports],
                },
                output_file,
                indent=2,
            )

        logger.info("Evaluation JSON report saved to: %s", json_save_path)

    def kinematics_eval(self) -> ModuleReport:
        """
        vehicle kinematics related evaluation.
        """
        report_builder = UniversalReportBuilder()
        kinematics_reports: list[EntityReport] = []

        for vid, vm in self.cav_world.get_vehicle_managers().items():
            actor_id = vm.vehicle.id

            raw_data = vm.agent.metrics_collector.get_raw()
            kinematics_reports.append(report_builder.build_entity_report(raw_data))

        return report_builder.build_module_report("planning", kinematics_reports)

    def localization_eval(self) -> ModuleReport:
        """
        Localization module evaluation.
        """
        report_builder = UniversalReportBuilder()
        localization_reports: list[EntityReport] = []
        for vid, vm in self.cav_world.get_vehicle_managers().items():
            actor_id = vm.vehicle.id

            raw_data = vm.localizer.metrics_collector.get_raw()
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
