"""
Script to run different scenarios.
"""

import sys
import enum
import errno
import json
from datetime import datetime
import pathlib
import logging
import argparse
from collections.abc import Collection
from types import ModuleType
from typing import cast

from opencda.version import __version__


DEFAULT_LOG_FILENAME = "opencda.log.json"
EVALUATION_OUTPUT_ROOT = pathlib.Path("simulation_output/evaluation_outputs")


def get_default_log_path(scenario_name: str, current_time: str) -> pathlib.Path:
    return EVALUATION_OUTPUT_ROOT / f"{scenario_name}_{current_time}" / DEFAULT_LOG_FILENAME


class VerbosityLevel(enum.IntEnum):
    # minimal output: important info, warnings and errors
    SILENT = 1
    # more info: might include some specific information about
    # servives, simulators' state - should match INFO logging level
    INFO = 2
    # all output, use this for development
    FULL = 3


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created,
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# Handle cavise log creation, obtain this logger later with a call to
# logging.getLogger('cavise'). Use for our (cavise) code only.
def create_logger(
    level: int, fmt: str = "- [%(asctime)s][%(name)s] %(message)s", datefmt: str = "%H:%M:%S", filename: str = DEFAULT_LOG_FILENAME
) -> logging.Logger:
    logger = logging.getLogger("cavise.opencda")
    try:
        import coloredlogs
    except ModuleNotFoundError:
        print("could not find coloredlogs module! Your life will look pale.")
        print("if you are interested in improving it: https://pypi.org/project/coloredlogs")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    else:
        coloredlogs.install(level=logging.DEBUG, logger=logger, fmt=fmt, datefmt=datefmt)
    # Duplicate logs to a JSON file
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    json_handler = logging.FileHandler(filename=filename, mode="w", encoding="utf-8")
    json_handler.setLevel(logging.DEBUG)
    json_handler.setFormatter(JsonFormatter())
    logger.addHandler(json_handler)

    logger.propagate = False  # noqa: DC05
    logger.setLevel(level=level)
    return logger


def install_traceback_handler(verbose: bool = True, suppress_modules: Collection[str] = ()) -> None:
    default_filtered_modules = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
        "scikit-learn",
        "scikit-image",
        "omegaconf",
    ]

    joined_strings = set(default_filtered_modules) & set(suppress_modules)
    joined: set[str | ModuleType] = set(joined_strings)
    try:
        from rich.traceback import install as rich_traceback_install
    except ModuleNotFoundError:
        print("Rich tracebacks are not available, all CLI configuration regarding tracebacks is ignored.")
        return

    rich_traceback_install(show_locals=verbose, suppress=joined)


# Parse command line args.
def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    # opencda basic args
    parser.add_argument(
        "-t",
        "--test-scenario",
        required=True,
        type=str,
        help="Define the name of the scenario you want to test. Notice, this only has effect on configurations that are picked up by scenario",
    )
    parser.add_argument("--record", action="store_true", help="Whether to record and save the simulation process to .log file")
    # NOTICE: temporary disabled until we update yolo models.
    # parser.add_argument("--apply-ml", action='store_true',
    #                     help='whether ml/dl framework such as sklearn/pytorch is needed in the testing. '
    #                          'Set it to true only when you have installed the pytorch/sklearn package.')
    parser.add_argument("-v", "--version", action="version", version=f"OpenCDA v{__version__}")
    parser.add_argument("--free-spectator", action="store_true", help="Enable free movement for the spectator camera.")
    parser.add_argument("-x", "--xodr", action="store_true", help="Run simulation using a custom map from an XODR file.")
    parser.add_argument("-c", "--cosim", action="store_true", help="Enable co-simulation with SUMO.")
    parser.add_argument("--carla-host", type=str, default="carla", help="IP address or hostname of the CARLA server (default: 'carla')")
    parser.add_argument("--carla-timeout", type=float, default=30.0, help="Timeout of the CARLA server response in seconds (default: 30.0)")

    # CAPI parameters
    parser.add_argument("--with-capi", action="store_true", help="wether to run a communication manager instance in this simulation.")
    parser.add_argument(
        "--artery-host", type=str, default="artery:7777", help="IP address or hostname and port of the Artery server (default: 'artery:7777')"
    )
    parser.add_argument(
        "--artery-send-timeout", type=float, default=5.0, help="Maximum time to send a message to the Artery server, in seconds (default: 5.0)."
    )
    parser.add_argument(
        "--artery-receive-timeout",
        type=float,
        default=300.0,
        help="Maximum time to wait for a reply from the Artery server, in seconds (default: 300.0).",
    )

    # Coperception models parameters
    parser.add_argument(
        "--with-coperception", action="store_true", help="Whether to enable the use of cooperative perception models in this simulation."
    )
    parser.add_argument("--model-dir", type=str, help="Continued training path")
    parser.add_argument("--show-video-vis", action="store_true", help="whether to show video visualization result")
    parser.add_argument("--save-vis", action="store_true", help="whether to save visualization result")
    parser.add_argument("--save-npy", action="store_true", help="whether to save prediction and gt result in npy_test file")

    # AdvCollaborativePerception module
    parser.add_argument("--with-advcp", action="store_true", help="Enable AdvCP-style attacks for cooperative perception.")
    parser.add_argument(
        "--advcp-config",
        type=str,
        help="AdvCP attack config name or path. Relative names are resolved from opencda/scenario_testing/config_yaml/advcp.",
    )

    def verbosity_wrapper(arg: str) -> VerbosityLevel:
        return VerbosityLevel(int(arg))

    parser.add_argument(
        "--verbose",
        action="store",
        type=verbosity_wrapper,
        default=VerbosityLevel.FULL,
        choices=[level.value for level in VerbosityLevel],
        help="Specifies overall verbosity of output.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=(
            "Filename for the json log output. If not specified, logs will be saved to "
            f"{EVALUATION_OUTPUT_ROOT}/<scenario>_<timestamp>/{DEFAULT_LOG_FILENAME}."
        ),
    )

    parser.add_argument("--ticks", type=int, help="number of simulation ticks to execute")
    return parser.parse_args()


def main() -> None:
    opt = arg_parse()
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    import omegaconf
    from omegaconf import DictConfig

    verbosity = opt.verbose
    if verbosity == VerbosityLevel.FULL:
        level = logging.DEBUG
    elif verbosity == VerbosityLevel.INFO:
        level = logging.INFO
    else:
        level = logging.WARNING

    log_path = pathlib.Path(opt.log_file) if opt.log_file is not None else get_default_log_path(opt.test_scenario, current_time)
    logger = create_logger(level=level, filename=str(log_path))
    install_traceback_handler(verbose=verbosity != VerbosityLevel.SILENT)

    logger.info(f"OpenCDA v{__version__}")

    cwd = pathlib.Path.cwd()
    default_yaml = config_yaml = cwd / "opencda/scenario_testing/config_yaml/default.yaml"
    config_yaml = cwd / f"opencda/scenario_testing/config_yaml/{opt.test_scenario}.yaml"
    advcp_config_dir = cwd / "opencda/scenario_testing/config_yaml/advcp-configs"
    if not config_yaml.is_file():
        logger.error(f"{config_yaml.relative_to(cwd)} not found!")
        sys.exit(errno.EPERM)

    if opt.with_advcp:
        if not opt.with_coperception:
            logger.error("--with-advcp requires --with-coperception.")
            sys.exit(errno.EPERM)

        if not opt.advcp_config:
            logger.error("--with-advcp requires --advcp-config.")
            sys.exit(errno.EPERM)

        advcp_config = pathlib.Path(opt.advcp_config)
        if not advcp_config.is_absolute():
            advcp_config = advcp_config_dir / advcp_config

        if advcp_config.suffix == "":
            advcp_config = advcp_config.with_suffix(".yaml")

        if not advcp_config.is_file():
            logger.error(f"AdvCP config not found: {advcp_config}")
            sys.exit(errno.EPERM)

        opt.advcp_config = str(advcp_config)

    # allow OpenCOOD imports
    sys.path.append(str(cwd.joinpath("OpenCOOD")))

    # set the yaml file for the specific testing scenario
    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = omegaconf.OmegaConf.load(str(default_yaml))
    scene_dict = omegaconf.OmegaConf.load(str(config_yaml))
    scene_dict = omegaconf.OmegaConf.merge(default_dict, scene_dict)
    scene_dict = cast(DictConfig, scene_dict)

    # NOTICE: temporary measure (while option is turned off)
    opt.apply_ml = False

    # this function might setup crucial components in Scenario, so
    # we should import as late as possible
    from opencda.scenario_testing.scenario import run_scenario

    run_scenario(opt, scene_dict, current_time=current_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("- Exited by user.")
