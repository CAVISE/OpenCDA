"""
Script to run different scenarios.
"""

import os
import sys
import enum
import errno
import typing
import pathlib
import logging
import argparse
import omegaconf
import subprocess

from opencda.version import __version__


try:
    from rich.traceback import install as rich_traceback_install
except ModuleNotFoundError:
    rich_traceback_install = None
    print("Rich tracebacks are not available, all CLI configuration regarding tracebacks is ignored.")


try:
    import coloredlogs
except ModuleNotFoundError:
    coloredlogs = None
    print("could not find coloredlogs module! Your life will look pale.")
    print("if you are interested in improving it: https://pypi.org/project/coloredlogs")


BUILD_COMPLETED_FLAG = "BUILD_COMPLETED_FLAG"


class VerbosityLevel(enum.IntEnum):
    # minimal output: important info, warnings and errors
    SILENT = 1
    # more info: might include some specific information about
    # servives, simulators' state - should match INFO logging level
    INFO = 2
    # all output, use this for development
    FULL = 3


# Handle cavise log creation, obtain this logger later with a call to
# logging.getLogger('cavise'). Use for our (cavise) code only.
def create_logger(level: int, fmt: str = "- [%(asctime)s][%(name)s] %(message)s", datefmt: str = "%H:%M:%S") -> logging.Logger:
    logger = logging.getLogger("cavise")
    if coloredlogs is not None:
        coloredlogs.install(level=level, logger=logger, fmt=fmt, datefmt=datefmt)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handler.setLevel(level)
        logger.addHandler(handler)
    logger.propagate = False  # noqa: DC05
    return logger


def install_traceback_handler(verbose: bool = True, suppress_modules: typing.Collection[str] = ()):
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

    joined = set(default_filtered_modules) & set(suppress_modules)
    if rich_traceback_install is not None:
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
    parser.add_argument(
        "-v", "--version", type=str, default="0.9.15", help="Specify the CARLA simulator version (this does not have any effect in our fork)"
    )
    parser.add_argument("--free-spectator", action="store_true", help="Enable free movement for the spectator camera.")
    parser.add_argument("-x", "--xodr", action="store_true", help="Run simulation using a custom map from an XODR file.")
    parser.add_argument("-c", "--cosim", action="store_true", help="Enable co-simulation with SUMO.")
    parser.add_argument("--with-capi", action="store_true", help="wether to run a communication manager instance in this simulation.")
    parser.add_argument("--carla-host", type=str, default="carla", help="IP address or hostname of the CARLA server (default: 'carla')")
    parser.add_argument("--carla-timeout", type=float, default=30.0, help="Timeout of the CARLA server response (default: '30.0')")

    # Coperception models parameters
    parser.add_argument(
        "--with-coperception", action="store_true", help="Whether to enable the use of cooperative perception models in this simulation."
    )
    parser.add_argument("--model-dir", type=str, help="Continued training path")
    parser.add_argument("--fusion-method", type=str, default="late", help="late, early or intermediate")
    parser.add_argument("--show-vis", action="store_true", help="whether to show image visualization result")
    parser.add_argument(
        "--show-sequence", action="store_true", help="whether to show video visualization result. It can not be set true with show_vis together."
    )
    parser.add_argument("--save-vis", action="store_true", help="whether to save visualization result")
    parser.add_argument("--save-npy", action="store_true", help="whether to save prediction and gt result in npy_test file")
    parser.add_argument(
        "--global-sort-detections",
        action="store_true",
        help="whether to globally sort detections by confidence score."
        "If set to True, it is the mainstream AP computing method,"
        "but would increase the tolerance for FP (False Positives).",
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

    # [CoDrivingInt] Codriveing models parametrs
    parser.add_argument("--with-mtp", action="store_true", help="Whether to enable the use of cooperative driving models in this simulation.")
    parser.add_argument("--mtp-config", type=str, default="mtp_config_default", help="Define configuration of cooperative driving model.")
    # [CoDrivingInt]

    parser.add_argument("--ticks", type=int, help="number of simulation ticks to execute")
    return parser.parse_args()


def check_buld_for_utils(module_path: str, cwd: pathlib.PurePath, verbose: bool, logger: logging.Logger) -> bool:
    marker_file = cwd.joinpath(f"OpenCOOD/{module_path}/{BUILD_COMPLETED_FLAG}")
    module_name = f"opencood.{module_path.split('/')[-2]}"
    if os.path.isfile(marker_file):
        logger.info(f"{module_name} is already built")
        return True

    try:
        logger.info(f"Building {module_name} ...")
        result = subprocess.run(
            ["python", f"{module_path}setup.py", "build_ext", "--inplace"], check=True, cwd=cwd.joinpath("OpenCOOD"), capture_output=True, text=True
        )
        os.close(os.open(str(marker_file), os.O_CREAT))
        logger.info(f"Complete building {module_name}")
        if verbose:
            logger.info(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        logger.info(f"Compilation error {module_name}:")
        if verbose:
            logger.info(e.stderr)
        return False


def main() -> None:
    opt = arg_parse()

    verbosity = opt.verbose
    if verbosity == VerbosityLevel.FULL:
        level = logging.DEBUG
    elif verbosity == VerbosityLevel.INFO:
        level = logging.INFO
    else:
        level = logging.WARNING

    logger = create_logger(level)
    install_traceback_handler(verbose=verbosity != VerbosityLevel.SILENT)

    logger.info(f"OpenCDA Version: {__version__}")

    cwd = pathlib.PurePath(os.getcwd())
    default_yaml = config_yaml = cwd / "opencda/scenario_testing/config_yaml/default.yaml"
    config_yaml = cwd / f"opencda/scenario_testing/config_yaml/{opt.test_scenario}.yaml"
    if not os.path.isfile(config_yaml):
        logger.error(f"{config_yaml.relative_to(cwd)} not found!")
        sys.exit(errno.EPERM)

    # allow OpenCOOD imports
    sys.path.append(str(cwd.joinpath("OpenCOOD")))

    # set the yaml file for the specific testing scenario
    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = omegaconf.OmegaConf.load(str(default_yaml))
    scene_dict = omegaconf.OmegaConf.load(str(config_yaml))
    scene_dict = omegaconf.OmegaConf.merge(default_dict, scene_dict)

    # NOTICE: temporary measure (while option is turned off)
    opt.apply_ml = False

    if opt.with_coperception:
        opencood_utils = "opencood/utils/"
        opencood_pcdet_utils = "opencood/pcdet_utils/"
        if not check_buld_for_utils(opencood_utils, cwd, verbosity == VerbosityLevel.FULL, logger):
            logger.error("Failed to build opencood.utils")
        if not check_buld_for_utils(opencood_pcdet_utils, cwd, verbosity == VerbosityLevel.FULL, logger):
            logger.error("Failed to build opencood.pcdet_utils")

    # this function might setup crucial components in Scenario, so
    # we should import as late as possible
    from opencda.scenario_testing.scenario import run_scenario

    run_scenario(opt, scene_dict)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("- Exited by user.")
