"""
Script to run different scenarios.
"""

import sys
import enum
import errno
import json
import time
from datetime import datetime
import pathlib
import logging
import argparse
import cProfile
import pstats
from collections.abc import Collection
from types import ModuleType
from typing import cast

from opencda.version import __version__


def import_runtime_libs() -> None:
    global omegaconf, DictConfig
    import omegaconf
    from omegaconf import DictConfig


DEFAULT_LOG_FILENAME = "opencda.log.json"
EVALUATION_OUTPUT_ROOT = pathlib.Path("simulation_output/evaluation_outputs")


def get_default_output_path(scenario_name: str, current_time: str) -> pathlib.Path:
    return EVALUATION_OUTPUT_ROOT / f"{scenario_name}_{current_time}"


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


REPLAY_FLAG = "--replay-from"
# Remove a saved mutually exclusive option when replay overrides it.
EXCLUSIVE_OPTIONS = (("--model-id", "--model-dir"),)
# Do not restore original output paths during replay to avoid overwriting the original run artifacts.
REPLAY_OUTPUT_OPTIONS = ("--log-file",)


def _strip_option(argv: list[str], flag: str) -> list[str]:
    """Remove an option and its value from argv."""
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = True
            continue
        if arg.startswith(f"{flag}="):
            continue
        cleaned.append(arg)
    return cleaned


def _strip_replay_flag(argv: list[str]) -> list[str]:
    """Remove the replay flag and its path from argv."""
    return _strip_option(argv, REPLAY_FLAG)


# Parse command line args.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    # opencda basic args
    parser.add_argument(
        "-t",
        "--test-scenario",
        type=str,
        help="Define the name of the scenario you want to test. Notice, this only has effect on configurations that are picked up by scenario",
    )
    parser.add_argument(
        REPLAY_FLAG,
        dest="replay_from",
        type=str,
        default=None,
        help=(
            "Reproduce a previous run from its evaluation output folder: launch arguments are taken from run_manifest.json "
            "and the scenario config from reproducibility/config.yaml. Extra arguments given on the command line override the saved ones."
        ),
    )
    parser.add_argument(
        "--save-predictions",
        dest="save_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-tick cooperative perception boxes and confidence scores to the evaluation output folder (default: on).",
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
    parser.add_argument("--model-id", type=str, help="Logical model bundle ID from the CAVISE models repository.")
    parser.add_argument("--model-dir", type=str, help="Explicit local cooperative-perception model directory; disables automatic fetching.")
    parser.add_argument("--models-root", type=str, help="Models checkout path (default: CAVISE_MODELS_ROOT or the sibling models repository).")
    parser.add_argument("--models-repository", type=str, help="Git repository used to fetch missing model bundles.")
    parser.add_argument("--models-ref", type=str, help="Git branch or tag used when cloning the models repository (default: main).")
    parser.add_argument("--no-auto-fetch-models", action="store_true", help="Fail instead of fetching a missing model or runtime asset bundle.")
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
    parser.add_argument("--advcp-assets-id", type=str, default="base-car", help="Logical AdvCP runtime asset bundle ID.")

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
    parser.add_argument("--profile", action="store_true", help="Enable profiling.")

    parser.add_argument("--ticks", type=int, help="number of simulation ticks to execute")
    return parser


def arg_parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    opt = parser.parse_args(argv)
    if opt.test_scenario is None and opt.replay_from is None:
        parser.error(f"the following arguments are required: -t/--test-scenario (or {REPLAY_FLAG})")
    return opt


def build_command_from_options(opt: argparse.Namespace) -> list[str]:
    """Build a canonical command from parsed options.

    Only non-default options are included.
    """
    argv: list[str] = []
    for action in build_parser()._actions:
        if not action.option_strings or action.dest in {"help", "version", "replay_from"} or not hasattr(opt, action.dest):
            continue
        value = getattr(opt, action.dest)
        flag = max(action.option_strings, key=len)
        if isinstance(action, argparse.BooleanOptionalAction):
            # option_strings: ["--save-predictions", "--no-save-predictions"]
            if value != action.default:
                argv.append(action.option_strings[0] if value else action.option_strings[1])
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                argv.append(flag)
        elif value is not None and value != action.default:
            argv.extend([flag, str(value.value if isinstance(value, enum.Enum) else value)])
    return argv


def _cli_sets_option(cli_argv: list[str], flag: str) -> bool:
    """Check whether an option was given explicitly on the command line (in any spelling)."""
    return flag in build_command_from_options(arg_parse(cli_argv))


def resolve_launch_arguments(cli_argv: list[str]) -> tuple[argparse.Namespace, list[str], dict | None]:
    """Load saved launch options when replaying a run."""
    opt = arg_parse(cli_argv)
    if opt.replay_from is None:
        return opt, build_command_from_options(opt), None

    from opencda.scenario_testing.utils.run_metadata import load_run_manifest

    replay_dir = pathlib.Path(opt.replay_from).resolve()
    manifest = load_run_manifest(replay_dir)
    if manifest.get("config_redacted"):
        raise ValueError("Saved replay configuration contains redacted credentials and cannot be replayed without secure replacements.")

    expected_config_hash = manifest.get("config_sha256")
    if expected_config_hash is not None:
        from opencda.scenario_testing.utils.run_metadata import (
            CONFIG_FILENAME,
            reproducibility_dir,
            sha256_file,
        )

        saved_config = reproducibility_dir(replay_dir) / CONFIG_FILENAME
        if not saved_config.is_file():
            raise FileNotFoundError(f"Saved replay configuration not found: {saved_config}")

        actual_config_hash = sha256_file(saved_config)
        if actual_config_hash != expected_config_hash:
            raise ValueError("Saved replay configuration SHA-256 mismatch: config.yaml was modified or corrupted.")

    saved_argv = _strip_replay_flag(manifest["launch"]["argv"])
    for flag in REPLAY_OUTPUT_OPTIONS:
        saved_argv = _strip_option(saved_argv, flag)
    for first, second in EXCLUSIVE_OPTIONS:
        if _cli_sets_option(cli_argv, first):
            saved_argv = _strip_option(saved_argv, second)
        if _cli_sets_option(cli_argv, second):
            saved_argv = _strip_option(saved_argv, first)
    effective_argv = [*saved_argv, *_strip_replay_flag(cli_argv)]
    opt = arg_parse(effective_argv)
    opt.replay_from = str(replay_dir)
    return opt, build_command_from_options(opt), manifest


def main() -> None:
    global logger
    opt, effective_argv, replay_manifest = resolve_launch_arguments(sys.argv[1:])
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    import_runtime_libs()

    verbosity = opt.verbose
    if verbosity == VerbosityLevel.FULL:
        level = logging.DEBUG
    elif verbosity == VerbosityLevel.INFO:
        level = logging.INFO
    else:
        level = logging.WARNING

    log_path = (
        pathlib.Path(opt.log_file) if opt.log_file is not None else get_default_output_path(opt.test_scenario, current_time) / DEFAULT_LOG_FILENAME
    )
    logger = create_logger(level=level, filename=str(log_path))
    install_traceback_handler(verbose=verbosity != VerbosityLevel.SILENT)

    logger.info(f"OpenCDA v{__version__}")

    cwd = pathlib.Path.cwd()
    default_yaml = config_yaml = cwd / "opencda/scenario_testing/config_yaml/default.yaml"
    config_yaml = cwd / f"opencda/scenario_testing/config_yaml/{opt.test_scenario}.yaml"
    advcp_config_dir = cwd / "opencda/scenario_testing/config_yaml/advcp-configs"
    replay_dir = pathlib.Path(opt.replay_from) if opt.replay_from else None
    if replay_dir is not None:
        logger.info(f"replaying run from {replay_dir}")
        from opencda.scenario_testing.utils.run_metadata import CONFIG_FILENAME, reproducibility_dir

        config_yaml = reproducibility_dir(replay_dir) / CONFIG_FILENAME
    if not config_yaml.is_file():
        logger.error(f"{config_yaml.relative_to(cwd) if config_yaml.is_relative_to(cwd) else config_yaml} not found!")
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

        # Replay uses the saved AdvCP config unless explicitly overridden.
        use_saved_advcp_config = replay_dir is not None and not _cli_sets_option(sys.argv[1:], "--advcp-config")
        saved_advcp_config = config_yaml.parent / "inputs" / "advcp_config.yaml" if use_saved_advcp_config else None
        if saved_advcp_config is not None and saved_advcp_config.is_file():
            advcp_config = saved_advcp_config
        else:
            if saved_advcp_config is not None:
                logger.warning(f"saved AdvCP config {saved_advcp_config} not found, using {advcp_config}: replay may differ from the original run")
            if not advcp_config.is_file():
                logger.error(f"AdvCP config not found: {advcp_config}")
                sys.exit(errno.EPERM)

        opt.advcp_config = str(advcp_config)

    from opencda.core.plan.model_resolver import resolve_runtime_models

    resolve_runtime_models(opt)

    if replay_dir is not None:
        # The replay configuration is already merged.
        scene_dict = cast(DictConfig, omegaconf.OmegaConf.load(str(config_yaml)))
    else:
        # set the yaml file for the specific testing scenario
        # load the default yaml file and the scenario yaml file as dictionaries
        default_dict = omegaconf.OmegaConf.load(str(default_yaml))
        scene_dict = omegaconf.OmegaConf.load(str(config_yaml))
        scene_dict = omegaconf.OmegaConf.merge(default_dict, scene_dict)
        scene_dict = cast(DictConfig, scene_dict)

    # NOTICE: temporary measure (while option is turned off)
    opt.apply_ml = False

    from opencda.scenario_testing.utils import run_metadata

    evaluation_output_dir = get_default_output_path(opt.test_scenario, current_time)
    run_metadata.write_run_artifacts(
        evaluation_output_dir,
        opt=opt,
        argv=effective_argv,
        raw_argv=sys.argv[1:],
        merged_config=scene_dict,
        current_time=current_time,
        input_files={
            "default_config": default_yaml if replay_dir is None else None,
            "scenario_config": config_yaml if replay_dir is None else None,
            "replayed_config": config_yaml if replay_dir is not None else None,
            "advcp_config": pathlib.Path(opt.advcp_config) if opt.with_advcp else None,
        },
        replayed_from=opt.replay_from,
    )

    runtime_stats = {}
    run_started_at = time.perf_counter()
    run_status, run_error = "failed", None
    # Record every failure after manifest creation in its final status.
    try:
        if replay_manifest is not None:
            for warning in run_metadata.compare_with_original(replay_manifest, opt):
                logger.warning(f"replay may differ from the original run: {warning}")

        # this function might setup crucial components in Scenario, so
        # we should import as late as possible
        from opencda.scenario_testing.scenario import run_scenario

        if opt.profile:
            profiler = cProfile.Profile()
            profiler.enable()

            try:
                scenario_started_at = time.perf_counter()
                run_scenario(opt, scene_dict, current_time=current_time, runtime_stats=runtime_stats)
                runtime_stats["full_scenario"] = time.perf_counter() - scenario_started_at
            finally:
                evaluation_output_dir.mkdir(parents=True, exist_ok=True)
                profiler_output = evaluation_output_dir / "profile_output.prof"
                profiler.disable()
                profiler.dump_stats(profiler_output)
                logger.info(f"Profiler output saved to {profiler_output}")

                stats = pstats.Stats(profiler)
                stats.sort_stats("cumulative")
                stats.print_stats(30)
        else:
            scenario_started_at = time.perf_counter()
            run_scenario(opt, scene_dict, current_time=current_time, runtime_stats=runtime_stats)
            runtime_stats["full_scenario"] = time.perf_counter() - scenario_started_at
        run_status = "completed"
    except KeyboardInterrupt:
        run_status = "interrupted"
        raise
    except Exception as error:
        run_error = repr(error)
        raise
    finally:
        run_metadata.finalize_run_manifest(
            evaluation_output_dir,
            status=run_status,
            duration_seconds=round(time.perf_counter() - run_started_at, 3),
            error=run_error,
        )

    show_performance_stats(runtime_stats)


def show_performance_stats(runtime_stats: dict):
    global logger
    logger.debug("Scenario performance stats:")
    logger.debug(f"Full scenario: {runtime_stats['full_scenario']:.3f}s")
    logger.debug(f"Scenatio __init__: {runtime_stats['scenario__init__']:.3f}s")
    logger.debug(f"Agents initialization: {runtime_stats['_init_agents']:.3f}s")
    logger.debug(f"Avarage tick time: {sum(runtime_stats['ticks']) / len(runtime_stats['ticks']):.3f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("- Exited by user.")
