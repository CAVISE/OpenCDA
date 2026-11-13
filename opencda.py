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
from contextlib import nullcontext
from functools import cache
from types import ModuleType
from typing import Any, cast

from opencda.version import __version__

logger = logging.getLogger("cavise.opencda")
DEFAULT_LOG_FILENAME = "opencda.log.json"
EVALUATION_OUTPUT_ROOT = pathlib.Path("simulation_output/evaluation_outputs")


@cache
def import_runtime_libs() -> None:
    """Load simulation dependencies after CLI parsing, once per process.

    Keeping these imports deferred lets help and version commands run without
    installed runtime packages. Cached loading preserves module references
    across repeated launches in the same process.
    """
    global DictConfig, OmegaConf, coloredlogs, rich_traceback_install
    global model_resolver, component_lifecycle, run_metadata
    from omegaconf import DictConfig, OmegaConf

    try:
        import coloredlogs
    except ModuleNotFoundError:
        coloredlogs = None

    try:
        from rich.traceback import install as rich_traceback_install
    except ModuleNotFoundError:
        rich_traceback_install = None

    from opencda.core.plan import model_resolver
    from opencda.scenario_testing import component_lifecycle
    from opencda.scenario_testing.utils import run_metadata


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
        """Serialize a diagnostic record as JSON.

        Parameters
        ----------
        record : logging.LogRecord
            Message and optional exception to store in the run's JSON log.

        Returns
        -------
        str
            JSON record including any formatted exception.
        """
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
    """Configure console and JSON logging for a scenario run.

    Parameters
    ----------
    level : int
        Minimum severity accepted by the OpenCDA logger.
    fmt, datefmt : str, optional
        Console message and timestamp formats.
    filename : str, optional
        JSON log destination, created with its parent directories.

    Returns
    -------
    logging.Logger
        Configured OpenCDA logger with propagation disabled.
    """
    logger = logging.getLogger("cavise.opencda")
    if coloredlogs is None:
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
    """Install the optional Rich traceback handler.

    Parameters
    ----------
    verbose : bool, optional
        Include local variables in formatted tracebacks.
    suppress_modules : collection of str, optional
        Known dependency modules whose traceback frames should be condensed.
    """
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
    if rich_traceback_install is None:
        print("Rich tracebacks are not available, all CLI configuration regarding tracebacks is ignored.")
        return

    rich_traceback_install(show_locals=verbose, suppress=joined)


REPLAY_FLAG = "--replay-from"


# Parse command line args.
def build_parser() -> argparse.ArgumentParser:
    """Build the scenario runner's command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser defining scenario, replay and runtime options and defaults.
    """
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.", allow_abbrev=False)
    # opencda basic args
    parser.add_argument(
        "-t",
        "--test-scenario",
        type=str,
        help="Define the name of or the global path to the scenario you want to test. Notice, this only has effect on configurations that are picked up by scenario",
    )
    parser.add_argument(
        REPLAY_FLAG,
        dest="replay_from",
        type=str,
        default=None,
        help=(
            "Reproduce a previous run from its evaluation output folder: launch arguments are taken from run_manifest.json "
            "and the scenario config from reproducibility/config.yaml. Use --replay-from on its own; saved experiment parameters cannot be overridden."
        ),
    )
    parser.add_argument(
        "--record", action=argparse.BooleanOptionalAction, default=False, help="Whether to record and save the simulation process to .log file"
    )
    parser.add_argument("-v", "--version", action="version", version=f"OpenCDA v{__version__}")
    parser.add_argument(
        "--free-spectator", action=argparse.BooleanOptionalAction, default=False, help="Enable free movement for the spectator camera."
    )
    parser.add_argument(
        "-x", "--xodr", action=argparse.BooleanOptionalAction, default=False, help="Run simulation using a custom map from an XODR file."
    )
    parser.add_argument("-c", "--cosim", action=argparse.BooleanOptionalAction, default=False, help="Enable co-simulation with SUMO.")
    parser.add_argument("--carla-host", type=str, default="carla", help="IP address or hostname of the CARLA server (default: 'carla')")
    parser.add_argument("--carla-timeout", type=float, default=10.0, help="Timeout of the CARLA server response in seconds (default: 10.0)")

    # CAPI parameters
    parser.add_argument(
        "--with-capi", action=argparse.BooleanOptionalAction, default=False, help="wether to run a communication manager instance in this simulation."
    )
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
        "--with-coperception",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to enable the use of cooperative perception models in this simulation.",
    )
    parser.add_argument("--model-id", type=str, help="Logical model bundle ID from the CAVISE models repository.")
    parser.add_argument("--model-dir", type=str, help="Explicit local cooperative-perception model directory; disables automatic fetching.")
    parser.add_argument("--models-root", type=str, help="Models checkout path (default: CAVISE_MODELS_ROOT or the sibling models repository).")
    parser.add_argument("--models-repository", type=str, help="Git repository used to fetch missing model bundles.")
    parser.add_argument("--models-ref", type=str, help="Git branch or tag used when cloning the models repository (default: main).")
    parser.add_argument("--no-auto-fetch-models", action="store_true", help="Fail instead of fetching a missing model or runtime asset bundle.")
    parser.add_argument("--show-video-vis", action=argparse.BooleanOptionalAction, default=False, help="whether to show video visualization result")
    parser.add_argument("--save-vis", action=argparse.BooleanOptionalAction, default=False, help="whether to save visualization result")
    parser.add_argument(
        "--save-npy", action=argparse.BooleanOptionalAction, default=False, help="whether to save prediction and gt result in npy_test file"
    )

    # AdvCollaborativePerception module
    parser.add_argument(
        "--with-advcp", action=argparse.BooleanOptionalAction, default=False, help="Enable AdvCP-style attacks for cooperative perception."
    )
    parser.add_argument(
        "--advcp-config",
        type=str,
        help="AdvCP attack config name or path. Relative names are resolved from opencda/scenario_testing/config_yaml/advcp.",
    )
    parser.add_argument("--advcp-assets-id", type=str, default="base-car", help="Logical AdvCP runtime asset bundle ID.")

    # CARLA and SUMO lifecycle management parameters
    parser.add_argument(
        "--manage-components", action=argparse.BooleanOptionalAction, default=False, help="Start and stop dedicated component adapters for this run."
    )
    parser.add_argument("--carla-adapter-url", default="http://carla:8080", help="CARLA adapter base URL.")
    parser.add_argument("--sumo-adapter-url", default="http://sumo:8080", help="SUMO adapter base URL.")

    def verbosity_wrapper(arg: str) -> VerbosityLevel:
        """Convert a command-line verbosity value to its runtime enum.

        Parameters
        ----------
        arg : str
            Integer verbosity level supplied on the command line.

        Returns
        -------
        VerbosityLevel
            Corresponding supported logging verbosity.

        Raises
        ------
        ValueError
            If the value is not an integer or a supported verbosity level.
        """
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
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False, help="Enable profiling.")

    parser.add_argument("--ticks", type=int, help="number of simulation ticks to execute")
    return parser


def arg_parse(argv: list[str] | None = None, parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    """Parse a scenario or replay command.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the executable name. None reads sys.argv[1:].
    parser : argparse.ArgumentParser, optional
        Existing parser to reuse; None builds the standard parser.

    Returns
    -------
    argparse.Namespace
        Parsed options with defaults applied.

    Raises
    ------
    SystemExit
        If arguments are invalid or neither a scenario nor a replay is given.
    """
    parser = parser or build_parser()
    opt = parser.parse_args(argv)
    if getattr(opt, "test_scenario", None) is None and getattr(opt, "replay_from", None) is None:
        parser.error(f"the following arguments are required: -t/--test-scenario (or {REPLAY_FLAG})")
    return opt


def resolve_launch_arguments(cli_argv: list[str]) -> tuple[argparse.Namespace, list[str], dict | None]:
    """Load a new launch or repeat a saved run without parameter overrides.

    Parameters
    ----------
    cli_argv : list of str
        Launch arguments, or --replay-from followed by a saved run directory.

    Returns
    -------
    argparse.Namespace
        Runtime options with saved model and asset paths restored for replay.
    list of str
        Original launch arguments to retain in the next manifest.
    dict or None
        Saved manifest for replay, or None for a new run.

    Raises
    ------
    SystemExit
        If replay is combined with experiment parameters or arguments are invalid.
    FileNotFoundError
        If the saved manifest is missing.
    """
    parser = build_parser()
    opt = arg_parse(cli_argv, parser)
    if opt.replay_from is not None:
        replay_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        replay_parser.add_argument(REPLAY_FLAG)
        _, extra = replay_parser.parse_known_args(cli_argv)
        if extra:
            parser.error("--replay-from must be used alone; start a new run to change experiment parameters")
    import_runtime_libs()
    if opt.replay_from is None:
        return opt, list(cli_argv), None

    replay_dir = pathlib.Path(opt.replay_from).resolve()
    manifest = run_metadata.load_run_manifest(replay_dir)
    saved_argv = list(manifest["launch"]["argv"])
    # Accept manifests from before prediction capture and auto-fetch flags were removed.
    saved_argv = [arg for arg in saved_argv if arg not in {"--save-predictions", "--no-save-predictions", "--auto-fetch-models"}]
    opt = arg_parse(saved_argv, parser)
    # Give each replay its own log instead of truncating the original run's log.
    opt.log_file = None
    saved_model = manifest.get("model") or {}
    model_directory = saved_model.get("directory") or (manifest.get("checkpoint") or {}).get("local_dir")
    if opt.with_coperception and model_directory:
        opt._pinned_model_id = saved_model.get("id") or opt.model_id
        opt.model_id, opt.model_dir = None, model_directory
    if opt.with_advcp and saved_model.get("advcp_assets_dir"):
        opt.advcp_assets_dir = saved_model["advcp_assets_dir"]
    opt.replay_from = str(replay_dir)
    return opt, saved_argv, manifest


def main() -> None:
    global logger
    opt, effective_argv, replay_manifest = resolve_launch_arguments(sys.argv[1:])
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    verbosity = opt.verbose
    if verbosity == VerbosityLevel.FULL:
        level = logging.DEBUG
    elif verbosity == VerbosityLevel.INFO:
        level = logging.INFO
    else:
        level = logging.WARNING

    scenario_path = pathlib.Path(opt.test_scenario)
    test_scenario_name = scenario_path.stem if scenario_path.suffix.lower() == ".yaml" else opt.test_scenario
    base_output_dir = get_default_output_path(test_scenario_name, current_time)
    evaluation_output_dir = base_output_dir
    time_prefix = current_time
    suffix = 0
    # Reserve the run before opening its log; same-second launches must not overwrite prior artifacts.
    while True:
        try:
            evaluation_output_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            if not evaluation_output_dir.exists() and not evaluation_output_dir.is_symlink():
                raise
            suffix += 1
            current_time = f"{time_prefix}_{suffix}"
            evaluation_output_dir = base_output_dir.with_name(f"{base_output_dir.name}_{suffix}")
    log_path = pathlib.Path(opt.log_file) if opt.log_file is not None else evaluation_output_dir / DEFAULT_LOG_FILENAME
    logger = create_logger(level=level, filename=str(log_path))
    install_traceback_handler(verbose=verbosity != VerbosityLevel.SILENT)

    logger.info(f"OpenCDA v{__version__}")

    cwd = pathlib.Path.cwd()
    default_yaml = config_yaml = cwd / "opencda/scenario_testing/config_yaml/default.yaml"
    if scenario_path.suffix.lower() == ".yaml":
        config_yaml = scenario_path
        opt.test_scenario = test_scenario_name
    else:
        config_yaml = cwd / f"opencda/scenario_testing/config_yaml/{opt.test_scenario}.yaml"
    advcp_config_dir = cwd / "opencda/scenario_testing/config_yaml/advcp-configs"
    replay_dir = pathlib.Path(opt.replay_from) if opt.replay_from else None
    use_saved_scenario = replay_dir is not None
    if use_saved_scenario:
        logger.info(f"replaying run from {replay_dir}")
        config_yaml = run_metadata.reproducibility_dir(replay_dir) / run_metadata.CONFIG_FILENAME
    if not config_yaml.is_file():
        logger.error(f"{config_yaml} not found!")
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

        # Replay uses the saved AdvCP configuration.
        saved_input = (replay_manifest or {}).get("inputs", {}).get("advcp_config") or {}
        use_saved_advcp_config = replay_dir is not None and bool(saved_input)
        if use_saved_advcp_config:
            saved_copy = saved_input.get("copy", "inputs/advcp_config.yaml")
            replay_root = run_metadata.reproducibility_dir(replay_dir).resolve()
            saved_advcp_config = (replay_root / saved_copy).resolve()
            if not saved_advcp_config.is_relative_to(replay_root / "inputs"):
                raise ValueError("Invalid saved AdvCP config path outside the run inputs directory.")
            if not saved_advcp_config.is_file():
                raise FileNotFoundError(f"Saved AdvCP configuration not found: {saved_advcp_config}")
            advcp_config = saved_advcp_config
        elif not advcp_config.is_file():
            logger.error(f"AdvCP config not found: {advcp_config}")
            sys.exit(errno.EPERM)

        opt.advcp_config = str(advcp_config)

    model_resolver.resolve_runtime_models(opt)

    if use_saved_scenario:
        # The replay configuration is already merged.
        scene_dict = cast(DictConfig, OmegaConf.load(str(config_yaml)))
    else:
        # set the yaml file for the specific testing scenario
        # load the default yaml file and the scenario yaml file as dictionaries
        default_dict = OmegaConf.load(str(default_yaml))
        scene_dict = cast(DictConfig, OmegaConf.merge(default_dict, OmegaConf.load(str(config_yaml))))

    # NOTICE: temporary measure (while option is turned off)
    opt.apply_ml = False

    run_metadata.write_run_artifacts(
        evaluation_output_dir,
        opt=opt,
        argv=effective_argv,
        merged_config=scene_dict,
        input_files={"advcp_config": pathlib.Path(opt.advcp_config) if opt.with_advcp else None},
        replayed_from=opt.replay_from,
    )

    runtime_stats: dict[str, Any] = {}
    run_status, run_error = "failed", None
    # Record every failure after manifest creation in its final status.
    try:
        if replay_manifest is not None:
            for warning in run_metadata.compare_with_original(replay_manifest):
                logger.warning(f"replay may differ from the original run: {warning}")

        # Scenario imports CARLA/OpenCOOD and initializes optional runtime components.
        # Defer it until arguments, model paths and run artifacts are ready.
        from opencda.scenario_testing.scenario import run_scenario

        environment = component_lifecycle.ManagedComponents(opt) if opt.manage_components else nullcontext()
        with environment:
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
    except SystemExit as error:
        if error.code is None or error.code == 0:
            run_status = "completed"
        else:
            run_error = repr(error)
        raise
    except Exception as error:
        run_error = repr(error)
        raise
    finally:
        run_metadata.finalize_run_manifest(
            evaluation_output_dir,
            status=run_status,
            error=run_error,
        )

    show_performance_stats(runtime_stats)


def show_performance_stats(runtime_stats: dict) -> None:
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
