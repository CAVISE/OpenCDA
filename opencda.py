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
from types import ModuleType
from typing import Any, cast

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
from opencda.version import __version__

logger = logging.getLogger("cavise.opencda")
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
        """Serialize diagnostics without persisting HTTP URL credentials.

        Parameters
        ----------
        record : logging.LogRecord
            Message and optional exception to store in the run's JSON log.

        Returns
        -------
        str
            JSON record with credential-bearing URLs sanitized in every field.
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

        return json.dumps(run_metadata.to_jsonable(payload), ensure_ascii=False)


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
# Remove a saved mutually exclusive option when replay overrides it.
EXCLUSIVE_OPTIONS = (("--model-id", "--model-dir"),)
# Do not restore original output paths during replay to avoid overwriting the original run artifacts.
REPLAY_OUTPUT_OPTIONS = ("--log-file",)


def _strip_option(argv: list[str], flag: str) -> list[str]:
    """Remove a value-taking option in split or equals form.

    Parameters
    ----------
    argv : list of str
        Command arguments to filter without modifying the original list.
    flag : str
        Exact option spelling whose occurrences and values are removed.

    Returns
    -------
    list of str
        Arguments with the specified option removed.
    """
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


# Parse command line args.
def build_parser() -> argparse.ArgumentParser:
    """Build the scenario runner's command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser defining scenario, replay and runtime options and defaults.
    """
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
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
    parser.add_argument(
        "--no-auto-fetch-models", action="store_true", default=None, help="Fail instead of fetching a missing model or runtime asset bundle."
    )
    parser.add_argument(
        "--auto-fetch-models",
        dest="no_auto_fetch_models",
        action="store_false",
        default=None,
        help="Allow fetching missing model or runtime asset bundles.",
    )
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


def build_command_from_options(opt: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> list[str]:
    """Build canonical launch arguments from resolved options.

    Parameters
    ----------
    opt : argparse.Namespace
        Parsed or replay-merged runtime options.
    parser : argparse.ArgumentParser, optional
        Existing parser to reuse; None builds the standard parser.

    Returns
    -------
    list of str
        Canonical non-default arguments, excluding the replay path itself.
    """
    argv: list[str] = []
    for action in (parser or build_parser())._actions:
        if not action.option_strings or action.dest in {"help", "version", "replay_from"} or not hasattr(opt, action.dest):
            continue
        value = getattr(opt, action.dest)
        if value == action.default:
            continue
        flag = max(action.option_strings, key=len)
        if isinstance(action, argparse.BooleanOptionalAction):
            positive = next((option for option in action.option_strings if option.startswith("--")), action.option_strings[0])
            argv.append(positive if value else f"--no-{positive[2:]}")
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                argv.append(flag)
        elif value is not None:
            argument = str(value.value if isinstance(value, enum.Enum) else value)
            if argument.startswith("-"):
                argv.append(f"{flag}={argument}")
            else:
                argv.extend([flag, argument])
    return argv


def resolve_launch_arguments(cli_argv: list[str]) -> tuple[argparse.Namespace, list[str], dict | None]:
    """Merge saved launch options with explicitly supplied CLI values.

    Parameters
    ----------
    cli_argv : list of str
        User arguments without the executable name, parsed once. Explicit
        values override saved values even when equal to parser defaults.

    Returns
    -------
    argparse.Namespace
        Effective runtime options; _explicit_options records user-supplied
        destinations for selecting saved versus overridden AdvCP inputs.
    list of str
        Canonical effective arguments for the new run manifest.
    dict or None
        Original manifest on replay, or None for a new run.

    Raises
    ------
    ValueError
        If a saved merged configuration was redacted or its hash differs.
    FileNotFoundError
        If a required saved manifest or configuration is missing.
    """
    parser = build_parser()
    # Suppress defaults while parsing the CLI so explicit default values still override saved values.
    defaults = {action.dest: action.default for action in parser._actions}
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    supplied = vars(arg_parse(cli_argv, parser))
    for action in parser._actions:
        action.default = defaults[action.dest]
    opt = argparse.Namespace(**{key: value for key, value in defaults.items() if value != argparse.SUPPRESS})
    vars(opt).update(supplied)
    if opt.replay_from is None:
        opt._explicit_options = set(supplied)
        return opt, build_command_from_options(opt, parser), None

    replay_dir = pathlib.Path(opt.replay_from).resolve()
    manifest = run_metadata.load_run_manifest(replay_dir)
    if manifest.get("config_redacted"):
        raise ValueError("Saved replay configuration contains redacted credentials and cannot be replayed without secure replacements.")

    expected_config_hash = manifest.get("config_sha256")
    if expected_config_hash is not None:
        saved_config = run_metadata.reproducibility_dir(replay_dir) / run_metadata.CONFIG_FILENAME
        if not saved_config.is_file():
            raise FileNotFoundError(f"Saved replay configuration not found: {saved_config}")

        actual_config_hash = run_metadata.sha256_file(saved_config)
        if actual_config_hash != expected_config_hash:
            raise ValueError("Saved replay configuration SHA-256 mismatch: config.yaml was modified or corrupted.")

    saved_argv = _strip_option(manifest["launch"]["argv"], REPLAY_FLAG)
    for flag in REPLAY_OUTPUT_OPTIONS:
        saved_argv = _strip_option(saved_argv, flag)
    for first, second in EXCLUSIVE_OPTIONS:
        if first.removeprefix("--").replace("-", "_") in supplied:
            saved_argv = _strip_option(saved_argv, second)
        if second.removeprefix("--").replace("-", "_") in supplied:
            saved_argv = _strip_option(saved_argv, first)
    opt = parser.parse_args(saved_argv)
    vars(opt).update(supplied)
    if supplied.get("with_coperception") is False:
        for name in ("model_id", "model_dir"):
            if name not in supplied:
                setattr(opt, name, None)
    opt._explicit_options = set(supplied)
    opt.replay_from = str(replay_dir)
    if opt.test_scenario is None:
        parser.error("the saved run and command line do not specify -t/--test-scenario")
    return opt, build_command_from_options(opt, parser), manifest


def main() -> None:
    global logger
    opt, effective_argv, replay_manifest = resolve_launch_arguments(sys.argv[1:])
    explicit_options = opt._explicit_options
    del opt._explicit_options
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
    if replay_dir is not None:
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

        # Replay uses the saved AdvCP config unless explicitly overridden.
        use_saved_advcp_config = replay_dir is not None and "advcp_config" not in explicit_options
        if use_saved_advcp_config:
            saved_input = (replay_manifest or {}).get("inputs", {}).get("advcp_config") or {}
            saved_copy = saved_input.get("copy", "inputs/advcp_config.yaml")
            replay_root = config_yaml.parent.resolve()
            saved_advcp_config = (replay_root / saved_copy).resolve()
            if not saved_advcp_config.is_relative_to(replay_root) or not saved_advcp_config.is_relative_to(replay_root / "inputs"):
                raise ValueError("Invalid saved AdvCP config path outside the run inputs directory.")
            if not saved_advcp_config.is_file():
                raise FileNotFoundError(f"Saved AdvCP configuration not found: {saved_advcp_config}")
            expected_hash = saved_input.get("sha256")
            if expected_hash is not None and run_metadata.sha256_file(saved_advcp_config) != expected_hash:
                raise ValueError(f"Saved AdvCP configuration SHA-256 mismatch: {saved_advcp_config.name} was modified or corrupted.")
            if saved_input.get("redacted"):
                raise ValueError("Saved AdvCP configuration contains redacted credentials; supply --advcp-config with a secure replacement.")
            advcp_config = saved_advcp_config
        elif not advcp_config.is_file():
            logger.error(f"AdvCP config not found: {advcp_config}")
            sys.exit(errno.EPERM)

        opt.advcp_config = str(advcp_config)

    model_resolver.resolve_runtime_models(opt)

    if replay_dir is not None:
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

    runtime_stats: dict[str, Any] = {}
    run_started_at = time.perf_counter()
    run_status, run_error = "failed", None
    # Record every failure after manifest creation in its final status.
    try:
        if replay_manifest is not None:
            for warning in run_metadata.compare_with_original(replay_manifest, opt):
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
