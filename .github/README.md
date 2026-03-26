# <img src="https://raw.githubusercontent.com/Haralishev77/Haralishev77/main/CAVISE-square-logo.png" alt="CAVISE" width="28" style="border-radius: 6px;" /> OpenCDA (CAVISE Fork)

<p align="center">
  <img src="https://raw.githubusercontent.com/Haralishev77/Haralishev77/main/CAVISE.png" alt="CAVISE banner" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/CAVISE/opencda/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/CAVISE/opencda?style=for-the-badge&color=9BFFCE&logo=git&logoColor=D9E0EE&labelColor=1E202B" /></a>
  <a href="https://github.com/CAVISE/opencda/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/CAVISE/opencda?style=for-the-badge&logo=github&color=9BFFCE&logoColor=D9E0EE&labelColor=1E202B" /></a>
  <a href="https://github.com/CAVISE/opencda/releases"><img alt="Latest Release" src="https://img.shields.io/github/v/release/CAVISE/opencda?style=for-the-badge&logo=github&color=9BFFCE&logoColor=D9E0EE&labelColor=1E202B" /></a>
  <a href="https://github.com/CAVISE/CAVISE"><img alt="CAVISE" src="https://img.shields.io/badge/CAVISE-main%20repository-9BFFCE?style=for-the-badge&logo=github&logoColor=D9E0EE&labelColor=1E202B" /></a>
  <a href="https://github.com/CAVISE/artery"><img alt="Artery" src="https://img.shields.io/badge/CAVISE-Artery-9BFFCE?style=for-the-badge&logo=github&logoColor=D9E0EE&labelColor=1E202B" /></a>
</p>

OpenCDA is a research and engineering framework for cooperative driving automation and other various applications built on top of CARLA and SUMO.

This repository is a fork of the original OpenCDA project. The upstream repository is available at [ucla-mobility/OpenCDA](https://github.com/ucla-mobility/OpenCDA).

## Overview

The upstream project already provides:
- YAML-driven scenario configuration
- CARLA-only simulation mode
- CARLA + SUMO co-simulation mode
- platooning, sensoring etc.

This fork adds:
- Artery-based V2X wireless links simulation
- cooperative perception pipelines backed by bundled OpenCOOD models
- CAPI as the data exchange interface between OpenCDA and Artery for much more realistic signal propagation simulation
- AIM-based cooperative driving integration

## Requirements

For full installation and launch instructions, use the CAVISE wiki:

- [Install & Launch](https://github.com/CAVISE/CAVISE/wiki/Install&Launch)

At a high level, the current fork expects:

- CARLA `0.9.16`
- Python `3.10` or higher
- CUDA and support for GPU inside Docker runtime

This fork is intended to run inside the CAVISE Docker environment. Running this fork outside Docker has not been tested.

## Quick Start

The canonical environment setup is also documented in the CAVISE wiki:

- [Install & Launch](https://github.com/CAVISE/CAVISE/wiki/Install&Launch)

The commands below assume the CAVISE containerized setup from the wiki.

Typical OpenCDA commands in this repository:

```bash
# Run a CARLA-only scenario
python opencda.py -t v2xp_datadump_town06_carla

# Run a CARLA + SUMO co-simulation scenario
python opencda.py -t v2xp_datadump_town06_carla --cosim

# Run a scenario with cooperative perception
python opencda.py -t v2xp_datadump_town06_carla --with-coperception \
--model-dir opencda/coperception_models/pointpillar-where2comm-intermediate-v2xsim-50 --fusion-method intermediate
```

Useful notes:

- Scenario configurations live in `opencda/scenario_testing/config_yaml`.
- `opencda/scenario_testing/config_yaml/default.yaml` is the shared base configuration loaded for every scenario.
- `--carla-host` defaults to `carla` in the containerized setup.
- When running OpenCDA inside the container against CARLA on Windows, use `--carla-host host.docker.internal` as documented in the CAVISE wiki.

## Usage

The main entry point is:

```bash
python3 opencda.py -t <scenario_name> [options]
```

Scenario files are loaded from `opencda/scenario_testing/config_yaml`. OpenCDA loads `default.yaml` together with `opencda/scenario_testing/config_yaml/<scenario>.yaml`.

### Core options

- `-t, --test-scenario`: Define the name of the scenario you want to test. Notice, this only has effect on configurations that are picked up by scenario
- `--record`: Whether to record and save the simulation process to .log file
- `-v, --version`: Specify the CARLA simulator version (this does not have any effect in our fork)
- `--free-spectator`: Enable free movement for the spectator camera.
- `--ticks`: number of simulation ticks to execute
- `--verbose`: Specifies overall verbosity of output.

### Simulation backend

- `-x, --xodr`: Run simulation using a custom map from an XODR file.
- `-c, --cosim`: Enable co-simulation with SUMO. Requires a running SUMO container configured according to the selected scenario.
- `--carla-host`: IP address or hostname of the CARLA server (default: 'carla')
- `--carla-timeout`: Timeout of the CARLA server response in seconds (default: 30.0)

### CAPI v2 / Artery integration

CAPI v2 is the data exchange interface between OpenCDA and Artery. In this fork it is used to exchange OpenCDA and Artery data for more realistic signal propagation simulation.

- `--with-capi`: wether to run a communication manager instance in this simulation.
- `--artery-host`: IP address or hostname and port of the Artery server (default: 'artery:7777')
- `--artery-send-timeout`: Maximum time to send a message to the Artery server, in seconds (default: 5.0).
- `--artery-receive-timeout`: Maximum time to wait for a reply from the Artery server, in seconds (default: 300.0).

### Cooperative perception

- `--with-coperception`: Whether to enable the use of cooperative perception models in this simulation.
- `--model-dir`: Continued training path
- `--fusion-method`: late, early or intermediate
- `--show-vis`: whether to show image visualization result
- `--show-sequence`: whether to show video visualization result. It can not be set true with show_vis together.
- `--save-vis`: whether to save visualization result
- `--save-npy`: whether to save prediction and gt result in npy_test file
- `--global-sort-detections`: whether to globally sort detections by confidence score.If set to True, it is the mainstream AP computing method,but would increase the tolerance for FP (False Positives).

Example:

```bash
python3 opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --model-dir opencda/coperception_models/pointpillar-late-opv2v-30 \
  --fusion-method late \
  --save-vis
```

### AIM

AIM is the cooperative driving module used by `--with-mtp`. The scenario YAML can define its parameters in the `aim:` block, for example in `opencda/scenario_testing/config_yaml/codriving_check.yaml`.

- `--with-mtp`: Whether to enable the use of cooperative driving models in this simulation. Note that this mode requires `--cosim` to be enabled.
- `--mtp-config`: Define configuration of cooperative driving model.

Example:

```bash
python3 opencda.py -t codriving_check --cosim --with-mtp
```

### Output directories

- `simulation_output/data_dumping/`: recorder logs and scenario dumps
- `simulation_output/coperception/`: cooperative perception predictions, visualizations, and results

## Architecture / How It Works

The current fork is organized around a scenario runner that turns YAML configuration into a live simulation.

### Execution flow

1. `opencda.py` parses CLI arguments and loads `default.yaml` together with the selected scenario YAML.
2. A `CavWorld` instance holds shared runtime state and model managers.
3. `ScenarioManager` or `CoScenarioManager` constructs the CARLA-only or CARLA + SUMO simulation.
4. Vehicle, platoon, RSU, and background traffic managers are created from the scenario definition.
5. Each tick updates sensing, localization, planning, control, communication, and optional ML modules.
6. At shutdown, evaluation artifacts and optional recorder outputs are written to `simulation_output/`.

### Key capabilities in this fork

- YAML-first scenario definition with a shared baseline config in `opencda/scenario_testing/config_yaml/default.yaml`
- single-vehicle, platooning, intersection, RSU, and cooperative perception scenarios
- CARLA traffic manager support and SUMO-backed co-simulation
- full-stack ego logic with sensing, localization, planning, control, safety, and map management
- cooperative perception model integration through the bundled `OpenCOOD` subtree
- Artery communication stack with CAPI v2 protobuf-based message handling
- AIM/MTP cooperative driving integration for scenario-controlled trajectory generation

### Repository map

- `opencda/core`: runtime modules for sensing, planning, control, map, safety, applications, and common managers
- `opencda/scenario_testing`: scenario runner, YAML configs, evaluation, and utility APIs
- `opencda/customize`: extension points for custom algorithms
- `opencda/core/common/communication`: communication stack, protobuf messages, and tests
- `OpenCOOD`: cooperative perception code and bundled model support
- `docs`: repository-local documentation and API stubs

## Cooperative Perception Examples

### Scenario `v2xp_datadump_town06_carla`

Run cooperative perception with the bundled Where2Comm intermediate fusion model:

```bash
python3 opencda.py \
  -t v2xp_datadump_town06_carla \
  --with-coperception \
  --model-dir opencda/coperception_models/pointpillar-where2comm-intermediate-v2xsim-50 \
  --fusion-method intermediate \
  --save-vis
```

<p align="center">
  <img src="docs/md_files/images/v2xp_datadump_town06_carla_3d.gif" alt="v2xp_datadump_town06_carla 3D view" width="49%" />
  <img src="docs/md_files/images/v2xp_datadump_town06_carla_bev.gif" alt="v2xp_datadump_town06_carla BEV view" width="49%" />
</p>

<p align="center"><em>Left: 3D view. Right: BEV.</em></p>

## Development Environment Setup

This repository uses `pre-commit` to keep formatting and basic checks consistent before code reaches CI.

Install and enable it once per clone. After that, it will run checks automatically before each commit.:

```bash
python3 -m pip install pre-commit
pre-commit install
```

Run all hooks manually:

```bash
pre-commit run --all-files
```

Useful day-to-day commands:

```bash
# Run only Ruff linting/fixes
pre-commit run ruff-check --all-files

# Run only formatting
pre-commit run ruff-format --all-files
```

The current hook set includes:

- `ruff-check` with auto-fix enabled
- `ruff-format`
- `hadolint` for `Dockerfile`
- common safety and hygiene checks for YAML, JSON, TOML, merge conflicts, whitespace, symlinks, and requirements files

CI also runs additional checks such as `pytest`, `mypy`, `deadcode`, and repository-wide `pre-commit`, so it is worth running hooks locally before opening a PR.

## Documentation and Links

- CAVISE wiki: [Install & Launch](https://github.com/CAVISE/CAVISE/wiki/Install&Launch)
- Local docs entry points:
  - [Introduction](docs/md_files/introduction.md)
  - [Quick Start](docs/md_files/getstarted.md)
  - [Logic Flow](docs/md_files/logic_flow.md)
  - [Customization Guide](docs/md_files/customization.md)
  - [Developer Tutorial](docs/md_files/developer_tutorial.md)
  - [API Docs Index](docs/modules.rst)

TODO: the in-repository documentation is still rough and partially outdated. It will be updated incrementally as the fork evolves.

## Contributing

Contributions are welcome and appreciated.

We are happy to review pull requests and contribution ideas for this fork.

Before starting development, please make sure your environment is set up according to the [Development Environment Setup](#development-environment-setup) section.

If you are planning a larger change, opening an issue first is a good way to align scope and avoid duplicated work.

## Contributors

A huge thank you to everyone who contributes to this fork.

[![OpenCDA contributors](https://contrib.rocks/image?repo=CAVISE/opencda)](https://github.com/CAVISE/opencda/graphs/contributors)

We look forward to your contributions to help make the CAVISE OpenCDA fork even better.

## Contact

For bug reports and feature requests related to this fork, please visit [GitHub Issues](https://github.com/CAVISE/opencda/issues). For installation and launch details, please refer to the [CAVISE Install & Launch wiki](https://github.com/CAVISE/CAVISE/wiki/Install%26Launch). We're happy to help with OpenCDA, Artery, CAPI v2, cooperative perception, and AIM workflows in this fork.
