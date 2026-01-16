# OpenCDA (Fork)

> **This is a fork of the original OpenCDA project**  
> Original authors: UCLA Mobility Lab  
> Original paper: [OpenCDA: An Open Cooperative Driving Automation Framework Integrated with Co-simulation](https://arxiv.org/abs/2107.06260)

---

OpenCDA is a **SIMULATION** tool integrated with a prototype cooperative driving automation (CDA; see SAE J3216) pipeline as well as regular automated driving components (e.g., perception, localization, planning, control). The tool integrates automated driving simulation (CARLA), traffic simulation (SUMO), and Co-simulation.

## Key Features

- **Comprehensive CDA Pipeline**: Full stack from perception to control
- **V2X Communication**: Vehicle-to-vehicle and vehicle-to-infrastructure
- **CARLA Integration**: High-fidelity 3D simulation environment
- **SUMO Co-simulation**: Large-scale traffic flow simulation
- **Modular Design**: Easy customization and extension
- **Platooning Support**: Built-in cooperative platooning algorithms
- **Testing Framework**: Scenario-based testing and evaluation

## Quick Start

### Prerequisites

- Python 3.10+
- CARLA 0.9.X (recommended 0.9.16)
- SUMO 1.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/CAVISE/OpenCDA
cd OpenCDA

# Create conda environment
conda env create -f environment.yml
conda activate opencda

# Install in development mode
pip install -e .
```

### Run Your First Scenario

```bash
# Start CARLA server (in a separate terminal)
cd /path/to/CARLA
./CarlaUE4.sh

# Run a single vehicle scenario
python opencda.py -t single_2lanefree_carla -v 0.9.12

# Run a platoon scenario  
python opencda.py -t platoon_stability_2lanefree_carla -v 0.9.12
```

## Documentation

Full documentation is available at: [OpenCDA Documentation](https://opencda-documentation.readthedocs.io/en/latest/index.html)


## Architecture

OpenCDA consists of several key components:

```
OpenCDA
├── core/               # Core autonomous driving modules
│   ├── sensing/       # Perception and localization
│   ├── plan/          # Behavior planning and local planning
│   ├── actuation/     # Vehicle control
│   └── application/   # Platooning and other applications
├── customize/         # Customization examples
├── scenario_testing/  # Testing framework
└── co_simulation/     # CARLA-SUMO integration
```

## Use Cases

- **Research**: Test cooperative driving algorithms in realistic scenarios
- **Education**: Learn about autonomous driving system architecture
- **Development**: Prototype and validate CDA applications
- **Benchmarking**: Compare different planning and control strategies

## License

OpenCDA is released under the MIT License.

**Note**: This project is for non-commercial research and education purposes only. For commercial use, please contact the original authors.

## Related Projects

- [Original OpenCDA](https://github.com/ucla-mobility/OpenCDA) - UCLA Mobility Lab
- [CARLA Simulator](https://carla.org/) - Open-source simulator for autonomous driving
- [SUMO](https://www.eclipse.org/sumo/) - Traffic simulation suite

Based on: OpenCDA by UCLA Mobility Lab