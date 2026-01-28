# OpenCDA (Fork)

> **This is a fork of the original OpenCDA project**
> Original authors: UCLA Mobility Lab
> Original paper: [OpenCDA: An Open Cooperative Driving Automation Framework Integrated with Co-simulation](https://arxiv.org/abs/2107.06260)

---

OpenCDA is a open co-simulation-based framework integrated with a prototype cooperative driving automation (CDA; see SAE J3216) pipeline as well as regular automated driving components (e.g., perception, localization, planning, control). The tool integrates automated driving simulation (CARLA), traffic simulation (SUMO), Co-simulation, detection framework (OpenCOOD) and V2X simulation framework (Artery).

## Key Features

- **Comprehensive CDA Pipeline**: Full stack from perception to control
- **V2X Communication**: Vehicle-to-vehicle and vehicle-to-infrastructure
- **CARLA Integration**: High-fidelity 3D simulation environment
- **SUMO Co-simulation**: Large-scale traffic flow simulation
- **Modular Design**: Easy customization and extension
- **Platooning Support**: Built-in cooperative platooning algorithms
- **Cooperative Perception Support**: Built-in cooperative perception algorithms
- **Testing Framework**: Scenario-based testing and evaluation

## Documentation

Full fork documentation is not available yet...
Full original documentation is available at: [OpenCDA Documentation](https://opencda-documentation.readthedocs.io/en/latest/index.html)


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

## License

OpenCDA is released under the MIT License.

**Note**: This project is for non-commercial research and education purposes only. For commercial use, please contact the original authors.

## Related Projects

- [Original OpenCDA](https://github.com/ucla-mobility/OpenCDA) - UCLA Mobility Lab
- [Original OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) - detection framework
- [CARLA Simulator](https://carla.org/) - Open-source simulator for autonomous driving
- [SUMO](https://www.eclipse.org/sumo/) - Traffic simulation suite
- [Artery](https://github.com/riebl/artery) - V2X simulation framework

Based on: OpenCDA by UCLA Mobility Lab
