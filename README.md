OpenCDA (Fork)

🔗 This is a fork of the original OpenCDA project
Original authors: UCLA Mobility Lab
Original paper: OpenCDA: An Open Cooperative Driving Automation Framework Integrated with Co-simulation


OpenCDA is a SIMULATION tool integrated with a prototype cooperative driving automation (CDA; see SAE J3216) pipeline as well as regular automated driving components (e.g., perception, localization, planning, control). The tool integrates automated driving simulation (CARLA), traffic simulation (SUMO), and Co-simulation.
Key Features:

  Comprehensive CDA Pipeline: Full stack from perception to control
  V2X Communication: Vehicle-to-vehicle and vehicle-to-infrastructure
  CARLA Integration: High-fidelity 3D simulation environment
  SUMO Co-simulation: Large-scale traffic flow simulation
  Modular Design: Easy customization and extension
  Platooning Support: Built-in cooperative platooning algorithms
  Testing Framework: Scenario-based testing and evaluation

  Quick Start
Prerequisites

Python 3.10+
CARLA 0.9.Х (recommended 0.9.16)
(Optional) SUMO 1.10+

Installation
bash# Clone the repository
git clone [https://github.com/CAVISE/OpenCDA]
cd OpenCDA

# Create conda environment
conda env create -f environment.yml
conda activate opencda

# Install in development mode
pip install -e .
For detailed installation instructions, see Installation Guide.
Run Your First Scenario
bash# Start CARLA server (in a separate terminal)
cd /path/to/CARLA
./CarlaUE4.sh

# Run a single vehicle scenario
python opencda.py -t single_2lanefree_carla -v 0.9.12

# Run a platoon scenario  
python opencda.py -t platoon_stability_2lanefree_carla -v 0.9.12
 Documentation
Full documentation is available at: OpenCDA Documentation
Quick links:

Getting Started Guide
Developer Tutorial
Customization Guide
API Reference

  Architecture
OpenCDA consists of several key components:
OpenCDA
├── core/               # Core autonomous driving modules
│   ├── sensing/       # Perception and localization
│   ├── plan/          # Behavior planning and local planning
│   ├── actuation/     # Vehicle control
│   └── application/   # Platooning and other applications
├── customize/         # Customization examples
├── scenario_testing/  # Testing framework
└── co_simulation/     # CARLA-SUMO integration
   Use Cases

Research: Test cooperative driving algorithms in realistic scenarios
Education: Learn about autonomous driving system architecture
Development: Prototype and validate CDA applications
Benchmarking: Compare different planning and control strategies

   Contributing
Contributions are welcome! Please see our Contributing Guidelines.
For the original project's contribution history, see the original repository.
   Citation
If you use OpenCDA in your research, please cite the original paper:
bibtex@inproceedings{xu2021opencda,
  title={OpenCDA: an open cooperative driving automation framework integrated with co-simulation},
  author={Xu, Runsheng and Guo, Yi and Han, Xu and Xia, Xin and Xiang, Hao and Ma, Jiaqi},
  booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)},
  pages={1155--1162},
  year={2021},
  organization={IEEE}
}
   License
OpenCDA is released under the MIT License.
Note: This project is for non-commercial research and education purposes only. For commercial use, please contact the original authors.
🔗 Related Projects

Original OpenCDA - UCLA Mobility Lab
CARLA Simulator - Open-source simulator for autonomous driving
SUMO - Traffic simulation suite

💬 Support

Issues: Report bugs and request features via GitHub Issues
Discussions: Join conversations in GitHub Discussions
Original Project: For questions about the original OpenCDA, see UCLA's repository


Maintainers: [Your Team Info Here]
Based on: OpenCDA by UCLA Mobility Lab