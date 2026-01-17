# OpenCDA Overview

> 📍 **Note:** This is a fork of the original OpenCDA project developed by UCLA Mobility Lab. This fork maintains compatibility with the original while providing updates and improvements.

## What is OpenCDA?

OpenCDA (Open Cooperative Driving Automation) is a comprehensive research and engineering framework for developing, testing, and evaluating cooperative driving automation (CDA) systems. Built on top of industry-standard simulators, OpenCDA bridges the gap between autonomous vehicle research and practical implementation.

### Core Purpose

Current autonomous driving simulation platforms primarily focus on single-vehicle intelligence. OpenCDA fills this critical gap by enabling:

- **Vehicle-to-Vehicle (V2V) Communication:** Test cooperative behaviors between multiple connected vehicles
- **Vehicle-to-Infrastructure (V2I) Communication:** Integrate infrastructure intelligence into driving scenarios
- **Full-Stack Development:** Complete pipeline from perception to control, all in Python
- **Realistic Testing:** High-fidelity simulation with CARLA and large-scale traffic with SUMO

## Major Components

OpenCDA consists of four integrated subsystems:

```
┌─────────────────────────────────────────────────────────────┐
│                   OpenCDA Ecosystem                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │  Cooperative     │◄────►│  Co-Simulation   │             │
│  │  Driving System  │      │     Tools        │             │
│  └──────────────────┘      └──────────────────┘             │
│          │                          │                       │
│          │                          │                       │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │  Scenario        │◄────►│  Data Manager &  │             │
│  │  Manager         │      │  Repository      │             │
│  └──────────────────┘      └──────────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1. Cooperative Driving System

The heart of OpenCDA - a complete autonomous driving stack with cooperative capabilities:

**Sensing Layer:**
- **Perception:** Camera-based object detection (YOLOv5), LiDAR processing, sensor fusion
- **Localization:** GPS/IMU fusion with Kalman filtering for accurate positioning

**Planning Layer:**
- **Behavior Planning:** Traffic rule compliance, lane change decisions, intersection handling
- **Local Planning:** Trajectory generation with collision avoidance
- **Cooperative Planning:** Platooning algorithms, cooperative merging, coordinated maneuvers

**Control Layer:**
- **Vehicle Control:** PID controllers for longitudinal and lateral control
- **Customizable Controllers:** Easy integration of advanced control algorithms

**Communication Layer:**
- **V2X Manager:** Handles all vehicle-to-vehicle and vehicle-to-infrastructure communication
- **Latency Simulation:** Realistic network delays and packet loss modeling

### 2. Co-Simulation Tools

OpenCDA integrates multiple simulation platforms:

**CARLA Simulator (Primary Graphics Engine):**
- Latest supported: CARLA 0.9.15 (UE4) and 0.10.0 (UE5)
- High-fidelity 3D environments with realistic sensors
- Physics-based vehicle dynamics
- Weather and lighting variations

**SUMO Integration (Traffic Simulation):**
- Latest supported: SUMO 1.25.0
- Large-scale traffic flow simulation
- Realistic driver behaviors
- Custom traffic patterns

**Flexible Modes:**
- **CARLA-only:** Focus on vehicle-level testing
- **SUMO-only:** Focus on traffic-level analysis
- **Co-simulation:** Combined high-fidelity and large-scale simulation

### 3. Scenario Manager

Powerful scenario creation and management:

- **YAML-based Configuration:** Easy scenario definition without code changes
- **Traffic Generation:** Automated background traffic with customizable behaviors
- **Spawn Management:** Flexible vehicle and platoon placement
- **OpenSCENARIO Support:** Industry-standard scenario description
- **Benchmark Scenarios:** Pre-built scenarios for standardized testing

### 4. Data Manager & Repository

Comprehensive data handling capabilities:

- **Data Collection:** Automated recording of sensor data, trajectories, and events
- **Dataset Generation:** Create datasets for machine learning training
- **Replay Functionality:** Review and analyze recorded scenarios
- **Cooperative Perception Data:** Multi-vehicle perception datasets (like OPV2V)

## Key Features

### Why Choose OpenCDA?

| Feature | Description |
|---------|-------------|
|  **Pure Python** | Entire stack in Python for rapid prototyping and easy debugging |
|  **Highly Modular** | Replace any component with custom implementations |
|  **Cooperation-First** | Built from ground up for V2V and V2I communication |
|  **Comprehensive Testing** | Full pipeline evaluation from perception to control |
|  **Research-Friendly** | Designed for academic research with extensive documentation |
|  **Industry-Ready** | Interfaces with CARMA XiL tools for advanced testing |
|  **Open Source** | Fully open source with active community |

### Advanced Capabilities

**Platooning:**
- CACC (Cooperative Adaptive Cruise Control)
- Platoon formation, joining, and splitting
- Leader-follower coordination
- Stability analysis tools

**Cooperative Perception:**
- Multi-vehicle sensor fusion
- Object tracking across vehicles
- Occlusion handling through cooperation
- Bandwidth-efficient data sharing

**Scenario Testing:**
- Highway scenarios (2-lane, multi-lane)
- Urban scenarios (Town06, Town10)
- Custom map support
- Intersection coordination

**Evaluation Metrics:**
- Safety metrics (TTC, collision detection)
- Efficiency metrics (fuel consumption, travel time)
- Comfort metrics (acceleration, jerk)
- Cooperation metrics (communication overhead, latency)

## Use Cases

### Research Applications

- **Cooperative Driving Algorithms:** Develop and test V2V/V2I coordination strategies
- **Perception Fusion:** Multi-vehicle perception and tracking algorithms
- **Planning Strategies:** Cooperative lane changes, merging, and intersection negotiation
- **Communication Protocols:** Test V2X communication under various conditions

### Educational Applications

- **Autonomous Systems Education:** Hands-on learning with complete AV stack
- **Traffic Engineering:** Study cooperative traffic flow optimization
- **Software Engineering:** Learn modular system design and integration

### Industry Applications

- **Algorithm Validation:** Test algorithms before real-world deployment
- **Safety Assessment:** Evaluate edge cases and failure modes
- **Standard Development:** Support CDA standard development (SAE J3216)
- **Integration Testing:** Test with CARMA and other platforms

## Scientific Foundation

OpenCDA is designed to support initial-stage fundamental research for CDA:

- Developed in collaboration with U.S. DOT CDA Research
- Interfaces with FHWA CARMA Program
- Compatible with CARMA XiL tools
- Supports SAE J3216 CDA standards

## Related Ecosystems

OpenCDA is part of a larger research ecosystem:

**OPV2V Dataset (ICRA 2022):**
- First large-scale dataset for cooperative perception
- 73 scenarios with multiple connected vehicles
- LiDAR and camera data from all vehicles
- [Paper](https://arxiv.org/abs/2109.07644) | [Code](https://github.com/DerrickXuNu/OpenCOOD)

**V2X-ViT (ECCV 2022):**
- State-of-the-art cooperative perception with Vision Transformers
- Powered by OpenCDA simulation framework
- [Paper](https://arxiv.org/abs/2203.10638) | [Code](https://github.com/DerrickXuNu/v2x-vit)

## Vision

OpenCDA aims to accelerate CDA research by providing:

- **Accessible Tools:** Lower barriers to entry for CDA research
- **Reproducible Results:** Standardized benchmarks and scenarios
- **Community Building:** Foster collaboration across institutions
- **Industry Bridge:** Connect research with practical deployment

---

**Original Project:** [UCLA Mobility Lab OpenCDA](https://github.com/ucla-mobility/OpenCDA)  
**Documentation:** [OpenCDA Docs](https://opencda-documentation.readthedocs.io/)  
**Forum:** [GitHub Discussions](https://github.com/ucla-mobility/OpenCDA/discussions)  
**Issues:** Report bugs and request features via [GitHub Issues](https://github.com/ucla-mobility/OpenCDA/issues)