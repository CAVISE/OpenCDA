.. OpenCDA documentation master file

Welcome to OpenCDA's Documentation!
====================================

.. note::
    **This is a fork of the original** `OpenCDA project <https://github.com/ucla-mobility/OpenCDA>`_ 
    developed by UCLA Mobility Lab. We maintain compatibility with the original while providing 
    updates and improvements.

    **Original Authors**: UCLA Mobility Lab | **Original Paper**: `arXiv:2107.06260 <https://arxiv.org/abs/2107.06260>`_

Overview
--------

**OpenCDA** (Open Cooperative Driving Automation) is a comprehensive **research and engineering framework** for developing, 
testing, and evaluating cooperative driving automation systems. It integrates prototype cooperative driving automation 
(CDA; see `SAE J3216 <https://www.sae.org/standards/content/j3216_202005/>`_) pipelines with standard automated 
driving components.

OpenCDA is enabling researchers to rapidly prototype, simulate, and test CDA algorithms. The framework 
supports both task-specific evaluation (e.g., object detection accuracy) and pipeline-level assessment (e.g., traffic safety).

Key Features
------------

**Full-Stack CDA Pipeline**
   Complete autonomous driving stack from perception to control, with cooperative capabilities built-in.

**V2X Communication**
   Realistic vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) communication simulation with 
   latency and packet loss modeling.

**Multi-Platform Integration**
   - **CARLA** (0.9.15, 0.9.16): High-fidelity 3D simulation
   - **SUMO** (1.18.0+): Large-scale traffic simulation
   - **Co-simulation**: Combined CARLA + SUMO

**Modular Design**
   Easily replace any module (perception, planning, control) with custom implementations.

**Benchmark Scenarios**
   Pre-built scenarios for platooning, cooperative merging, urban navigation, and more.

**Research-Ready**
   Designed for academic research with extensive documentation, evaluation metrics, and data collection tools.

Collaboration
-------------

In collaboration with `U.S. DOT CDA Research <https://its.dot.gov/cda/>`_ and the 
`FHWA CARMA Program <https://highways.dot.gov/research/operations/CARMA>`_, OpenCDA supports 
**early-stage fundamental research** for cooperative driving automation. Through collaboration 
with CARMA Collaborative, OpenCDA interfaces with the `CARMA XiL tools <https://github.com/usdot-fhwa-stol/carma-simulation>`_ 
for advanced simulation testing of CDA features.

Quick Links
-----------

**Documentation**
   * :doc:`OpenCDA/docs/md_files/introduction` - Detailed overview of OpenCDA
   * :doc:`OpenCDA/docs/md_files/installation` - Installation instructions
   * :doc:`OpenCDA/docs/md_files/getstarted` - Quick start guide
   * :doc:`modules` - API reference

**External Resources**
   * `OpenCDA GitHub Repository <https://github.com/CAVISE/OpenCDA>`_
   * `Original OpenCDA <https://github.com/ucla-mobility/OpenCDA>`_
   * `CARLA Simulator <https://carla.org/>`_
   * `SUMO Traffic Simulator <https://eclipse.dev/sumo/>`_

**Getting Started**
   New to OpenCDA? Start with our :doc:`md_files/installation` guide, then try the 
   :doc:`md_files/getstarted` tutorial to run your first simulation!

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   md_files/introduction
   md_files/installation
   md_files/getstarted

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   md_files/logic_flow
   md_files/traffic_generation
   md_files/yaml_define

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   md_files/developer_tutorial
   md_files/customization
   md_files/codebase_structure

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   md_files/contributor
   changelog
   modules

Advanced Topics
---------------

**Cooperative Perception**
   OpenCDA enables data collection for multi-vehicle perception research. The framework 
   was used to create the **OPV2V dataset** (ICRA 2022), the first large-scale cooperative 
   perception benchmark.

**Platooning**
   Built-in support for CACC (Cooperative Adaptive Cruise Control), platoon formation, 
   joining, and splitting maneuvers.

**Custom Maps**
   Import custom maps from OpenDRIVE format or use CARLA's built-in maps (Town01-Town10).

**ML Integration**
   Integrate your deep learning models for perception, prediction, or decision making. 
   PyTorch integration included.

**Paper Link**: https://arxiv.org/abs/2107.06260

Related Publications
--------------------

**OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with V2V Communication** (ICRA 2022)
   Uses OpenCDA's data dumping functionality for cooperative perception research.
   `Paper <https://arxiv.org/abs/2109.07644>`_ | `Code <https://github.com/DerrickXuNu/OpenCOOD>`_

**V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer** (ECCV 2022)
   Powered by OpenCDA simulation framework.
   `Paper <https://arxiv.org/pdf/2203.10638.pdf>`_ | `Code <https://github.com/DerrickXuNu/v2x-vit>`_

**The OpenCDA Open-source Ecosystem for Cooperative Driving Automation Research** (IEEE T-IV 2023)
   Comprehensive overview of the OpenCDA ecosystem.
   `Paper <https://ieeexplore.ieee.org/document/10045043>`_

License
-------

OpenCDA is released under the MIT License (see LICENSE file for details).

.. important::
   OpenCDA is intended for **non-commercial research and educational use only**. 
   For commercial applications, please contact the original authors at UCLA Mobility Lab.

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

----

.. note::
   **Work in Progress**: OpenCDA is actively maintained and continuously improved. 
   Many features are being developed.

----