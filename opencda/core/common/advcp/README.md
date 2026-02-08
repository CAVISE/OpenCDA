## Comprehensive Description of opencda/core/common/advcp Module

### 1. Source of Batches and Simulation Context

**Batches come from OpenCOOD**, which is an external collaborative perception framework included as a submodule (`OpenCOOD/` directory). The data flow is:

- **Dataset**: [`opencda/core/common/advcp/mvp/data/opv2v_dataset.py`](opencda/core/common/advcp/mvp/data/opv2v_dataset.py:1) loads OPV2V dataset (a collaborative perception dataset for autonomous driving)
- **Data Structure**: Multi-vehicle, multi-frame cases where each vehicle has:
  - `lidar`: Point cloud data (N×4 or N×5 array: x, y, z, intensity, optional)
  - `lidar_pose`: 6-DOF pose [x, y, z, roll, yaw, pitch]
  - `gt_bboxes`: Ground truth bounding boxes (7-D: x, y, z, l, w, h, yaw)
  - `object_ids`: Unique IDs for objects
  - `camera` data (optional)
  - `occupied_areas` / `free_areas` (polygon areas for defense)
  - `pred_bboxes` / `pred_scores` (after perception inference)

- **Integration**: [`CoperceptionModelManager`](opencda/core/common/coperception_model_manager.py:20) creates the OpenCOOD dataset via `build_dataset()` and wraps it in a DataLoader. When `with_advcp=True`, it initializes [`AdvCPManager`](opencda/core/common/advcp/advcp_manager.py:28) which intercepts the perception pipeline.

### 2. Purpose of Each File in mvp/ Submodule

#### **Core Configuration**
- [`mvp/config.py`](opencda/core/common/advcp/mvp/config.py:1): Defines root paths (`data_root`, `model_root`, `third_party_root`), 3D model paths, class maps, color maps, and scenario-to-Carla-map mappings.

#### **Data Layer**
- [`mvp/data/dataset.py`](opencda/core/common/advcp/mvp/data/dataset.py:1): Abstract base class with `load_feature()` utility.
- [`mvp/data/opv2v_dataset.py`](opencda/core/common/advcp/mvp/data/opv2v_dataset.py:14): Main dataset class. Loads OPV2V metadata from pickle files, builds cases:
  - `single_vehicle`: Individual vehicle frames
  - `multi_vehicle`: Single frame with multiple vehicles
  - `multi_frame`: Sequential frames (default 10 frames)
  - `scenario`: Full scenario sequences
  - Also pre-builds attack configurations (`_build_attacks()`)
- [`mvp/data/util.py`](opencda/core/common/advcp/mvp/data/util.py:1): Coordinate transformations:
  - `pcd_sensor_to_map()` / `pcd_map_to_sensor()`: Point cloud coordinate transforms
  - `bbox_sensor_to_map()` / `bbox_map_to_sensor()`: Bounding box transforms
  - `pose_to_transformation()`: 6-DOF pose to 4×4 matrix
  - `read_pcd()`: Loads Open3D point clouds
  - `rotation_matrix()`: Creates rotation matrices

#### **Perception Layer**
- [`mvp/perception/perception.py`](opencda/core/common/advcp/mvp/perception/perception.py:1): Abstract base class defining interface.
- [`mvp/perception/opencood_perception.py`](opencda/core/common/advcp/mvp/perception/opencood_perception.py:28): Concrete implementation wrapping OpenCOOD models:
  - Supports: `pixor`, `voxelnet`, `second`, `pointpillar`, `v2vnet`, `fpvrcnn`
  - Fusion methods: `early`, `intermediate`, `late`
  - `run()`: Processes multi-vehicle case, returns pred_bboxes, pred_scores
  - `attack_late()`: Special method for late-fusion attacks (modifies predictions directly)
  - Loads model from `model_root/OpenCOOD/` and dataset from `data_root/OPV2V/`

#### **Attack Layer**
- [`mvp/attack/attacker.py`](opencda/core/common/advcp/mvp/attack/attacker.py:9): Base class with benchmark building/loading utilities. Uses `apply_ray_tracing()` to modify lidar points.
- [`mvp/attack/lidar_remove_early_attacker.py`](opencda/core/common/advcp/mvp/attack/lidar_remove_early_attacker.py:14): **Early fusion attack** - removes points belonging to a target object by ray-tracing with wall meshes. Modes:
  - `dense=1`: DenseA (inject points along rays)
  - `dense=2`: DenseAll (inject from all vehicle perspectives)
  - `dense=3`: Sampled (sparse injection)
  - `sync=1`: Async mode (temporal consistency)
  - `advshape>0`: Uses adversarial mesh perturbation
- [`mvp/attack/lidar_spoof_early_attacker.py`](opencda/core/common/advcp/mvp/attack/lidar_spoof_early_attacker.py:13): **Early fusion attack** - spoofs fake object by injecting points from a 3D car model mesh.
- [`mvp/attack/lidar_remove_intermediate_attacker.py`](opencda/core/common/advcp/mvp/attack/lidar_remove_intermediate_attacker.py:7): **Intermediate fusion attack** - optimizes perturbation in feature space using gradient-based methods (`step` iterations, `online` learning).
- [`mvp/attack/lidar_remove_late_attacker.py`](opencda/core/common/advcp/mvp/attack/lidar_remove_late_attacker.py:6): **Late fusion attack** - modifies final bounding box predictions directly via `perception.attack_late()`.
- [`mvp/attack/adv_shape_attacker.py`](opencda/core/common/advcp/mvp/attack/adv_shape_attacker.py:12): **Adversarial shape attack** - uses genetic algorithm (pygad) to optimize mesh vertex perturbations for universal attack.

#### **Defense Layer**
- [`mvp/defense/defender.py`](opencda/core/common/advcp/mvp/defense/defender.py:1): Base class.
- [`mvp/defense/perception_defender.py`](opencda/core/common/advcp/mvp/defense/perception_defender.py:13): **CAD (Consistency-Aware Defense)**:
  - Loads lane area maps from `data_root/carla/*_lane_areas.pkl`
  - `run()`: Merges occupied/free areas from all vehicles, checks consistency between predicted bboxes and occupancy maps
  - Detects spoofed (in free space) and removed (occupied but not detected) objects
  - Returns defense score and metrics per vehicle
- [`mvp/defense/detection_util.py`](opencda/core/common/advcp/mvp/defense/detection_util.py:1): Utilities for segmentation-based detection.
- [`mvp/defense/sync_handler.py`](opencda/core/common/advcp/mvp/defense/sync_handler.py:40): Temporal synchronization for multi-frame defense using object tracking (`Tracker`). Predicts point cloud positions across frames.

#### **Evaluation Layer**
- [`mvp/evaluate/accuracy.py`](opencda/core/common/advcp/mvp/evaluate/accuracy.py:5): Computes TP/FP/F1 by comparing predicted vs ground truth bboxes with IoU threshold.
- [`mvp/evaluate/detection.py`](opencda/core/common/advcp/mvp/evaluate/detection.py:1): Likely detection-specific metrics.

#### **Visualization Layer**
- [`mvp/visualize/general.py`](opencda/core/common/advcp/mvp/visualize/general.py:1): General utilities:
  - `draw_matplotlib()` / `draw_open3d()`: Render point clouds and bboxes
  - `draw_multi_vehicle_case()`: Multi-vehicle visualization with coordinate transforms
  - `draw_polygons()`: Draw occupied/free areas
  - `draw_pointclouds()`, `draw_bboxes_2d()`, `draw_trajectories()`
- [`mvp/visualize/attack.py`](opencda/core/common/advcp/mvp/visualize/attack.py:9): `draw_attack()` compares normal vs attack cases, shows attacker/victim positions, gt/pred bboxes, and attack target location.
- [`mvp/visualize/defense.py`](opencda/core/common/advcp/mvp/visualize/defense.py:1): `visualize_defense()` shows occupied/free areas, ego areas, pred/gt bboxes, and error regions.
- [`mvp/visualize/evaluate.py`](opencda/core/common/advcp/mvp/visualize/evaluate.py:1): Evaluation result visualization.

#### **Tools**
- [`mvp/tools/ray_tracing.py`](opencda/core/common/advcp/mvp/tools/ray_tracing.py:1): Core attack primitive:
  - `get_model_mesh()`: Loads and transforms 3D car model
  - `get_wall_mesh()`: Creates wall meshes for point removal
  - `ray_intersection()`: Open3D raycasting to find lidar-mesh intersections (where points should be removed/spoofed)
- [`mvp/tools/ground_detection.py`](opencda/core/common/advcp/mvp/tools/ground_detection.py:1): Ground plane segmentation (RANSAC).
- [`mvp/tools/lidar_seg.py`](opencda/core/common/advcp/mvp/tools/lidar_seg.py:1): Lidar segmentation for object extraction.
- [`mvp/tools/object_tracking.py`](opencda/core/common/advcp/mvp/tools/object_tracking.py:1): Multi-object tracking for temporal defense.
- [`mvp/tools/polygon_space.py`](opencda/core/common/advcp/mvp/tools/polygon_space.py:1): Occupancy map generation (`get_occupied_space()`, `get_free_space()`).
- [`mvp/tools/sensor_calib.py`](opencda/core/common/advcp/mvp/tools/sensor_calib.py:1): Sensor coordinate transformations.
- [`mvp/tools/iou.py`](opencda/core/common/advcp/mvp/tools/iou.py:1): 3D IoU calculation.
- [`mvp/tools/cluster.py`](opencda/core/common/advcp/mvp/tools/cluster.py:1): Point clustering (DBSCAN).
- [`mvp/tools/squeezeseg/`](opencda/core/common/advcp/mvp/tools/squeezeseg/): Semantic segmentation model.

### 3. Data Flow and Integration with coperception_model_manager.py

**Flow**:
```
CoperceptionModelManager.make_prediction(tick_number)
  ├─ Load batch from DataLoader (OpenCOOD dataset)
  ├─ If AdvCP enabled:
  │   └─ AdvCPManager.process_tick(tick_number)
  │       ├─ Get original predictions via inference_utils (early/intermediate/late)
  │       ├─ AdvCPManager._apply_attack():
  │       │   ├─ Select attacker vehicles (based on attackers_ratio)
  │       │   ├─ For each attacker:
  │       │   │   └─ Attacker.run() modifies multi_frame_case
  │       │   │       └─ Returns modified lidar data (point removal/spoofing)
  │       │   └─ Returns modified_data (dict[vehicle_id][pred_bboxes, pred_scores])
  │       ├─ AdvCPManager._apply_defense() (if enabled):
  │       │   └─ Defender.run() checks consistency, returns defended_data
  │       └─ Return modified_data, defense_score, defense_metrics
  │   └─ Use AdvCP's pred_bboxes/pred_scores for evaluation/visualization
  └─ Else: Use original predictions
```

**Key Points**:
- [`coperception_model_manager.py`](opencda/core/common/coperception_model_manager.py:114) calls `self.advcp_manager.process_tick(i)` passing tick index
- AdvCP manager internally calls `coperception_manager.make_prediction()` recursively to get original predictions (line 145), then applies attacks/defenses on top
- Modified data structure: `Dict[vehicle_id, Dict]` containing `pred_bboxes`, `pred_scores`, and optionally modified `lidar`
- All files in `mvp/` participate when AdvCP is enabled; they are **not** used by default in `coperception_model_manager.py` unless `with_advcp=True`

### 4. Visualization

**What is visualized**:
- **Point clouds**: Lidar points from all vehicles (transformed to map coordinates)
- **Ground truth bboxes**: Green 3D boxes
- **Predicted bboxes**: Red 3D boxes
- **Occupied/free areas**: Filled polygons (defense visualization)
- **Ego vehicle areas**: Outlined polygons
- **Attack effects**: 
  - Spoofed object position (red marker)
  - Removed object location
  - Attacker (red dot) and victim (green dot) vehicle positions
- **Defense metrics**: Error areas (spoofed detections in free space, missing detections in occupied space)

**Visualization backends**:
- **Matplotlib**: 2D top-down plots (saved as PNG)
- **Open3D**: Interactive 3D visualization
- **Output**: Controlled by `--save-vis` / `--show-vis` flags in [`coperception_model_manager.py`](opencda/core/common/coperception_model_manager.py:181)

**Saving**: Visualizations saved to `simulation_output/coperception/vis_3d/` and `vis_bev/` with tick-numbered filenames.

### 5. Summary

The **AdvCP (Advanced Collaborative Perception)** module is a comprehensive framework for **adversarial attacks and defenses** on collaborative perception systems. It:

- Integrates with OpenCOOD for model inference
- Supports early/intermediate/late fusion attacks
- Implements point removal, point spoofing, and adversarial shape attacks via ray-tracing and gradient-based optimization
- Provides CAD defense using geometric consistency checks
- Includes full visualization and evaluation tooling
- All components are in `mvp/` submodule and are activated only when `with_advcp=True` in configuration

The attacks modify lidar point clouds at the **sensor level** (early), **feature level** (intermediate), or **detection level** (late), while defense operates on **occupancy maps** derived from fused perceptions.