# AdvCP: Adversarial Cooperative Perception

This package implements the AdvCP family of adversarial attacks against
cooperative perception in OpenCDA. The attacks are integrated into the
inference pipeline so that any cooperative-perception scenario can be
re-run with one or more compromised CAVs (Connected Autonomous Vehicles)
acting as attackers.

The reference paper and the original implementation that this package is
modelled after live at <https://github.com/zqzqz/AdvCollaborativePerception>.

## Scope

AdvCP supports two attack modes against three cooperative-perception
fusion strategies:

| Fusion          | Spoofing | Removal |
|-----------------|----------|---------|
| Early           | yes      | yes     |
| Intermediate    | yes      | yes     |
| Late            | yes      | yes     |

Each combination supports a single attacker as well as multiple
attackers acting jointly within the same tick.

- Spoofing injects a fake object into the cooperative perception output:
  the ego vehicle "sees" something that does not exist.
- Removal suppresses an existing object from the cooperative perception
  output: the ego vehicle fails to see something that does exist.

## Module map

```
opencda/core/attack/advcp/
    __init__.py                       Public re-exports.
    types.py                          Typed dictionaries, dataclasses,
                                      and type aliases shared by every
                                      attack module.
    attack_helper.py                  Utilities used by every fusion
                                      strategy: config validation,
                                      attacker resolution, target box
                                      construction, mesh helpers.
    early_fusion_attack.py            Early-fusion attack: rewrites the
                                      attacker LiDAR point cloud before
                                      the cooperative pipeline.
    intermediate_fusion_attack.py     Intermediate-fusion attack:
                                      perturbs the attacker spatial
                                      feature map produced by the
                                      backbone (gradient-based, Adam).
    late_fusion_attack.py             Late-fusion attack: rewrites the
                                      attacker per-CAV detections
                                      before late-stage NMS.
    adv_coperception_model_manager.py Top-level manager that wires the
                                      attack into the cooperative
                                      perception inference pipeline.
```

## High-level data flow

The AdvCP pipeline starts from two configuration sources:

1. the scenario YAML, which defines the simulation setup;
2. the AdvCP YAML config, which defines the attack mode, attackers, targets, and attack-specific parameters.

These configuration files are loaded by `AdvCoperceptionModelManager`.

The manager is responsible for the top-level AdvCP orchestration. It performs three main tasks:

1. loads the AdvCP attack configuration;
2. validates the configured attacker vehicles;
3. dispatches execution to the correct inference hook depending on the cooperative perception fusion mode.

The attack implementation then modifies the cooperative perception pipeline at the appropriate representation level:

1. early fusion attacks rewrite the LiDAR `npt.NDArray` before model inference;
2. intermediate fusion attacks perturb spatial feature maps inside the model pipeline;
3. late fusion attacks rewrite per-CAV detection results before final fusion.

After the attack-specific manipulation is applied, the pipeline produces an `AdvCPAttackResult`.

The result contains:

1. `pred_box_tensor` — final predicted bounding boxes;
2. `pred_score` — confidence scores for predicted boxes;
3. `gt_box_tensor` — ground-truth bounding boxes;
4. `AdvCPVisualizationContext` — metadata required by visualization and metrics.

The `AdvCPAttackResult` is then passed to the downstream metrics framework and visualizer.

The manager always returns an `AdvCPAttackResult`, regardless of the fusion mode. The visualization context carries the original target boxes and the list of effective attacker IDs. This allows downstream components to:

1. color predictions and attack-related objects in visualizations;
2. compute attack success rate;
3. compute target confidence;
4. compute attacker/benign visibility statistics;
5. evaluate other attack-related metrics consistently across fusion modes.

## Single attacker vs. multiple attackers

All three fusion modes accept a list of attacker ids in the AdvCP
config. The handling differs by fusion strategy:

- Early fusion iterates over each present attacker independently and
  rewrites that attacker's LiDAR point cloud in shared memory. The
  cooperative perception backbone then consumes the modified point
  clouds without further coordination.
- Intermediate fusion uses a dedicated joint optimization path
  (`_optimize_joint`) when more than one attacker is present in the
  current batch. A single Adam optimizer holds all per-attacker
  perturbation tensors and the forward pass injects all of them at
  once. This is required because the cooperative backbone fuses the
  attackers' features together; optimizing them independently would
  ignore the interaction and produce weaker attacks.
- Late fusion iterates over each present attacker independently and
  rewrites that attacker's predicted boxes (and scores) before the
  ego-side NMS.

## Two-mode configuration

Both spoofing and removal share the same config format and the same
target-box machinery. The differences live inside each fusion module:

- Early spoofing builds adversarial points by ray-tracing a synthetic
  car mesh placed at the configured target position. Removal
  replaces points already inside the target box with points sampled
  from a wall mesh or an adversarial shape mesh.
- Intermediate spoofing minimises the negative log-likelihood of high
  confidence on the target location (encourages a detection there).
  Removal minimises the negative log-likelihood of low confidence on
  the target location (suppresses any existing detection there).
- Late spoofing appends fake detections directly to the attacker's
  per-CAV output. Removal filters out any of the attacker's
  detections that overlap with the target box by IoU above a
  threshold.

## Glossary

The following terms recur throughout the module. Their meaning is not
always obvious from the surrounding code.

- attacker: A CAV whose data flow is being manipulated by AdvCP. The
  attacker is identified by `attacker_id` and is configured under
  `attacker_ids` in the AdvCP YAML.
- attacker id: A string identifier matching the agent id in the
  scenario (for example `cav-1`). When the attacker is the ego CAV
  itself, the late-fusion path internally maps it to the literal
  string `"ego"` because that is the key OpenCOOD uses inside the
  collated batch.
- ego: The CAV whose perception is the perspective of interest. The
  cooperative perception output is reported in the ego frame. The ego
  CAV may itself be an attacker.
- target / target box: The bounding box that the attacker is trying
  to inject (spoofing) or suppress (removal). Configured in the AdvCP
  YAML under `boxes`.
- mode: One of `"spoofing"` or `"removal"`. Selected by the AdvCP
  YAML under `mode`.
- memory_data: A nested structure of per-batch, per-CAV snapshots
  used by OpenCOOD as the input for cooperative perception. AdvCP
  reads from it (to recover ego/attacker poses and LiDAR clouds) and,
  for early fusion, writes a modified copy back through
  `dataset.update_database`.
- init_perturbation: For intermediate fusion, an optional warm start
  for the perturbation tensor. When `online: true` is configured,
  the best perturbation found at tick `t-1` is reused as the initial
  value at tick `t`, accelerating convergence.
- perturbation: A torch tensor with the same shape as a patch of the
  attacker's spatial feature map. Added to that patch before the
  fused detection head runs. Optimized by Adam under an L-infinity
  ball of radius `max_perturb`.
- center: Pixel coordinates inside the attacker's spatial feature
  map, computed from the target box's world-space centre via the
  voxel grid. Drives where the perturbation is applied.
- visualization context (`AdvCPVisualizationContext`): A small
  dataclass carried alongside the prediction tensors. Holds the mode,
  the list of effective attacker ids, and the target box tensor for
  the current tick. Consumed by the visualizer (to color boxes) and
  by ASR / confidence metrics.
- density: An integer in `{0, 1, 2, 3}` derived from the
  `density` config value (`"replace"`, `"dense_a"`, `"dense_all"`,
  `"sampled"`). Controls how many synthetic points the early-fusion
  attack writes per target box.
- advshape: Boolean flag. When enabled, the early-fusion removal
  attack uses an adversarial-shape mesh instead of a plain
  bounding-box wall mesh.
- AdvCP YAML config: User-facing configuration that selects mode,
  fusion-mode-specific parameters, attacker ids, and target boxes.
  Resolved by `AdvCoperceptionModelManager.load_config`.

## Configuration

A typical AdvCP YAML config looks like:

```yaml
mode: spoofing
attacker_ids: ["cav-1"]
boxes:
  - relative: [5.0, 0.0, 0.0, 0.0, 90.0, 0.0]
default_size: [4.5, 2.0, 1.6]
density: sampled
sync: true
init: true
online: true
step: 25
max_perturb: 10.0
lr: 0.05
feature_size: 10
```

- `boxes` accepts entries with either a `relative` pose (relative to
  ego) or an `absolute` world-frame pose, plus an optional `size`
  (length, width, height). Internally each entry resolves to a
  per-attacker LiDAR-frame box with shape `(7,)`:
  `[x, y, z, length, width, height, yaw]`.
- `step`, `lr`, `max_perturb`, `feature_size` are intermediate-fusion
  optimization parameters.
- `density` and `dense_distance` are early-fusion parameters.
- `sync`, `init`, and `online` control how state from previous ticks
  is reused (intermediate fusion only).

The full list of recognised keys is enumerated in
`AdvCoperceptionModelManager.load_config`.

## Entry points

- Top-level: `AdvCoperceptionModelManager` (subclass of the standard
  `CoperceptionModelManager`). Use this in place of the regular
  manager to enable AdvCP for a scenario.
- Per-fusion attack runners:
  - `AdvCoperceptionEarlyFusionAttack.run`
  - `AdvCoperceptionIntermediateFusionAttack.run`
  - `AdvCoperceptionLateFusionAttack.run`
  Each `run` returns an `AdvCPAttackResult`.

## Related metrics

The attack populates `AdvCPVisualizationContext.fake_box_tensor`
(spoofing) or `removed_box_tensor` (removal). Two metrics in
`opencda/metrics_tools/metrics/` consume those tensors:

- `attack_success_rate.AttackSuccessRateMetric`: For removal, counts
  how many GT objects whose centroid lies inside the removal zone
  are no longer detected. For spoofing, counts how many unique
  configured spoof boxes are detected by the cooperative output.
- `attacker_target_confidence.AttackerTargetConfidenceMetric`:
  Reports the cooperative model's confidence on each target box.

Both metrics deduplicate target boxes coming from multiple attackers
that target the same physical object.

## Tests

Unit tests live in `opencda/core/attack/advcp/test/test_advcp.py`.
They cover config validation, attacker resolution, the fallback paths
when configured attackers are missing from the current batch, and
the dispatch between single-attacker and joint optimization for
intermediate fusion.
