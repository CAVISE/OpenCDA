# AdvCP Module - Test Command Examples

This document provides various command examples to test the AdvCP (Advanced Collaborative Perception) module with different configurations, attack types, and defense mechanisms.

## Basic Test Commands

### 1. Simple Early Attack (No Defense)
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --attack-type lidar_remove_early \
  --attackers-ratio 0.3
```

### 2. Simple Intermediate Attack (No Defense)
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25
```

### 3. Simple Late Attack (No Defense)
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method late \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-late-opv2v-30 \
  --attack-type lidar_remove_late \
  --attackers-ratio 0.2
```

## Attack Types Testing

### 4. Early Remove Attack
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --attack-type lidar_remove_early \
  --attackers-ratio 0.3 \
  --attack-target random
```

### 5. Early Spoof Attack
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --attack-type lidar_spoof_early \
  --attackers-ratio 0.3 \
  --attack-target specific_vehicle
```

### 6. Intermediate Remove Attack
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_remove_intermediate \
  --attackers-ratio 0.25
```

### 7. Intermediate Spoof Attack
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --attack-target all_non_attackers
```

### 8. Late Remove Attack
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method late \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-late-opv2v-30 \
  --attack-type lidar_remove_late \
  --attackers-ratio 0.2
```

### 9. Late Spoof Attack
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method late \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-late-opv2v-30 \
  --attack-type lidar_spoof_late \
  --attackers-ratio 0.2
```

### 10. Advanced Shape Attack
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type adv_shape \
  --attackers-ratio 0.25
```

## Defense Testing

### 11. Attack with CAD Defense (Early)
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --attack-type lidar_remove_early \
  --attackers-ratio 0.3 \
  --apply-cad-defense \
  --defense-threshold 0.7
```

### 12. Attack with CAD Defense (Intermediate)
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --apply-cad-defense \
  --defense-threshold 0.8
```

### 13. Attack with CAD Defense (Late)
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method late \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-late-opv2v-30 \
  --attack-type lidar_remove_late \
  --attackers-ratio 0.2 \
  --apply-cad-defense \
  --defense-threshold 0.6
```

### 14. Defense Only (No Attack)
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --apply-cad-defense \
  --defense-threshold 0.7
```

## Attackers Ratio Variations

### 15. Low Attackers Ratio (10%)
```bash
python opencda.py \
  -t 4cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.1
```

### 16. Medium Attackers Ratio (30%)
```bash
python opencda.py \
  -t 4cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.3
```

### 17. High Attackers Ratio (50%)
```bash
python opencda.py \
  -t 4cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.5
```

### 18. Very High Attackers Ratio (80%)
```bash
python opencda.py \
  -t 4cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.8
```

## Target Selection Strategies

### 19. Random Target Selection
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --attack-target random
```

### 20. Specific Vehicle Target
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --attack-target specific_vehicle
```

### 21. All Non-Attackers Target
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --attack-target all_non_attackers
```

## Visualization Tests

### 22. Basic Visualization (Matplotlib 2D)
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --attack-type lidar_remove_early \
  --attackers-ratio 0.3 \
  --advcp-vis \
  --advcp-vis-mode matplotlib \
  --advcp-vis-save \
  --advcp-vis-show
```

### 23. 3D Visualization (Open3D)
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --advcp-vis \
  --advcp-vis-mode open3d \
  --advcp-vis-save
```

### 24. Both 2D and 3D Visualization
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method late \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-late-opv2v-30 \
  --attack-type lidar_remove_late \
  --attackers-ratio 0.2 \
  --advcp-vis \
  --advcp-vis-mode both \
  --advcp-vis-save \
  --advcp-vis-types attack defense evaluation
```

### 25. Visualization with Custom Output Directory
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --attack-type lidar_spoof_early \
  --attackers-ratio 0.3 \
  --advcp-vis \
  --advcp-vis-mode matplotlib \
  --advcp-vis-save \
  --advcp-vis-output-dir /tmp/advcp_test_output \
  --advcp-vis-types attack defense tracking roc
```

## Custom Configuration File

### 26. Using Custom AdvCP Config
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --advcp-config /path/to/custom_advcp_config.yaml \
  --attackers-ratio 0.4 \
  --attack-target random
```

### 27. Custom Config with All Options
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --advcp-config opencda/core/common/advcp/advcp_config.yaml \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --attack-target all_non_attackers \
  --apply-cad-defense \
  --defense-threshold 0.75 \
  --advcp-vis \
  --advcp-vis-mode matplotlib \
  --advcp-vis-save
```

## Different Scenarios

### 28. Single Vehicle Scenario
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --attack-type lidar_remove_early \
  --attackers-ratio 0.5
```

### 29. Two Cars Scenario
```bash
python opencda.py \
  -t 2cars_no_rsu_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.5
```

### 30. Three Cars Scenario
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_remove_intermediate \
  --attackers-ratio 0.33
```

### 31. Four Cars Scenario
```bash
python opencda.py \
  -t 4cars_coperception \
  --with-coperception \
  --fusion-method late \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-late-opv2v-30 \
  --attack-type lidar_spoof_late \
  --attackers-ratio 0.25
```

### 32. Cars with RSU Scenario
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method late \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-late-opv2v-30 \
  --attack-type lidar_remove_late \
  --attackers-ratio 0.25 \
  --apply-cad-defense \
  --defense-threshold 0.7
```

### 33. Platooning Scenario
```bash
python opencda.py \
  -t platoon_joining_2lanefree_carla \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.2 \
  --apply-cad-defense
```

### 34. Cooperative Intersection Scenario
```bash
python opencda.py \
  -t cooperative_intersection \
  --with-coperception \
  --fusion-method late \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-late-opv2v-30 \
  --attack-type lidar_remove_late \
  --attackers-ratio 0.25
```

### 35. Single Intersection Town06
```bash
python opencda.py \
  -t single_intersection_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --attack-type lidar_spoof_early \
  --attackers-ratio 0.3
```

## Co-Simulation with SUMO

### 36. Co-Simulation with Early Attack
```bash
python opencda.py \
  -t single_town06_cosim \
  --cosim \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-early-opv2v-15 \
  --attack-type lidar_remove_early \
  --attackers-ratio 0.3
```

### 37. Co-Simulation with Intermediate Attack and Defense
```bash
python opencda.py \
  -t single_2lanefree_cosim \
  --cosim \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --apply-cad-defense \
  --defense-threshold 0.8
```

### 38. Realistic Co-Simulation with Late Attack
```bash
python opencda.py \
  -t realistic_town06_cosim \
  --cosim \
  --with-coperception \
  --fusion-method late \
  --with-advcp \
  --apply-ml \
  --model-dir opencda/coperception_models/pointpillar-late-opv2v-30 \
  --attack-type lidar_remove_late \
  --attackers-ratio 0.2
```

## Comprehensive Test Matrix

### 39. Full Matrix Test - All Attack Types with Early Fusion
```bash
# Early fusion - all attack types
python opencda.py -t single_town06_carla --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3
python opencda.py -t single_town06_carla --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3
```

### 40. Full Matrix Test - All Attack Types with Intermediate Fusion
```bash
# Intermediate fusion - all attack types
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_remove_intermediate --attackers-ratio 0.25
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.25
```

### 41. Full Matrix Test - All Attack Types with Late Fusion
```bash
# Late fusion - all attack types
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.2
```

## Stress Testing

### 42. High Attackers Ratio Stress Test
```bash
python opencda.py \
  -t 4cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.8 \
  --apply-cad-defense \
  --defense-threshold 0.5
```

### 43. Long Duration Attack Test
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --attack-type lidar_remove_early \
  --attackers-ratio 0.3 \
  --apply-cad-defense \
  --defense-threshold 0.7 \
  --verbose FULL
```

## Edge Cases

### 44. Single Attacker (Minimal)
```bash
python opencda.py \
  -t 4cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --attack-target specific_vehicle
```

### 45. All Vehicles Attackers (Maximum)
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method late \
  --with-advcp \
  --attack-type lidar_spoof_late \
  --attackers-ratio 1.0
```

### 46. No Attackers (Defense Only)
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --apply-cad-defense \
  --defense-threshold 0.7
```

## Verbosity and Logging

### 47. Silent Mode
```bash
python opencda.py \
  -t single_town06_carla \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --attack-type lidar_remove_early \
  --attackers-ratio 0.3 \
  --verbose 1
```

### 48. Full Verbose Mode with Debug
```bash
python opencda.py \
  -t 3cars_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --apply-cad-defense \
  --verbose 3 \
  --advcp-vis \
  --advcp-vis-save
```

## Combined Features

### 49. Attack + Defense + Visualization + Co-Simulation
```bash
python opencda.py \
  -t realistic_town06_cosim \
  --cosim \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --attack-target random \
  --apply-cad-defense \
  --defense-threshold 0.75 \
  --advcp-vis \
  --advcp-vis-mode both \
  --advcp-vis-save \
  --advcp-vis-types attack defense evaluation
```

### 50. All Attack Types Rotation Test Script
```bash
#!/bin/bash
# Test all attack types with different scenarios

SCENARIOS=("single_town06_carla" "3cars_coperception" "2cars_2rsu_coperception")
ATTACK_TYPES=("lidar_remove_early" "lidar_spoof_early" "lidar_remove_intermediate" "lidar_spoof_intermediate" "lidar_remove_late" "lidar_spoof_late" "adv_shape")
ATTACKERS_RATIOS=(0.1 0.2 0.3 0.5)
DEFENSE_THRESHOLDS=(0.5 0.7 0.9)

for scenario in "${SCENARIOS[@]}"; do
  for attack_type in "${ATTACK_TYPES[@]}"; do
    for ratio in "${ATTACKERS_RATIOS[@]}"; do
      echo "Testing: scenario=$scenario, attack=$attack_type, ratio=$ratio"
      python opencda.py \
        -t "$scenario" \
        --with-coperception \
        --fusion-method intermediate \
        --with-advcp \
        --attack-type "$attack_type" \
        --attackers-ratio "$ratio" \
        --verbose 1
    done
  done
done
```

## Notes

1. **Fusion Method Compatibility**:
   - `lidar_remove_early` and `lidar_spoof_early` require `--fusion-method early`
   - `lidar_remove_intermediate`, `lidar_spoof_intermediate`, and `adv_shape` require `--fusion-method intermediate`
   - `lidar_remove_late` and `lidar_spoof_late` require `--fusion-method late`

2. **Scenario Compatibility**:
   - Use scenarios with `coperception` in the name for cooperative perception tests
   - Single vehicle scenarios work best with early fusion
   - Multi-vehicle scenarios work with all fusion methods

3. **Performance Considerations**:
   - Visualization (`--advcp-vis`) significantly impacts performance
   - High attackers ratios (>0.5) may cause performance degradation
   - Co-simulation (`--cosim`) adds overhead

4. **Output**:
   - Simulation logs are saved to `simulation_logs/` by default
   - Visualizations are saved to `simulation_output/advcp_vis/` when `--advcp-vis-save` is used
   - Use `--record` flag to save detailed execution logs

5. **Dependencies**:
   - Ensure `--with-coperception` is set for AdvCP to work
   - OpenCOOD models must be available in `opencda/coperception_models/`
   - CUDA extension is built automatically on first run with `--with-advcp`

## Quick Reference Table

| Attack Type | Fusion Method | Description |
|-------------|---------------|-------------|
| `lidar_remove_early` | early | Remove LiDAR points of target object before fusion |
| `lidar_spoof_early` | early | Inject fake LiDAR points before fusion |
| `lidar_remove_intermediate` | intermediate | Remove features in intermediate representation |
| `lidar_spoof_intermediate` | intermediate | Inject fake features in intermediate representation |
| `lidar_remove_late` | late | Remove detected bounding boxes after fusion |
| `lidar_spoof_late` | late | Inject fake bounding boxes after fusion |
| `adv_shape` | intermediate | Adversarial shape perturbation attack |

| Target Strategy | Description |
|-----------------|-------------|
| `random` | Randomly select targets among non-attackers |
| `specific_vehicle` | Attack a specific target vehicle |
| `all_non_attackers` | Attack all non-attacker vehicles |

| Visualization Mode | Description |
|--------------------|-------------|
| `matplotlib` | 2D plots (top-down view) |
| `open3d` | 3D interactive visualization |
| `both` | Generate both 2D and 3D visualizations |
