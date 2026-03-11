# Comprehensive Test Commands for AdvCP Module

## Basic Test Structure
```bash
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate [ADVCP_OPTIONS]
```

---

## 1. Attack Type Tests (No Defense, No Visualization)

### 1.1 LIDAR Remove Attacks
```bash
# Early frame removal
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.5

# Intermediate frame removal
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_remove_intermediate --attackers-ratio 0.5

# Late frame removal
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.5
```

### 1.2 LIDAR Spoof Attacks
```bash
# Early frame spoofing
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5

# Intermediate frame spoofing
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.5

# Late frame spoofing
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.5
```

### 1.3 Adversarial Shape Attack
```bash
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.5
```

---

## 2. Attack Target Strategy Tests
```bash
# Random target selection
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --attack-target random

# Specific vehicle target
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --attack-target specific_vehicle

# All non-attackers target
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --attack-target all_non_attackers
```

---

## 3. Attackers Ratio Variation Tests
```bash
# Low ratio (20%)
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.2

# Medium ratio (50%)
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5

# High ratio (80%)
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.8
```

---

## 4. Defense Mechanism Tests

### 4.1 Without Defense
```bash
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5
```

### 4.2 With CAD Defense (Default Threshold 0.7)
```bash
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --apply-cad-defense
```

### 4.3 With Different Defense Thresholds
```bash
# Low threshold (0.3) - more permissive
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --apply-cad-defense --defense-threshold 0.3

# Medium threshold (0.5)
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --apply-cad-defense --defense-threshold 0.5

# High threshold (0.9) - more strict
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --apply-cad-defense --defense-threshold 0.9
```

---

## 5. Visualization Tests

### 5.1 Basic Visualization (Matplotlib 2D)
```bash
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-show
```

### 5.2 3D Visualization (Open3D)
```bash
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --advcp-vis --advcp-vis-mode open3d --advcp-vis-show
```

### 5.3 Combined 2D+3D Visualization
```bash
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --advcp-vis --advcp-vis-mode both --advcp-vis-show
```

### 5.4 Save Visualization to Disk (No Display)
```bash
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --advcp-vis --advcp-vis-save --advcp-vis-output-dir simulation_output/advcp_vis_test1
```

### 5.5 Specific Visualization Types
```bash
# Only attack visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --advcp-vis --advcp-vis-types attack

# Only defense visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --apply-cad-defense --advcp-vis --advcp-vis-types defense

# Only evaluation visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --advcp-vis --advcp-vis-types evaluation

# Multiple visualization types
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --advcp-vis --advcp-vis-types attack defense evaluation
```

---

## 6. Combined Attack + Defense + Visualization Tests
```bash
# Full featured test with all options
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --attack-target random --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-show --advcp-vis-types attack defense evaluation
```

---

## 7. Different Fusion Methods with AdvCP
```bash
# Late fusion with attack
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-late-opv2v-20 --fusion-method late --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5

# Early fusion with attack
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-early-opv2v-20 --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5

# Intermediate fusion with attack (default)
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5
```

---

## 8. Attack Parameter Variations (Advanced)
```bash
# Dense parameter variations (1, 2, 3)
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --attack-parameters dense=1

# Sync parameter (0=Async, 1=Sync)
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --attack-parameters sync=0

# Adversarial shape parameter
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.5 --attack-parameters advshape=1
```

---

## 9. Complete Test Matrix (All Attack Types × With/Without Defense × Visualization Modes)

### 9.1 Without Defense
```bash
# lidar_remove_early
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.5

# lidar_remove_intermediate
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_remove_intermediate --attackers-ratio 0.5

# lidar_remove_late
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.5

# lidar_spoof_early
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5

# lidar_spoof_intermediate
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.5

# lidar_spoof_late
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.5

# adv_shape
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.5
```

### 9.2 With Defense (CAD)
```bash
# lidar_remove_early with defense
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.5 --apply-cad-defense

# lidar_remove_intermediate with defense
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_remove_intermediate --attackers-ratio 0.5 --apply-cad-defense

# lidar_remove_late with defense
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.5 --apply-cad-defense

# lidar_spoof_early with defense
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --apply-cad-defense

# lidar_spoof_intermediate with defense
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.5 --apply-cad-defense

# lidar_spoof_late with defense
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.5 --apply-cad-defense

# adv_shape with defense
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.5 --apply-cad-defense
```

---

## 10. Edge Cases and Special Tests

### 10.1 Edge Attackers Ratios
```bash
# No attackers (only defense mechanism test)
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attackers-ratio 0.0 --apply-cad-defense

# All vehicles are attackers
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 1.0
```

### 10.2 Different Scenarios
```bash
# Using 3cars_coperception scenario
python opencda.py -t 3cars_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5

# Using 2cars_no_rsu_coperception scenario
python opencda.py -t 2cars_no_rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5
```

### 10.3 Recording and Logging
```bash
# With recording
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --record

# With verbose output
python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.5 --verbose 3
```

---

## 11. Quick Test Scripts

### 11.1 Quick All-Attack-Types Test (No Defense, No Vis)
```bash
#!/bin/bash
ATTACK_TYPES=("lidar_remove_early" "lidar_remove_intermediate" "lidar_remove_late" "lidar_spoof_early" "lidar_spoof_intermediate" "lidar_spoof_late" "adv_shape")
for attack in "${ATTACK_TYPES[@]}"; do
    echo "Testing attack: $attack"
    python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type $attack --attackers-ratio 0.5
done
```

### 11.2 Defense Effectiveness Test
```bash
#!/bin/bash
ATTACK_TYPES=("lidar_spoof_early" "lidar_remove_early" "adv_shape")
RATIOS=(0.2 0.5 0.8)
THRESHOLDS=(0.3 0.5 0.7 0.9)
for attack in "${ATTACK_TYPES[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        for thresh in "${THRESHOLDS[@]}"; do
            echo "Testing $attack with ratio=$ratio, threshold=$thresh"
            python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type $attack --attackers-ratio $ratio --apply-cad-defense --defense-threshold $thresh
        done
    done
done
```

### 11.3 Visualization Test Suite
```bash
#!/bin/bash
ATTACK_TYPES=("lidar_spoof_early" "adv_shape")
VIS_MODES=("matplotlib" "open3d" "both")
VIS_TYPES=("attack" "defense" "evaluation" "attack defense evaluation")
for attack in "${ATTACK_TYPES[@]}"; do
    for mode in "${VIS_MODES[@]}"; do
        for types in "${VIS_TYPES[@]}"; do
            echo "Testing $attack with vis_mode=$mode, vis_types=$types"
            python opencda.py -t 2cars_2rsu_coperception --with-coperception --model-dir opencda/coperception_models/pointpillar-intermediate-opv2v-20 --fusion-method intermediate --with-advcp --attack-type $attack --attackers-ratio 0.5 --advcp-vis --advcp-vis-mode $mode --advcp-vis-save --advcp-vis-types $types
        done
    done
done
```

---

## Summary

These commands cover:
- ✅ All 7 attack types (3 remove variants × 3 spoof variants + adv_shape)
- ✅ All attack target strategies (random, specific_vehicle, all_non_attackers)
- ✅ Various attacker ratios (0.0, 0.2, 0.5, 0.8, 1.0)
- ✅ Defense mechanism (with/without CAD, different thresholds)
- ✅ All visualization modes (matplotlib, open3d, both)
- ✅ All visualization types (attack, defense, evaluation, ground_seg, tracking, roc)
- ✅ Different fusion methods (late, early, intermediate)
- ✅ Different scenarios (2cars_2rsu, 3cars, 2cars_no_rsu)
- ✅ Recording and logging options

The test matrix allows comprehensive validation of the AdvCP module's capabilities including attack generation, defense mechanisms, and visualization outputs.