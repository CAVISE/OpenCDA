# AdvCP Module - Test Command Examples

This document provides various command examples to test the AdvCP (Advanced Collaborative Perception) module with different configurations, attack types, and defense mechanisms.

## Basic Test Commands

### 1. Simple Early Attack (No Defense)
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
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
  --attack-type lidar_remove_late \
  --attackers-ratio 0.2
```

## Attack Types Testing

### 4. Early Remove Attack
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --attack-type lidar_remove_early \
  --attackers-ratio 0.3 \
  --attack-target random
```

### 5. Early Spoof Attack
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
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
  --attack-type adv_shape \
  --attackers-ratio 0.25
```

## Defense Testing

### 11. Attack with CAD Defense (Early)
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
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
  --attack-type lidar_remove_late \
  --attackers-ratio 0.2 \
  --apply-cad-defense \
  --defense-threshold 0.6
```

### 14. Defense Only (No Attack)
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
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
  --attack-type lidar_spoof_intermediate \
  --attackers-ratio 0.25 \
  --attack-target all_non_attackers
```

## Visualization Tests

### 22. Basic Visualization (Matplotlib 2D)
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
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
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --attack-type lidar_spoof_early \
  --attackers-ratio 0.3 \
  --advcp-vis \
  --advcp-vis-mode matplotlib \
  --advcp-vis-save \
  --advcp-vis-output-dir /tmp/advcp_test_output \
  --advcp-vis-types attack defense tracking roc
```

## Comprehensive Visualization Testing

### 51. All Attack Types - Matplotlib 2D Visualization
```bash
# Early fusion attacks with 2D visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save

# Intermediate fusion attacks with 2D visualization
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_remove_intermediate --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save

# Late fusion attacks with 2D visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save
```

### 52. All Attack Types - Open3D 3D Visualization
```bash
# Early fusion attacks with 3D visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save

# Intermediate fusion attacks with 3D visualization
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_remove_intermediate --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save

# Late fusion attacks with 3D visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save
```

### 53. Defense Mechanisms with Full Visualization
```bash
# CAD Defense with all visualization types
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation tracking roc

python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.8 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation

python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2 --apply-cad-defense --defense-threshold 0.6 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types defense ground_seg tracking
```

### 54. Defense Only with Visualization
```bash
# No attack, only defense monitoring
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types defense evaluation

python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --apply-cad-defense --defense-threshold 0.8 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-types defense tracking

python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --apply-cad-defense --defense-threshold 0.6 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types defense evaluation ground_seg
```

### 55. Visualization Type Variations
```bash
# Attack visualization only
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types attack

# Defense visualization only
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.8 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types defense

# Evaluation metrics visualization only
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types evaluation

# Ground segmentation visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-types ground_seg

# Tracking visualization
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types tracking

# ROC curve visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.2 --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types roc
```

### 56. Interactive Visualization (Show Window)
```bash
# Show matplotlib visualization interactively
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-show

# Show Open3D visualization interactively
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode open3d --advcp-vis-show

# Show both visualizations interactively
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode both --advcp-vis-show --advcp-vis-types attack defense
```

### 57. Co-Simulation with Visualization
```bash
# Early attack with visualization in co-simulation
python opencda.py -t single_town06_cosim --cosim --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save

# Intermediate attack with defense and visualization in co-simulation
python opencda.py -t single_2lanefree_cosim --cosim --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.8 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation

# Late attack with visualization in realistic co-simulation
python opencda.py -t realistic_town06_cosim --cosim --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save
```

### 58. Different Attackers Ratios with Visualization
```bash
# Test various attacker ratios with visualization
python opencda.py -t 4cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.1 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save
python opencda.py -t 4cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save
python opencda.py -t 4cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.5 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save
python opencda.py -t 4cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.8 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation
```

### 59. Target Selection Strategies with Visualization
```bash
# Random target selection with visualization
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --attack-target random --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types attack tracking

# Specific vehicle target with visualization
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --attack-target specific_vehicle --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-types attack tracking

# All non-attackers target with visualization
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --attack-target all_non_attackers --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation
```

### 60. Different Scenarios with Full Visualization
```bash
# Single vehicle scenario
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation

# Two cars scenario
python opencda.py -t 2cars_no_rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_remove_intermediate --attackers-ratio 0.5 --advcp-vis --advcp-vis-mode both --advcp-vis-save

# Three cars scenario
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.33 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense tracking

# Four cars scenario
python opencda.py -t 4cars_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode both --advcp-vis-save

# Cars with RSU scenario
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation tracking roc
```

### 61. Platooning and Intersection Scenarios with Visualization
```bash
# Platooning scenario with visualization
python opencda.py -t platoon_joining_2lanefree_carla --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.2 --apply-cad-defense --defense-threshold 0.75 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense tracking

# Cooperative intersection scenario
python opencda.py -t cooperative_intersection --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-types attack evaluation

# Single intersection scenario
python opencda.py -t single_intersection_town06_carla --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types attack defense
```

### 62. Stress Testing with Visualization
```bash
# High attackers ratio stress test with visualization
python opencda.py -t 4cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.8 --apply-cad-defense --defense-threshold 0.5 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types attack defense evaluation

# Long duration attack with visualization and full logging
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --apply-cad-defense --defense-threshold 0.7 --verbose FULL --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation tracking
```

### 63. Edge Cases with Visualization
```bash
# Single attacker (minimal) with visualization
python opencda.py -t 4cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --attack-target specific_vehicle --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types attack tracking

# All vehicles as attackers with visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 1.0 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-types attack defense

# No attackers (defense only) with full visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types defense evaluation ground_seg tracking roc
```

### 64. Custom Output Directories for Different Test Types
```bash
# Organize outputs by test category
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-output-dir simulation_output/advcp_vis/early_attack_2d

python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.8 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-output-dir simulation_output/advcp_vis/intermediate_defense_3d

python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-output-dir simulation_output/advcp_vis/late_attack_both
```

### 65. Verbosity Levels with Visualization
```bash
# Silent mode with visualization (minimal console output)
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types attack --verbose 1

# Info mode with visualization
python opencda.py -t 3cars_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.8 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --verbose 2

# Full verbose mode with visualization and all types
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.2 --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation tracking roc --verbose 3
```

### 66. Combined Co-Simulation + Defense + Visualization Matrix
```bash
# Comprehensive co-simulation tests with visualization
python opencda.py -t single_town06_cosim --cosim --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation

python opencda.py -t single_2lanefree_cosim --cosim --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.8 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-types attack tracking

python opencda.py -t realistic_town06_cosim --cosim --with-coperception --fusion-method late --with-advcp --attack-type adv_shape --attackers-ratio 0.2 --apply-cad-defense --defense-threshold 0.75 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types attack defense evaluation ground_seg
```

### 67. Complete Visualization Test Suite Script
```bash
#!/bin/bash
# Comprehensive AdvCP Visualization Test Suite
# This script runs a complete set of visualization tests

OUTPUT_DIR="simulation_output/advcp_vis_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "AdvCP Visualization Test Suite"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Test 1: Early fusion attacks with 2D viz
echo "[1/6] Testing early fusion attacks with matplotlib..."
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/early_remove" --advcp-vis-types attack defense
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/early_spoof" --advcp-vis-types attack defense

# Test 2: Intermediate fusion attacks with 3D viz
echo "[2/6] Testing intermediate fusion attacks with open3d..."
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_remove_intermediate --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/intermediate_remove"
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/intermediate_spoof"
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/adv_shape"

# Test 3: Late fusion attacks with both viz modes
echo "[3/6] Testing late fusion attacks with both visualization modes..."
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/late_remove" --advcp-vis-types attack defense evaluation
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/late_spoof" --advcp-vis-types attack defense evaluation

# Test 4: Defense mechanisms with full visualization
echo "[4/6] Testing defense mechanisms with full visualization..."
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/defense_early" --advcp-vis-types attack defense evaluation tracking roc
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.8 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/defense_intermediate" --advcp-vis-types attack defense evaluation tracking roc
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2 --apply-cad-defense --defense-threshold 0.6 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/defense_late" --advcp-vis-types attack defense evaluation tracking roc

# Test 5: Co-simulation with visualization
echo "[5/6] Testing co-simulation scenarios with visualization..."
python opencda.py -t 2cars_2rsu_coperception --cosim --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/cosim_early"
python opencda.py -t 2cars_2rsu_coperception --cosim --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.8 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/cosim_intermediate_defense"
python opencda.py -t 2cars_2rsu_coperception --cosim --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/cosim_late"

# Test 6: Various attacker ratios and targets with visualization
echo "[6/6] Testing various attacker ratios and target strategies..."
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.1 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/ratio_10"
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.5 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/ratio_50"
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --attack-target specific_vehicle --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/target_specific"
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --attack-target all_non_attackers --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-output-dir "$OUTPUT_DIR/target_all"

echo "=========================================="
echo "Visualization Test Suite Complete!"
echo "All outputs saved to: $OUTPUT_DIR"
echo "=========================================="
```

### 68. Quick Visualization Comparison Tests
```bash
# Compare attack vs no-attack with visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types attack defense
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types defense

# Compare defense thresholds with visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.5 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-output-dir simulation_output/advcp_vis/defense_thresh_0.5 --advcp-vis-types defense evaluation
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-output-dir simulation_output/advcp_vis/defense_thresh_0.7 --advcp-vis-types defense evaluation
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.9 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-output-dir simulation_output/advcp_vis/defense_thresh_0.9 --advcp-vis-types defense evaluation
```

### 69. All Visualization Types Individual Testing
```bash
# Test each visualization type separately for comprehensive coverage

# Attack visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types attack

# Defense visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type lidar_spoof_intermediate --attackers-ratio 0.25 --apply-cad-defense --defense-threshold 0.8 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types defense

# Evaluation visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types evaluation

# Ground segmentation visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3 --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-types ground_seg

# Tracking visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.25 --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types tracking

# ROC visualization
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_spoof_late --attackers-ratio 0.2 --apply-cad-defense --defense-threshold 0.7 --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types roc
```

### 70. Performance Benchmarking with Visualization
```bash
# Benchmark visualization performance impact
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3 --verbose FULL --advcp-vis --advcp-vis-mode matplotlib --advcp-vis-save --advcp-vis-types attack

python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method intermediate --with-advcp --attack-type adv_shape --attackers-ratio 0.25 --verbose FULL --advcp-vis --advcp-vis-mode open3d --advcp-vis-save --advcp-vis-types attack defense

python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method late --with-advcp --attack-type lidar_remove_late --attackers-ratio 0.2 --verbose FULL --advcp-vis --advcp-vis-mode both --advcp-vis-save --advcp-vis-types attack defense evaluation tracking
```

## Custom Configuration File

### 26. Using Custom AdvCP Config
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
  --advcp-config /path/to/custom_advcp_config.yaml \
  --attackers-ratio 0.4 \
  --attack-target random
```

### 27. Custom Config with All Options
```bash
python opencda.py \
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method intermediate \
  --with-advcp \
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
  -t 2cars_2rsu_coperception \
  --with-coperception \
  --fusion-method early \
  --with-advcp \
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
  --attack-type lidar_remove_late \
  --attackers-ratio 0.2
```

## Comprehensive Test Matrix

### 39. Full Matrix Test - All Attack Types with Early Fusion
```bash
# Early fusion - all attack types
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_remove_early --attackers-ratio 0.3
python opencda.py -t 2cars_2rsu_coperception --with-coperception --fusion-method early --with-advcp --attack-type lidar_spoof_early --attackers-ratio 0.3
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
  -t 2cars_2rsu_coperception \
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
  -t 2cars_2rsu_coperception \
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
  -t 2cars_2rsu_coperception \
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

SCENARIOS=("2cars_2rsu_coperception" "3cars_coperception" "2cars_2rsu_coperception")
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
