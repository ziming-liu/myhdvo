# VKITTI2 Experiments Setup

This directory contains annotation files for two different experiments on the VKITTI2 dataset.

## Experiment 1: Weather Robustness Testing

**Objective**: Compare Visual Odometry (VO) robustness under different weather conditions.

### Directory Structure

Annotations are organized by weather condition, with each weather folder containing scene-specific files:

```
annotations/vkitti2/
├── clone/
│   ├── scene01.json (446 samples)
│   ├── scene02.json (232 samples)
│   ├── scene06.json (269 samples)
│   ├── scene18.json (338 samples)
│   └── scene20.json (836 samples)
├── fog/
│   └── [same 5 scene files]
├── morning/
│   └── [same 5 scene files]
├── overcast/
│   └── [same 5 scene files]
├── rain/
│   └── [same 5 scene files]
└── sunset/
    └── [same 5 scene files]
```

### Weather Condition Summary

| Weather Condition | Directory | Total Samples | Scenes |
|------------------|-----------|---------------|--------|
| Clone (Clear) | `clone/` | 2,121 | 5 |
| Fog | `fog/` | 2,121 | 5 |
| Morning | `morning/` | 2,121 | 5 |
| Overcast | `overcast/` | 2,121 | 5 |
| Rain | `rain/` | 2,121 | 5 |
| Sunset | `sunset/` | 2,121 | 5 |

### Experiment Setup Options

**Option 1: Test all weather conditions on same scene**
```bash
# Compare weather impact on Scene01 (low dynamic activity)
annotations/vkitti2/clone/scene01.json
annotations/vkitti2/fog/scene01.json
annotations/vkitti2/rain/scene01.json
...
```

**Option 2: Test all scenes for one weather condition**
```bash
# Test all scenes under fog conditions
annotations/vkitti2/fog/scene01.json
annotations/vkitti2/fog/scene02.json
annotations/vkitti2/fog/scene06.json
...
```

**Option 3: Use combined files** (also available)
- `vkitti2_train_clone.json` - All 5 scenes in clone weather
- `vkitti2_train_fog.json` - All 5 scenes in fog weather
- etc.

### Recommended Analysis

1. Test each weather condition separately across all scenes
2. Compare performance metrics (trajectory error, pose accuracy, etc.)
3. Identify which weather conditions cause the most degradation
4. Analyze per-scene performance differences under different weather

---

## Experiment 2: Dynamic Objects Robustness Testing

**Objective**: Verify Visual Odometry robustness in scenarios with varying numbers of dynamic objects.

### Dynamic Objects Statistics (Clone Weather)

Based on analysis of `bbox.txt` files with unique `trackID` counts:

| Scene | Total Objects | Moving Objects | Static Objects | Recommended Use |
|-------|--------------|----------------|----------------|-----------------|
| **Scene20** | 127 | **77** | 50 | High dynamic activity |
| **Scene18** | 21 | **21** | 0 | All objects moving |
| **Scene06** | 15 | **14** | 1 | High motion ratio |
| **Scene01** | 93 | **9** | 84 | Low dynamic activity |
| **Scene02** | 17 | **9** | 8 | Low dynamic activity |

### Recommended Experiment Groups

For testing dynamic object robustness, use the **clone weather** annotation file and compare:

#### High Dynamic Activity Group
- **Scene20**: 77 moving objects (most dynamic)
- **Scene18**: 21 moving objects (100% moving ratio)
- **Scene06**: 14 moving objects (93% moving ratio)

#### Low Dynamic Activity Group
- **Scene01**: 9 moving objects (10% moving ratio)
- **Scene02**: 9 moving objects (53% moving ratio)

### Scene-Specific Annotation Files

For easier per-scene testing, individual annotation files are provided:

| Scene | Annotation File | Samples | Moving Objects | Motion Ratio |
|-------|----------------|---------|----------------|--------------|
| **Scene20** | `vkitti2_train_scene20_clone.json` | 836 | 77 | 60.6% |
| **Scene18** | `vkitti2_train_scene18_clone.json` | 338 | 21 | 100.0% |
| **Scene06** | `vkitti2_train_scene06_clone.json` | 269 | 14 | 93.3% |
| **Scene01** | `vkitti2_train_scene01_clone.json` | 446 | 9 | 9.7% |
| **Scene02** | `vkitti2_train_scene02_clone.json` | 232 | 9 | 52.9% |

See `scene_index.txt` for quick reference.

### Experiment Setup

**Option 1: Test all scenes together**
1. Use `vkitti2_train_clone.json` (contains all 2,121 samples)
2. Split evaluation by scene field to analyze performance vs. number of dynamic objects

**Option 2: Test each scene independently** (Recommended)
1. Use individual scene annotation files (e.g., `vkitti2_train_scene20_clone.json`)
2. Run VO algorithm separately on each scene
3. Compare VO accuracy metrics between high and low dynamic activity groups
4. Analyze correlation between moving object count and VO performance degradation

### Analysis Recommendations

- **Scene20** is ideal for testing extreme dynamic scenarios (77 moving objects)
- **Scene18** tests pure dynamic environments (all objects are moving)
- **Scene01** and **Scene02** serve as baseline (minimal dynamic activity)
- Compare trajectory drift, pose estimation errors, and computational robustness

---

## File Descriptions

- `vkitti2_train_*.json`: Weather-specific annotation files for Experiment 1
- `dynamic_objects_statistics.txt`: Detailed statistics for Experiment 2
- `vkitti2_train_scene01_clone.json`: Original single-scene annotation (legacy)

## Dataset Structure

Each annotation entry contains:
- `image_0_paths`: Left camera RGB images (2 consecutive frames)
- `image_1_paths`: Right camera RGB images (2 consecutive frames)
- `depth_0_paths`: Left camera depth maps
- `depth_1_paths`: Right camera depth maps
- `gt_poses`: Ground truth camera poses (rotation + translation)
- `K_0`, `K_1`: Camera intrinsic matrices
- `focal_0`, `focal_1`: Focal lengths
- `baseline_01`: Stereo baseline distance
- `scene`: Scene identifier (Scene01-Scene20)
- `variation`: Weather condition

## Usage Example

```python
import json

# Load weather-specific annotation
with open('annotations/vkitti2/vkitti2_train_fog.json', 'r') as f:
    fog_data = json.load(f)

# Filter by scene for dynamic object experiment
scene20_samples = [s for s in fog_data if s['scene'] == 'Scene20']
```

---

**Generated**: January 28, 2026  
**Dataset**: VKITTI2 (Virtual KITTI 2)  
**Scenes**: Scene01, Scene02, Scene06, Scene18, Scene20
