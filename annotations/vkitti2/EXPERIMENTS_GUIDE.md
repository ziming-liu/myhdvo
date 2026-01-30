# VKitti2 Experiments Guide

This directory contains organized annotation files for two key Visual Odometry experiments on the VKitti2 dataset.

## Directory Structure

```
annotations/vkitti2/
├── weather/              # Experiment 1: Weather Robustness
│   ├── clone/           # Clear weather (baseline)
│   ├── fog/             # Foggy conditions
│   ├── morning/         # Morning lighting
│   ├── overcast/        # Overcast conditions
│   ├── rain/            # Rainy weather
│   └── sunset/          # Sunset lighting
│
├── dynamic/             # Experiment 2: Dynamic Objects Robustness
│   ├── scene01.json     # Low dynamic activity (9 moving objects)
│   ├── scene02.json     # Low dynamic activity (9 moving objects)
│   ├── scene06.json     # High dynamic activity (14 moving objects)
│   ├── scene18.json     # Very high activity (21 moving objects)
│   ├── scene20.json     # Highest activity (77 moving objects)
│   └── all_scenes.json  # Combined dataset
│
└── [legacy files...]    # Old organization (can be ignored)
```

---

## Experiment 1: Weather Robustness Testing

**Location**: `annotations/vkitti2/weather/`

### Objective
Compare Visual Odometry performance under different weather and lighting conditions.

### Weather Conditions
- **clone**: Clear weather (baseline)
- **fog**: Heavy fog
- **morning**: Morning lighting
- **overcast**: Cloudy/overcast sky
- **rain**: Rainy conditions
- **sunset**: Sunset lighting

### Files per Weather Condition
Each weather directory contains:
- `scene01.json` - 446 samples
- `scene02.json` - 232 samples
- `scene06.json` - 269 samples
- `scene18.json` - 338 samples
- `scene20.json` - 836 samples
- `all_scenes.json` - 2,121 samples (combined)

### Usage Example
```python
import json

# Test on rainy conditions, Scene01
with open('annotations/vkitti2/weather/rain/scene01.json') as f:
    rain_data = json.load(f)

# Compare with clear weather
with open('annotations/vkitti2/weather/clone/scene01.json') as f:
    clear_data = json.load(f)

# Run your VO algorithm on both and compare results
```

### Recommended Analysis
1. Use **clone** weather as baseline
2. Test each weather condition separately
3. Compare trajectory errors across conditions
4. Identify most challenging weather for your VO method

---

## Experiment 2: Dynamic Objects Robustness Testing

**Location**: `annotations/vkitti2/dynamic/`

### Objective
Evaluate Visual Odometry robustness in scenes with varying numbers of dynamic (moving) objects.

### Scenes by Dynamic Activity

| Scene | Samples | Moving Objects | Total Objects | Motion Ratio | Activity Level |
|-------|---------|----------------|---------------|--------------|----------------|
| Scene20 | 836 | 77 | 127 | 60.6% | ⚠️ Very High |
| Scene18 | 338 | 21 | 21 | 100.0% | ⚠️ Extreme |
| Scene06 | 269 | 14 | 15 | 93.3% | ⚠️ High |
| Scene02 | 232 | 9 | 17 | 52.9% | ✓ Moderate |
| Scene01 | 446 | 9 | 93 | 9.7% | ✓ Low |

### Usage Example
```python
import json

# Test on high dynamic activity scene
with open('annotations/vkitti2/dynamic/scene20.json') as f:
    high_dynamic = json.load(f)

# Test on low dynamic activity scene
with open('annotations/vkitti2/dynamic/scene01.json') as f:
    low_dynamic = json.load(f)

# Compare VO performance
```

### Recommended Analysis
1. Test on low dynamic scenes (Scene01, Scene02) as baseline
2. Test on high dynamic scenes (Scene06, Scene18, Scene20)
3. Analyze correlation between moving object count and VO accuracy
4. Identify failure modes in highly dynamic scenarios

---

## Annotation Format

All annotation files contain the following fields:

```json
{
  "image_0_paths": ["path/to/left_cam/frame_t0.jpg", "path/to/left_cam/frame_t1.jpg"],
  "image_1_paths": ["path/to/right_cam/frame_t0.jpg", "path/to/right_cam/frame_t1.jpg"],
  "depth_0_paths": ["path/to/left_depth/frame_t0.png", "path/to/left_depth/frame_t1.png"],
  "depth_1_paths": ["path/to/right_depth/frame_t0.png", "path/to/right_depth/frame_t1.png"],
  "gt_poses": [
    "r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz",  // Absolute pose at t0
    "r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz"   // Absolute pose at t1
  ],
  "K_0": "fx 0 cx 0  0 fy cy 0  0 0 1 0",  // Left camera intrinsics
  "K_1": "fx 0 cx 0  0 fy cy 0  0 0 1 0",  // Right camera intrinsics
  "focal_0": "725.0087",
  "focal_1": "725.0087",
  "baseline_01": "0.532725",  // Stereo baseline in meters
  "seq_len": 2,
  "scene": "Scene01",
  "variation": "clone"
}
```

### Ground Truth Poses (gt_poses)

- **Format**: 3x4 transformation matrix (Camera-to-World) flattened as 12 values
- **Type**: Absolute poses (not relative)
- **Usage**: Compute relative transformation between frames:
  ```python
  import torch
  
  # Load poses
  abs_pose0 = parse_pose(gt_poses[0])  # 4x4 matrix
  abs_pose1 = parse_pose(gt_poses[1])  # 4x4 matrix
  
  # Compute relative transformation (frame 0 -> frame 1)
  rel_pose = torch.linalg.inv(abs_pose0) @ abs_pose1
  ```

---

## Dataset Statistics

### Total Annotations
- **Weather Experiment**: 6 conditions × 2,121 samples = 12,726 samples
- **Dynamic Experiment**: 5 scenes × avg 444 samples = 2,121 samples

### Scene Coverage
- All 5 available VKitti2 scenes: Scene01, Scene02, Scene06, Scene18, Scene20
- All 6 weather/lighting conditions covered
- Consistent camera parameters across all conditions

### Data Completeness
✅ All files contain:
- RGB stereo image pairs
- Ground truth depth maps
- Camera intrinsics (K matrices)
- Stereo baseline
- **Ground truth absolute poses** (gt_poses)
- Scene and variation metadata

---

## Quick Start

### 1. Weather Robustness Test
```bash
# Test your VO on rainy conditions
python tools/test.py configs/your_vo_config.py \
    --ann_file annotations/vkitti2/weather/rain/all_scenes.json
```

### 2. Dynamic Objects Test
```bash
# Test on high dynamic activity scene
python tools/test.py configs/your_vo_config.py \
    --ann_file annotations/vkitti2/dynamic/scene20.json
```

### 3. Comparative Analysis
```python
import json
import numpy as np

# Load results from different conditions
results = {}
for weather in ['clone', 'fog', 'rain', 'overcast']:
    with open(f'results/{weather}_trajectory.txt') as f:
        results[weather] = load_trajectory(f)

# Compare trajectory errors
for weather, traj in results.items():
    error = compute_ate(traj, ground_truth)
    print(f"{weather}: ATE = {error:.4f}m")
```

---

## Notes

- All scenes use **Camera_0** (left camera) poses as ground truth
- Stereo baseline is consistent: **0.532725 meters**
- Focal length: **725.0087 pixels**
- Principal point: **(620.5, 187.0)**
- Image resolution: **1242 × 375** pixels

---

## References

- **VKitti2 Dataset**: [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
- **Dynamic Object Statistics**: See `annotations/vkitti2/dynamic_objects_statistics.txt`
- **Weather Conditions Guide**: See `annotations/vkitti2/weather/README.txt`

---

**Last Updated**: January 29, 2026  
**Dataset Version**: VKitti2  
**Annotation Format**: HDVO-compatible JSON
