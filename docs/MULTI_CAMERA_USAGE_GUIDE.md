# Multi-Camera Point Cloud Fusion - Usage Guide

## Overview

The Visual Feedback Module now supports multi-camera setups for improved point cloud coverage and occlusion handling. This guide explains how to use the multi-camera functionality.

## Features

- **Multiple Camera Support**: Capture from 2+ cameras simultaneously
- **Automatic Transformation**: Transform each camera's point cloud to common frame
- **Point Cloud Merging**: Combine all point clouds into unified representation
- **Backward Compatible**: Single-camera mode still works with existing code

## Basic Usage

### Single Camera (Original API)

```python
from acktr.visual_feedback import VisualFeedbackModule, CameraConfig
import numpy as np

# Create single camera configuration
intrinsics = np.eye(3, dtype=np.float32)
intrinsics[0, 0] = 500  # fx
intrinsics[1, 1] = 500  # fy

extrinsics = np.eye(4, dtype=np.float32)

camera_config = CameraConfig(
    intrinsics=intrinsics,
    extrinsics=extrinsics,
    resolution=(640, 480),
    depth_range=(0.1, 10.0)
)

# Create module with single camera
vfm = VisualFeedbackModule(
    camera_config=camera_config,
    container_size=(10, 10, 10),
    simulation_mode=True
)
```

### Multi-Camera Setup

```python
from acktr.visual_feedback import VisualFeedbackModule, CameraConfig
import numpy as np

# Create multiple camera configurations
camera_configs = []

# Camera 1: Front view
intrinsics_1 = np.eye(3, dtype=np.float32)
intrinsics_1[0, 0] = 500
intrinsics_1[1, 1] = 500

extrinsics_1 = np.eye(4, dtype=np.float32)
extrinsics_1[0, 3] = 0.0  # x position
extrinsics_1[1, 3] = -5.0  # y position (in front)
extrinsics_1[2, 3] = 3.0  # z position (above)

camera_configs.append(CameraConfig(
    intrinsics=intrinsics_1,
    extrinsics=extrinsics_1,
    resolution=(640, 480),
    depth_range=(0.1, 10.0)
))

# Camera 2: Side view
intrinsics_2 = np.eye(3, dtype=np.float32)
intrinsics_2[0, 0] = 500
intrinsics_2[1, 1] = 500

extrinsics_2 = np.eye(4, dtype=np.float32)
extrinsics_2[0, 3] = 5.0  # x position (to the side)
extrinsics_2[1, 3] = 0.0  # y position
extrinsics_2[2, 3] = 3.0  # z position (above)

camera_configs.append(CameraConfig(
    intrinsics=intrinsics_2,
    extrinsics=extrinsics_2,
    resolution=(640, 480),
    depth_range=(0.1, 10.0)
))

# Camera 3: Top view
intrinsics_3 = np.eye(3, dtype=np.float32)
intrinsics_3[0, 0] = 500
intrinsics_3[1, 1] = 500

extrinsics_3 = np.eye(4, dtype=np.float32)
extrinsics_3[0, 3] = 0.0  # x position
extrinsics_3[1, 3] = 0.0  # y position
extrinsics_3[2, 3] = 8.0  # z position (high above)

camera_configs.append(CameraConfig(
    intrinsics=intrinsics_3,
    extrinsics=extrinsics_3,
    resolution=(640, 480),
    depth_range=(0.1, 10.0)
))

# Create module with multiple cameras
vfm = VisualFeedbackModule(
    camera_configs=camera_configs,
    container_size=(10, 10, 10),
    simulation_mode=True
)
```

## Capturing and Processing Point Clouds

### Method 1: Automatic (Recommended)

```python
# Capture, transform, and merge in one call
merged_cloud = vfm.capture_and_merge_multi_camera()

# Process unified cloud for box detection
detected_boxes = vfm.process_point_cloud(merged_cloud)

# Update height map
height_map = np.zeros((10, 10), dtype=np.float32)
updated_map = vfm.update_height_map(height_map, detected_boxes)
```

### Method 2: Manual (Step-by-Step)

```python
# Step 1: Capture from all cameras
point_clouds = vfm.capture_point_clouds_multi_camera()

# Step 2: Transform each to common frame
transformed_clouds = []
for pc, camera_config in zip(point_clouds, vfm.camera_configs):
    transformed = vfm.transform_point_cloud_to_common_frame(pc, camera_config)
    transformed_clouds.append(transformed)

# Step 3: Merge into unified cloud
merged_cloud = vfm.merge_point_clouds(transformed_clouds)

# Step 4: Process unified cloud
detected_boxes = vfm.process_point_cloud(merged_cloud)

# Step 5: Update height map
height_map = np.zeros((10, 10), dtype=np.float32)
updated_map = vfm.update_height_map(height_map, detected_boxes)
```

## Camera Calibration

### Extrinsic Parameters

The extrinsic matrix transforms points from world coordinates to camera coordinates:

```
[R | t]   where R is 3x3 rotation, t is 3x1 translation
[0 | 1]
```

For the inverse transformation (camera to world), the module automatically computes:

```python
extrinsics_inv = np.linalg.inv(extrinsics)
```

### Example: Camera at Position (2, 3, 5) Looking at Origin

```python
import numpy as np

# Camera position in world frame
camera_pos = np.array([2.0, 3.0, 5.0])

# Simple extrinsics (identity rotation, translation to camera position)
extrinsics = np.eye(4, dtype=np.float32)
extrinsics[0, 3] = camera_pos[0]
extrinsics[1, 3] = camera_pos[1]
extrinsics[2, 3] = camera_pos[2]

# For more complex orientations, include rotation matrix
# Example: 30-degree rotation around z-axis
angle = np.radians(30)
extrinsics[0, 0] = np.cos(angle)
extrinsics[0, 1] = -np.sin(angle)
extrinsics[1, 0] = np.sin(angle)
extrinsics[1, 1] = np.cos(angle)
```

## Best Practices

### 1. Camera Placement

- **Coverage**: Position cameras to minimize occlusions
- **Overlap**: Ensure some overlap between camera views
- **Height**: Place cameras above the container for better view
- **Angles**: Use different viewing angles (front, side, top)

### 2. Number of Cameras

- **Minimum**: 2 cameras for basic stereo coverage
- **Recommended**: 3-4 cameras for good all-around coverage
- **Maximum**: No hard limit, but more cameras = more processing time

### 3. Calibration

- **Accuracy**: Ensure accurate camera calibration
- **Validation**: Test with known objects to verify transformations
- **Updates**: Re-calibrate periodically to account for drift

### 4. Performance

- **Downsampling**: Consider downsampling large point clouds
- **Filtering**: Remove outliers and noise before merging
- **Caching**: Cache camera configurations if they don't change

## Troubleshooting

### Problem: Merged cloud has wrong number of points

**Solution**: Check that all cameras are capturing successfully
```python
point_clouds = vfm.capture_point_clouds_multi_camera()
for i, pc in enumerate(point_clouds):
    print(f"Camera {i}: {len(pc)} points")
```

### Problem: Transformed points are in wrong location

**Solution**: Verify extrinsic calibration
```python
# Test with known point
test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
transformed = vfm.transform_point_cloud_to_common_frame(test_point, camera_config)
print(f"Origin in world frame: {transformed}")
```

### Problem: Box detection fails with merged cloud

**Solution**: Check point cloud quality and density
```python
merged = vfm.capture_and_merge_multi_camera()
print(f"Total points: {len(merged)}")
print(f"Point cloud bounds: min={merged.min(axis=0)}, max={merged.max(axis=0)}")
```

## Integration with Existing Code

The multi-camera functionality integrates seamlessly with the existing packing system:

```python
from envs.bpp0.bin3D_reliable import ReliablePackingGame

# Create environment with multi-camera visual feedback
env = ReliablePackingGame(
    visual_feedback_enabled=True,
    camera_configs=camera_configs,  # Pass multi-camera config
    container_size=(10, 10, 10)
)

# Use as normal - visual feedback will use merged point clouds
obs = env.reset()
action = agent.predict(obs)
obs, reward, done, info = env.step(action)
```

## API Reference

### VisualFeedbackModule

#### Constructor
```python
VisualFeedbackModule(
    camera_config=None,           # Single camera (legacy)
    camera_configs=None,          # Multiple cameras (new)
    container_size=(10, 10, 10),  # Container dimensions
    grid_size=(10, 10),           # Height map resolution
    simulation_mode=True          # Use mock capture
)
```

#### Methods

**capture_point_clouds_multi_camera()**
- Captures from all cameras simultaneously
- Returns: `List[np.ndarray]` - List of point clouds

**transform_point_cloud_to_common_frame(point_cloud, camera_config)**
- Transforms point cloud to world frame
- Args: `point_cloud` (Nx3), `camera_config` (CameraConfig)
- Returns: `np.ndarray` - Transformed points (Nx3)

**merge_point_clouds(point_clouds)**
- Merges multiple point clouds
- Args: `point_clouds` (List[np.ndarray])
- Returns: `np.ndarray` - Merged cloud (Nx3)

**capture_and_merge_multi_camera()**
- Complete workflow: capture → transform → merge
- Returns: `np.ndarray` - Merged cloud in world frame (Nx3)

## Examples

See the following files for complete examples:
- `test_multi_camera_integration.py` - Integration tests
- `test_requirement_9_verification.py` - Requirements verification
- `TASK11_IMPLEMENTATION_SUMMARY.md` - Implementation details

## Support

For issues or questions about multi-camera functionality:
1. Check this guide for common solutions
2. Review the test files for usage examples
3. Verify camera calibration is correct
4. Ensure point clouds are non-empty before merging
