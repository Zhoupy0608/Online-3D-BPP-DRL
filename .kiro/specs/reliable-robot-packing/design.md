# Design Document

## Overview

This design document specifies the technical approach for enhancing the existing Online 3D Bin Packing DRL system with reliability improvements based on the paper "Towards reliable robot packing system based on deep reinforcement learning". The enhancements address the gap between simulation and physical robot deployment by introducing:

1. **Visual Feedback Module** - Uses 3D cameras and point cloud processing to detect actual item positions
2. **Parallel Entry Motion Primitive** - Implements buffer space strategy to mitigate placement uncertainty
3. **Uncertainty-Aware Training** - Adds placement noise during training to improve robustness
4. **Enhanced State Representation** - Incorporates uncertainty information into the DRL agent's observations

The design maintains backward compatibility with the existing codebase while adding new modules that can be enabled/disabled through configuration flags.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DRL Training Loop                        │
│  ┌────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │   Policy   │───▶│ Action       │───▶│  Environment   │  │
│  │  Network   │    │ Selection    │    │  (Simulation)  │  │
│  └────────────┘    └──────────────┘    └────────────────┘  │
│        ▲                                        │            │
│        │                                        ▼            │
│        │                              ┌──────────────────┐  │
│        └──────────────────────────────│ State + Reward   │  │
│                                       └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Physical Robot Deployment                       │
│  ┌────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │   Trained  │───▶│ Parallel     │───▶│  Robot Motion  │  │
│  │   Policy   │    │ Entry Motion │    │  Controller    │  │
│  └────────────┘    └──────────────┘    └────────────────┘  │
│        ▲                                        │            │
│        │                                        ▼            │
│  ┌─────────────┐                      ┌──────────────────┐  │
│  │   Visual    │◀─────────────────────│  Item Placed     │  │
│  │   Feedback  │                      └──────────────────┘  │
│  │   Module    │                                             │
│  └─────────────┘                                             │
│        │                                                     │
│        ▼                                                     │
│  ┌──────────────────┐                                       │
│  │ Updated Height   │                                       │
│  │ Map for Next     │                                       │
│  │ Placement        │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction

1. **Training Phase**: DRL agent learns in simulation with added placement noise
2. **Deployment Phase**: Trained policy uses parallel entry motion and visual feedback for reliable physical packing

## Components and Interfaces

### 1. Visual Feedback Module

**Location**: `acktr/visual_feedback.py`

**Purpose**: Process 3D camera data to estimate actual box positions and update the height map

**Key Classes**:

```python
class VisualFeedbackModule:
    def __init__(self, camera_config, container_size):
        """
        Args:
            camera_config: Configuration for 3D cameras (intrinsics, extrinsics)
            container_size: Tuple (width, length, height) of container
        """
        
    def capture_point_cloud(self) -> np.ndarray:
        """Capture point cloud from 3D camera(s)"""
        
    def process_point_cloud(self, point_cloud: np.ndarray) -> List[Box]:
        """
        Extract box positions from point cloud
        Returns: List of detected boxes with positions
        """
        
    def update_height_map(self, height_map: np.ndarray, boxes: List[Box]) -> np.ndarray:
        """
        Update height map based on detected box positions
        Args:
            height_map: Current height map (width x length)
            boxes: List of detected boxes
        Returns: Updated height map
        """
```

**Dependencies**:
- Open3D for point cloud processing
- NumPy for array operations
- Camera SDK (simulated in training, real in deployment)

### 2. Parallel Entry Motion Module

**Location**: `acktr/motion_primitive.py`

**Purpose**: Generate and select motion primitives with buffer space to mitigate placement uncertainty

**Key Classes**:

```python
class ParallelEntryMotion:
    def __init__(self, buffer_range=(1, 1), container_size=(10, 10, 10)):
        """
        Args:
            buffer_range: (delta_x, delta_y) buffer space range
            container_size: Container dimensions
        """
        
    def generate_motion_options(self, 
                                target_pos: Tuple[int, int, int],
                                box_size: Tuple[int, int, int],
                                height_map: np.ndarray) -> List[MotionOption]:
        """
        Generate candidate motion options around target position
        Args:
            target_pos: (x, y, z) target position from DRL agent
            box_size: (lx, ly, lz) dimensions of box to place
            height_map: Current height map
        Returns: List of motion options with weights
        """
        
    def select_best_option(self, options: List[MotionOption]) -> MotionOption:
        """
        Select motion option with maximum weight
        Args:
            options: List of candidate motion options
        Returns: Selected motion option
        """
        
    def check_collision(self, option: MotionOption, height_map: np.ndarray) -> bool:
        """Check if motion option causes collision"""
```

**Data Structures**:

```python
@dataclass
class MotionOption:
    position: Tuple[int, int, int]  # (x, y, z) placement position
    weight: float                    # Weight based on height map and buffer
    buffer_space: Tuple[int, int]   # Available buffer in (x, y) directions
    collision_free: bool             # Whether option is collision-free
```

### 3. Uncertainty Simulation Module

**Location**: `acktr/uncertainty.py`

**Purpose**: Add placement noise during training to simulate real-world uncertainty

**Key Classes**:

```python
class UncertaintySimulator:
    def __init__(self, noise_std=(0.5, 0.5, 0.1), enabled=True):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise for (x, y, z)
            enabled: Whether to apply uncertainty simulation
        """
        
    def add_placement_noise(self, 
                           position: Tuple[int, int, int],
                           box_size: Tuple[int, int, int],
                           height_map: np.ndarray) -> Tuple[int, int, int]:
        """
        Add random noise to placement position
        Args:
            position: Intended placement position
            box_size: Box dimensions
            height_map: Current height map
        Returns: Perturbed position (adjusted if collision occurs)
        """
        
    def validate_position(self, 
                         position: Tuple[int, int, int],
                         box_size: Tuple[int, int, int],
                         height_map: np.ndarray) -> Tuple[int, int, int]:
        """
        Validate and adjust position if needed to avoid collisions
        """
```

### 4. Enhanced Environment

**Location**: `envs/bpp0/bin3D_reliable.py`

**Purpose**: Extended PackingGame environment with uncertainty and visual feedback support

**Key Modifications**:

```python
class ReliablePackingGame(PackingGame):
    def __init__(self, 
                 uncertainty_enabled=False,
                 visual_feedback_enabled=False,
                 parallel_motion_enabled=False,
                 **kwargs):
        """
        Extended environment with reliability features
        Args:
            uncertainty_enabled: Add placement noise during training
            visual_feedback_enabled: Use visual feedback for height map updates
            parallel_motion_enabled: Use parallel entry motion primitive
        """
        super().__init__(**kwargs)
        self.uncertainty_sim = UncertaintySimulator() if uncertainty_enabled else None
        self.visual_feedback = VisualFeedbackModule() if visual_feedback_enabled else None
        self.parallel_motion = ParallelEntryMotion() if parallel_motion_enabled else None
        
    def step(self, action):
        """
        Override step to incorporate uncertainty and visual feedback
        """
        # 1. Convert action to target position
        # 2. Apply parallel entry motion if enabled
        # 3. Add placement noise if uncertainty enabled
        # 4. Execute placement
        # 5. Update height map with visual feedback if enabled
        # 6. Return observation, reward, done, info
```

### 5. Enhanced Network Architecture

**Location**: `acktr/model.py` (modifications)

**Purpose**: No major changes needed - existing CNNPro architecture is sufficient

**Key Points**:
- Current architecture already handles height map + box size encoding
- Mask prediction branch can be used for candidate map generation
- No structural changes required, only training procedure modifications

## Data Models

### State Representation

```python
@dataclass
class PackingState:
    height_map: np.ndarray          # (width, length) - current height at each position
    next_box: Tuple[int, int, int]  # (lx, ly, lz) - dimensions of next box
    candidate_map: np.ndarray       # (width, length) or (width*length*2) with rotation
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to neural network input format"""
        # Shape: (4, width, length) for non-rotation
        # Channel 0: height_map
        # Channel 1: next_box[0] (length) repeated
        # Channel 2: next_box[1] (width) repeated  
        # Channel 3: next_box[2] (height) repeated
```

### Box Representation

```python
@dataclass
class Box:
    x: int  # length
    y: int  # width
    z: int  # height
    lx: int # x position in container
    ly: int # y position in container
    lz: int # z position (height) in container
    rotated: bool = False  # whether box is rotated
```

### Camera Configuration

```python
@dataclass
class CameraConfig:
    intrinsics: np.ndarray  # 3x3 camera intrinsic matrix
    extrinsics: np.ndarray  # 4x4 camera extrinsic matrix (world to camera)
    resolution: Tuple[int, int]  # (width, height) in pixels
    depth_range: Tuple[float, float]  # (min_depth, max_depth) in meters
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property Reflection

After reviewing all testable properties from the prework analysis, I identify the following redundancies:

- Property 6.4 (rotation candidate maps) is redundant with Property 3.4
- Property 10.2 (space utilization calculation) is redundant with Property 4.2

These redundant properties will be consolidated into single comprehensive properties.

### Core Correctness Properties

**Property 1: Point cloud coordinate transformation preserves spatial relationships**
*For any* point cloud and camera transformation matrix, transforming points to the container coordinate system should preserve relative distances and angles between points
**Validates: Requirements 1.2**

**Property 2: Box extraction identifies all boxes in point cloud**
*For any* synthetic point cloud containing known boxes, the extraction algorithm should identify boxes with positions within acceptable error tolerance
**Validates: Requirements 1.3**

**Property 3: Height map update reflects detected box positions**
*For any* set of detected boxes, the updated height map should have heights at box footprints equal to the maximum z-coordinate of boxes at those positions
**Validates: Requirements 1.4**

**Property 4: Motion options are within buffer range**
*For any* target position and buffer range (Δx, Δy), all generated motion options should have positions within [target_x - Δx, target_x + Δx] and [target_y - Δy, target_y + Δy]
**Validates: Requirements 2.2**

**Property 5: Motion option weights are deterministic**
*For any* motion option and height map, calculating the weight twice should produce identical results
**Validates: Requirements 2.3**

**Property 6: Selected motion option has maximum weight**
*For any* non-empty list of collision-free motion options, the selected option should have weight greater than or equal to all other options
**Validates: Requirements 2.4**

**Property 7: Collision detection prevents invalid placements**
*For any* motion option that would cause a box to exceed container boundaries or overlap with existing boxes, collision checking should return False
**Validates: Requirements 2.5**

**Property 8: State encoding includes all required components**
*For any* container state, the encoded tensor should contain height map, next box dimensions, and candidate map with correct shapes
**Validates: Requirements 3.1, 3.2, 3.5**

**Property 9: Candidate map marks only valid positions**
*For any* item and container state, all positions marked as valid (1) in the candidate map should allow placement without exceeding container boundaries
**Validates: Requirements 3.3, 6.1**

**Property 10: Rotation generates two candidate maps**
*For any* item when rotation is enabled, the system should generate exactly two candidate maps (original orientation and 90-degree rotation)
**Validates: Requirements 3.4**

**Property 11: Space utilization is bounded**
*For any* container state, the calculated space utilization should be in the range [0, 1] and equal to (sum of packed box volumes) / (container volume)
**Validates: Requirements 4.2**

**Property 12: Successful placement increases utilization**
*For any* successful box placement, the space utilization after placement should be greater than before placement
**Validates: Requirements 4.1**

**Property 13: Invalid placement receives penalty**
*For any* invalid placement attempt, the reward should be less than or equal to zero
**Validates: Requirements 4.3**

**Property 14: Buffer space validation**
*For any* position when buffer space is enabled, if the position is marked valid, there should exist minimum clearance distance in x and y directions
**Validates: Requirements 6.2, 6.3**

**Property 15: Height map monotonicity**
*For any* height map update operation, the new height at each grid cell should be greater than or equal to the old height (heights never decrease)
**Validates: Requirements 7.4**

**Property 16: Grid cell height is maximum of overlapping boxes**
*For any* grid cell covered by multiple boxes, the height value should equal the maximum z-coordinate among all boxes covering that cell
**Validates: Requirements 7.5**

**Property 17: Placement noise follows Gaussian distribution**
*For any* large sample of placement noise values, the distribution should approximate a Gaussian with the configured mean and standard deviation
**Validates: Requirements 8.2**

**Property 18: Perturbed positions are valid**
*For any* placement position with added noise, the final perturbed position should not cause collisions or boundary violations
**Validates: Requirements 8.3**

**Property 19: Point cloud merging preserves all points**
*For any* set of point clouds in a common coordinate frame, the merged point cloud should contain all points from all input clouds
**Validates: Requirements 9.4**

**Property 20: Performance metrics are correctly computed**
*For any* set of packing episodes with known utilization values, the computed mean and standard deviation should match the statistical definitions
**Validates: Requirements 10.3**

## Error Handling

### Visual Feedback Errors

1. **Camera Failure**: If camera capture fails, log error and use predicted height map without visual correction
2. **Point Cloud Processing Failure**: If box extraction fails, fall back to predicted positions
3. **Invalid Box Detection**: If detected box positions are outside container, ignore and use predicted state

### Motion Primitive Errors

1. **No Valid Motion Options**: If all motion options result in collisions, return failure and skip item
2. **Buffer Space Unavailable**: If no positions have sufficient buffer, relax buffer constraints gradually
3. **Collision After Selection**: If selected option causes collision, try next best option up to N attempts

### Training Errors

1. **Noise Causes Invalid State**: If noise makes position invalid, adjust to nearest valid position
2. **Convergence Issues**: If training doesn't converge, reduce noise standard deviation
3. **Memory Overflow**: If point cloud processing uses too much memory, downsample point cloud

### Deployment Errors

1. **Robot Communication Failure**: Halt packing and report error to operator
2. **Visual Feedback Timeout**: Use predicted state and continue with warning
3. **Calibration Drift**: Periodically re-calibrate cameras using known reference objects

## Testing Strategy

### Unit Testing

Unit tests will verify specific functionality of individual components:

1. **Coordinate Transformation Tests**
   - Test point cloud projection with known camera parameters
   - Verify transformation matrices are applied correctly
   - Test edge cases (points at boundaries, behind camera)

2. **Motion Primitive Tests**
   - Test motion option generation for various target positions
   - Verify weight calculation formula
   - Test collision detection with known configurations

3. **Height Map Update Tests**
   - Test height map updates with single box
   - Test with multiple overlapping boxes
   - Verify monotonicity property

4. **Candidate Map Tests**
   - Test candidate map generation for various box sizes
   - Verify boundary checking
   - Test rotation handling

### Property-Based Testing

Property-based tests will verify universal properties across many randomly generated inputs using **Hypothesis** (Python PBT library):

**Configuration**: Each property test will run a minimum of 100 iterations with randomly generated inputs.

**Test Tagging**: Each property-based test will include a comment with the format:
`# Feature: reliable-robot-packing, Property {number}: {property_text}`

**Key Property Tests**:

1. **Coordinate Transformation Property** (Property 1)
   - Generate random point clouds and camera parameters
   - Verify distance preservation after transformation

2. **Motion Option Range Property** (Property 4)
   - Generate random target positions and buffer ranges
   - Verify all options are within specified range

3. **Weight Determinism Property** (Property 5)
   - Generate random motion options and height maps
   - Verify weight calculation is deterministic

4. **Maximum Weight Selection Property** (Property 6)
   - Generate random lists of motion options
   - Verify selected option has maximum weight

5. **Candidate Map Validity Property** (Property 9)
   - Generate random items and container states
   - Verify all marked positions are valid

6. **Space Utilization Bounds Property** (Property 11)
   - Generate random container states
   - Verify utilization is in [0, 1]

7. **Height Map Monotonicity Property** (Property 15)
   - Generate random height map updates
   - Verify heights never decrease

8. **Gaussian Noise Property** (Property 17)
   - Generate large sample of noise values
   - Verify distribution matches Gaussian

### Integration Testing

Integration tests will verify component interactions:

1. **Visual Feedback Integration**
   - Test full pipeline: capture → process → update height map
   - Verify DRL agent receives updated state

2. **Motion Primitive Integration**
   - Test DRL action → motion options → selection → execution
   - Verify fallback behavior when collisions occur

3. **End-to-End Packing**
   - Test complete packing episodes with all features enabled
   - Compare performance with and without reliability features

### Testing Framework

- **Unit Tests**: pytest
- **Property-Based Tests**: Hypothesis
- **Integration Tests**: pytest with fixtures
- **Performance Tests**: Custom benchmarking scripts

### Test Data

- **Synthetic Point Clouds**: Generated with known box positions for validation
- **Real Point Clouds**: Captured from physical setup (if available)
- **Packing Scenarios**: CUT-1, CUT-2, and random sequences from existing datasets

## Implementation Notes

### Backward Compatibility

All new features are opt-in through configuration flags:
- `uncertainty_enabled`: Enable placement noise during training
- `visual_feedback_enabled`: Enable visual feedback module
- `parallel_motion_enabled`: Enable parallel entry motion primitive

Existing code continues to work without modifications when flags are disabled.

### Performance Considerations

1. **Point Cloud Processing**: Use downsampling for large point clouds to maintain real-time performance
2. **Motion Option Generation**: Limit search range to avoid excessive computation
3. **Visual Feedback Frequency**: Update height map every N placements rather than every placement

### Dependencies

New dependencies to add to `requirements.txt`:
```
open3d>=0.15.0  # Point cloud processing
hypothesis>=6.0.0  # Property-based testing
```

### Configuration Parameters

New parameters to add to `arguments.py`:
```python
--uncertainty-enabled: Enable placement noise during training
--uncertainty-std: Standard deviation for placement noise (default: 0.5, 0.5, 0.1)
--visual-feedback-enabled: Enable visual feedback module
--parallel-motion-enabled: Enable parallel entry motion
--buffer-range: Buffer space range (default: 1, 1)
--camera-config: Path to camera configuration file
```

## Future Enhancements

1. **Adaptive Buffer Space**: Dynamically adjust buffer space based on observed placement errors
2. **Learning-Based Uncertainty**: Learn uncertainty model from real robot data
3. **Multi-Item Lookahead**: Extend parallel motion to consider multiple future items
4. **Real-Time Calibration**: Continuously update camera calibration during operation
5. **Irregular Items**: Extend to handle non-rectangular items using point cloud matching
