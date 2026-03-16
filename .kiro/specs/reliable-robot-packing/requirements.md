# Requirements Document

## Introduction

This document specifies the requirements for enhancing the existing Online 3D Bin Packing with Deep Reinforcement Learning system to incorporate reliability improvements based on the paper "Towards reliable robot packing system based on deep reinforcement learning" (Advanced Engineering Informatics 57, 2023). The enhancements focus on uncertainty mitigation, visual feedback integration, and improved robot motion primitives to achieve more reliable physical robot packing.

## Glossary

- **DRL Agent**: The Deep Reinforcement Learning agent that predicts packing actions and placement positions
- **Parallel Entry Motion**: A robot motion primitive where items are placed with buffer space to mitigate placement uncertainty
- **Visual Feedback Module**: A system using 3D cameras and point cloud processing to estimate actual box positions after placement
- **Height Map**: A 2D representation of the container state showing the maximum height at each (x,y) position
- **Candidate Map**: A binary map indicating feasible placement positions for the current item
- **Uncertainty Mitigation**: Techniques to handle placement errors and physical uncertainties in robotic packing
- **Packing Container**: The physical bin or container where items are placed (dimensions: width × length × height)
- **Item Sequence**: The ordered list of items to be packed into the container
- **Space Utilization**: The ratio of packed item volume to total container volume
- **Collision Detection**: The process of checking whether a placement would cause items to overlap

## Requirements

### Requirement 1: Visual Feedback System Integration

**User Story:** As a robotic packing system operator, I want the system to use visual feedback to detect actual item positions, so that placement errors can be corrected and subsequent packing decisions are based on real container state.

#### Acceptance Criteria

1. WHEN an item is placed in the container, THE Visual Feedback Module SHALL capture point cloud data from the 3D camera
2. WHEN point cloud data is captured, THE Visual Feedback Module SHALL project the point cloud to the container coordinate system
3. WHEN the point cloud is projected, THE Visual Feedback Module SHALL filter noise and extract box positions using region growing and orthogonal fitting
4. WHEN box positions are extracted, THE Visual Feedback Module SHALL update the height map based on actual measured positions
5. WHEN the height map is updated, THE DRL Agent SHALL use the corrected height map for subsequent placement decisions

### Requirement 2: Parallel Entry Motion Primitive

**User Story:** As a robotic packing system, I want to implement parallel entry motion with buffer space, so that placement uncertainty is mitigated and items are placed more reliably.

#### Acceptance Criteria

1. WHEN the DRL Agent predicts a target placement position (x_t, y_t, z_t), THE System SHALL generate parallel entry motion options with buffer space
2. WHEN generating motion options, THE System SHALL create subgrid options within a range (Δx, Δy) around the target position
3. WHEN evaluating motion options, THE System SHALL assign weights based on the sum of height map values and buffer space availability
4. WHEN a motion option is selected, THE System SHALL sort options by weight and select the option with maximum weight
5. WHEN the selected option requires collision checking, THE System SHALL verify no collision occurs before executing the motion
6. WHEN collision is detected, THE System SHALL select the next best option from the sorted list

### Requirement 3: Uncertainty-Aware State Representation

**User Story:** As a DRL agent, I want an enhanced state representation that accounts for placement uncertainty, so that I can make more robust packing decisions.

#### Acceptance Criteria

1. WHEN encoding the container state, THE System SHALL include the height map as a primary state component
2. WHEN encoding the next item, THE System SHALL include item dimensions (length, width, height) in the state representation
3. WHEN generating the candidate map, THE System SHALL mark positions where the item can be placed without exceeding container boundaries
4. WHEN rotation is enabled, THE System SHALL generate separate candidate maps for each rotation orientation
5. WHEN the state is complete, THE System SHALL concatenate height map, item features, and candidate map into a unified tensor

### Requirement 4: Enhanced Reward Function

**User Story:** As a training system, I want an improved reward function that encourages stable and efficient packing, so that the trained model achieves better space utilization and physical stability.

#### Acceptance Criteria

1. WHEN an item is successfully placed, THE System SHALL calculate reward based on the change in container space utilization
2. WHEN calculating space utilization, THE System SHALL compute the ratio of packed volume to total container volume
3. WHEN an invalid placement is attempted, THE System SHALL apply a penalty to the reward
4. WHEN the packing episode completes, THE System SHALL provide a terminal reward based on final space utilization
5. WHEN stability constraints are violated, THE System SHALL apply additional penalties to discourage unstable configurations

### Requirement 5: Robotic Motion Execution

**User Story:** As a physical robot, I want to execute packing motions safely and reliably, so that items are placed correctly without collisions or damage.

#### Acceptance Criteria

1. WHEN a placement action is selected, THE System SHALL compute the robot trajectory from current pose to target pose
2. WHEN computing the trajectory, THE System SHALL ensure the trajectory avoids collisions with placed items and container walls
3. WHEN executing the motion, THE System SHALL move the robot along the computed trajectory with appropriate speed
4. WHEN the robot reaches the target position, THE System SHALL release the item and retract to a safe position
5. WHEN motion execution fails, THE System SHALL report the error and halt the packing process

### Requirement 6: Candidate Map Generation with Buffer Space

**User Story:** As a DRL agent, I want candidate maps that account for buffer space requirements, so that placements have adequate clearance for uncertainty mitigation.

#### Acceptance Criteria

1. WHEN generating a candidate map, THE System SHALL identify all positions where the item fits within container boundaries
2. WHEN buffer space is enabled, THE System SHALL mark positions as invalid if insufficient buffer space exists around the placement
3. WHEN checking buffer space, THE System SHALL verify a minimum clearance distance in x and y directions
4. WHEN rotation is enabled, THE System SHALL generate candidate maps for both original and rotated orientations
5. WHEN all candidate maps are generated, THE System SHALL pass them to the DRL Agent as part of the state representation

### Requirement 7: Height Map Update from Visual Feedback

**User Story:** As a packing system, I want to update the height map based on actual measured positions, so that the internal state reflects the real physical configuration.

#### Acceptance Criteria

1. WHEN visual feedback provides box position measurements, THE System SHALL extract the bounding box coordinates (x_min, x_max, y_min, y_max, z_max)
2. WHEN bounding box coordinates are extracted, THE System SHALL map them to the discretized height map grid
3. WHEN mapping to the grid, THE System SHALL update all grid cells within the box footprint to the measured height
4. WHEN the height map is updated, THE System SHALL ensure height values are non-decreasing (new items cannot lower the height)
5. WHEN multiple boxes overlap in the grid, THE System SHALL use the maximum height value for each grid cell

### Requirement 8: Training with Simulated Uncertainty

**User Story:** As a training system, I want to simulate placement uncertainty during training, so that the learned policy is robust to real-world errors.

#### Acceptance Criteria

1. WHEN training in simulation, THE System SHALL add random noise to item placement positions
2. WHEN adding placement noise, THE System SHALL sample noise from a Gaussian distribution with configurable standard deviation
3. WHEN noise is applied, THE System SHALL ensure the perturbed position does not cause collisions or boundary violations
4. WHEN collisions occur due to noise, THE System SHALL adjust the position to the nearest valid location
5. WHEN training completes, THE System SHALL save the trained model with uncertainty-aware parameters

### Requirement 9: Multi-Camera Point Cloud Fusion

**User Story:** As a visual feedback system, I want to fuse point clouds from multiple cameras, so that occlusions are minimized and position estimates are more accurate.

#### Acceptance Criteria

1. WHEN multiple cameras are available, THE System SHALL capture point clouds from all cameras simultaneously
2. WHEN point clouds are captured, THE System SHALL transform each point cloud to a common coordinate frame
3. WHEN transforming point clouds, THE System SHALL use calibrated camera extrinsic parameters
4. WHEN point clouds are in the common frame, THE System SHALL merge them into a single unified point cloud
5. WHEN the unified point cloud is created, THE System SHALL use it for box position estimation

### Requirement 10: Performance Evaluation and Metrics

**User Story:** As a system evaluator, I want comprehensive metrics to assess packing performance, so that I can compare different methods and configurations.

#### Acceptance Criteria

1. WHEN a packing episode completes, THE System SHALL calculate the final space utilization ratio
2. WHEN calculating space utilization, THE System SHALL divide the total packed volume by the container volume
3. WHEN evaluating over multiple episodes, THE System SHALL compute mean and standard deviation of space utilization
4. WHEN comparing with baselines, THE System SHALL report the performance improvement percentage
5. WHEN physical robot experiments are conducted, THE System SHALL record success rate and execution time for each packing task
