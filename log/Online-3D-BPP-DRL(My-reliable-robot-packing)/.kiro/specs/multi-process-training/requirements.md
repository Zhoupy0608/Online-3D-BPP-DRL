# Requirements Document

## Introduction

This document specifies the requirements for implementing multi-process training for the Online 3D Bin Packing with Deep Reinforcement Learning system, aligning the implementation with the original paper's methodology. The system currently only supports single-process training, which is significantly slower than the paper's multi-process approach. This feature will enable parallel environment simulation and faster convergence to match the paper's reported results.

## Glossary

- **ACKTR**: Actor-Critic using Kronecker-Factored Trust Region, the reinforcement learning algorithm used in the paper
- **KFAC**: Kronecker-Factored Approximate Curvature, the second-order optimization method
- **Multi-Process Training**: Training approach where multiple environment instances run in parallel processes
- **ShmemVecEnv**: Shared memory vectorized environment for efficient multi-process communication
- **DummyVecEnv**: Single-process vectorized environment wrapper
- **Rollout**: A sequence of state-action-reward transitions collected during training
- **Episode**: A complete packing sequence from start to finish
- **Space Utilization**: The ratio of packed item volume to container volume
- **Height Map**: 2D representation of the container's current packing state

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to train the model using multiple parallel processes, so that I can achieve faster training and match the paper's reported performance.

#### Acceptance Criteria

1. WHEN the system starts multi-process training THEN the system SHALL create multiple parallel environment instances equal to the num_processes parameter
2. WHEN multiple processes are running THEN the system SHALL use shared memory for efficient inter-process communication
3. WHEN collecting rollouts THEN the system SHALL aggregate experiences from all parallel environments
4. WHEN updating the model THEN the system SHALL use batched gradients from all processes
5. WHEN training completes THEN the system SHALL achieve comparable space utilization to the paper's results (70%+ for cut_2 dataset)

### Requirement 2

**User Story:** As a developer, I want the KFAC optimizer to work correctly with multi-process training, so that the second-order optimization benefits are preserved.

#### Acceptance Criteria

1. WHEN computing Fisher information matrices THEN the system SHALL accumulate statistics across all parallel processes
2. WHEN updating parameters THEN the system SHALL apply KFAC updates using the aggregated Fisher matrices
3. WHEN the Kronecker factorization is computed THEN the system SHALL handle numerical stability issues with eigenvalue thresholding
4. WHEN device mismatches occur THEN the system SHALL automatically move tensors to the correct device
5. WHEN training progresses THEN the system SHALL maintain stable gradient updates without NaN or Inf values

### Requirement 3

**User Story:** As a system administrator, I want the training to be compatible with both Windows and Linux platforms, so that researchers can use their preferred operating system.

#### Acceptance Criteria

1. WHEN running on Windows THEN the system SHALL use spawn context for multiprocessing
2. WHEN running on Linux THEN the system SHALL use fork context for multiprocessing
3. WHEN initializing processes THEN the system SHALL properly serialize environment parameters
4. WHEN processes communicate THEN the system SHALL handle platform-specific shared memory differences
5. WHEN training starts THEN the system SHALL detect the platform and configure multiprocessing accordingly

### Requirement 4

**User Story:** As a researcher, I want proper synchronization between parallel environments, so that training is stable and reproducible.

#### Acceptance Criteria

1. WHEN environments reset THEN the system SHALL use different random seeds for each process
2. WHEN collecting rollouts THEN the system SHALL synchronize all processes at step boundaries
3. WHEN an episode completes in any process THEN the system SHALL properly handle the reset without blocking other processes
4. WHEN computing returns THEN the system SHALL correctly apply discount factors across episode boundaries
5. WHEN saving checkpoints THEN the system SHALL ensure all processes are synchronized

### Requirement 5

**User Story:** As a researcher, I want to monitor training progress across all processes, so that I can verify the system is learning effectively.

#### Acceptance Criteria

1. WHEN episodes complete THEN the system SHALL aggregate rewards from all processes
2. WHEN logging statistics THEN the system SHALL report mean, median, min, and max rewards across processes
3. WHEN using TensorBoard THEN the system SHALL log aggregated metrics from all parallel environments
4. WHEN training progresses THEN the system SHALL display space utilization statistics
5. WHEN reliability features are enabled THEN the system SHALL track feature usage across all processes

### Requirement 6

**User Story:** As a developer, I want proper memory management in multi-process training, so that the system can scale to many parallel environments without memory issues.

#### Acceptance Criteria

1. WHEN allocating rollout storage THEN the system SHALL size buffers based on num_processes and num_steps
2. WHEN sharing observations THEN the system SHALL use shared memory tensors to avoid duplication
3. WHEN processes terminate THEN the system SHALL properly clean up shared memory resources
4. WHEN GPU is used THEN the system SHALL manage device memory efficiently across processes
5. WHEN training runs for extended periods THEN the system SHALL not exhibit memory leaks

### Requirement 7

**User Story:** As a researcher, I want the training hyperparameters to match the paper's configuration, so that I can reproduce the published results.

#### Acceptance Criteria

1. WHEN training starts THEN the system SHALL use 16 parallel processes as specified in the paper
2. WHEN collecting rollouts THEN the system SHALL use 5 forward steps per update as specified in the paper
3. WHEN computing loss THEN the system SHALL use value_loss_coef=0.5, entropy_coef=0.01, invalid_coef=2
4. WHEN using KFAC THEN the system SHALL use stat_decay=0.99, kl_clip=0.001, damping=1e-2
5. WHEN training progresses THEN the system SHALL save models at regular intervals

### Requirement 8

**User Story:** As a researcher, I want proper error handling in multi-process training, so that failures in one process don't crash the entire training run.

#### Acceptance Criteria

1. WHEN a process encounters an error THEN the system SHALL log the error with process ID and stack trace
2. WHEN a process crashes THEN the system SHALL attempt to restart the process
3. WHEN multiple processes fail THEN the system SHALL gracefully terminate training
4. WHEN KFAC encounters numerical issues THEN the system SHALL apply eigenvalue thresholding and continue
5. WHEN device errors occur THEN the system SHALL provide clear error messages about GPU availability

### Requirement 9

**User Story:** As a researcher, I want to validate that multi-process training produces correct results, so that I can trust the trained models.

#### Acceptance Criteria

1. WHEN training completes THEN the system SHALL produce models with space utilization within 5% of paper results
2. WHEN comparing single-process and multi-process training THEN the system SHALL produce similar final performance
3. WHEN testing trained models THEN the system SHALL achieve consistent packing quality
4. WHEN using different num_processes values THEN the system SHALL scale training speed proportionally
5. WHEN reliability features are enabled THEN the system SHALL maintain robustness across all processes
