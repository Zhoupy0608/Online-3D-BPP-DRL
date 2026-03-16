# Requirements Document

## Introduction

This document specifies requirements for improving the training performance of the reliable robot packing system. Current training shows low space utilization (15-23%), high reward variance, and instability. This spec addresses hyperparameter tuning, training strategies, and reward shaping to achieve better convergence and performance.

## Glossary

- **Space Utilization**: The ratio of packed item volume to total container volume (target: >40%)
- **Reward Variance**: The standard deviation of episode rewards (lower is better for stable training)
- **Learning Rate**: The step size for gradient descent updates in the neural network
- **Entropy Coefficient**: Weight for the entropy term that encourages exploration
- **Value Loss Coefficient**: Weight for the value function loss in the actor-critic algorithm
- **ACKTR**: Actor-Critic using Kronecker-Factored Trust Region, the RL algorithm being used
- **Rollout Steps**: Number of environment steps before performing a policy update
- **Training Episodes**: Complete packing sequences from start to finish
- **Convergence**: The point where training performance stabilizes and stops improving significantly

## Requirements

### Requirement 1: Extended Training Duration

**User Story:** As a training system, I want to train for sufficient timesteps, so that the policy has enough experience to converge to good performance.

#### Acceptance Criteria

1. WHEN starting training, THE System SHALL run for a minimum of 1,000,000 timesteps
2. WHEN tracking progress, THE System SHALL log performance metrics every 10 updates
3. WHEN 1,000,000 timesteps are reached, THE System SHALL evaluate whether further training improves performance
4. WHEN performance plateaus, THE System SHALL allow early stopping to save computation time
5. WHEN training completes, THE System SHALL save the best performing model checkpoint

### Requirement 2: Learning Rate Optimization

**User Story:** As a training system, I want an appropriate learning rate, so that the policy learns effectively without instability.

#### Acceptance Criteria

1. WHEN initializing training, THE System SHALL use a learning rate between 1e-4 and 1e-3 for ACKTR
2. WHEN training shows instability (large value loss spikes), THE System SHALL reduce the learning rate by a factor of 2
3. WHEN training shows slow progress, THE System SHALL increase the learning rate by a factor of 1.5
4. WHEN adjusting learning rate, THE System SHALL log the change and monitor subsequent performance
5. WHEN optimal learning rate is found, THE System SHALL maintain it for the remainder of training

### Requirement 3: Reward Function Tuning

**User Story:** As a training system, I want a well-shaped reward function, so that the agent receives clear learning signals for good packing behavior.

#### Acceptance Criteria

1. WHEN an item is placed successfully, THE System SHALL provide immediate positive reward proportional to volume utilization increase
2. WHEN an invalid action is attempted, THE System SHALL apply a penalty of at least -1.0
3. WHEN the episode ends, THE System SHALL provide a terminal bonus based on final space utilization
4. WHEN items are placed with good stability, THE System SHALL provide additional reward for height map smoothness
5. WHEN reward components are combined, THE System SHALL normalize them to similar scales to prevent dominance

### Requirement 4: Entropy Coefficient Adjustment

**User Story:** As a training system, I want appropriate exploration-exploitation balance, so that the agent explores sufficiently early but exploits learned knowledge later.

#### Acceptance Criteria

1. WHEN training begins, THE System SHALL use an entropy coefficient of 0.01 to encourage exploration
2. WHEN training reaches 25% of total timesteps, THE System SHALL reduce entropy coefficient to 0.005
3. WHEN training reaches 50% of total timesteps, THE System SHALL reduce entropy coefficient to 0.001
4. WHEN entropy coefficient changes, THE System SHALL log the adjustment and monitor policy diversity
5. WHEN policy becomes too deterministic prematurely, THE System SHALL increase entropy coefficient temporarily

### Requirement 5: Batch Size and Rollout Optimization

**User Story:** As a training system, I want optimal batch sizes and rollout lengths, so that gradient estimates are accurate and training is efficient.

#### Acceptance Criteria

1. WHEN configuring training, THE System SHALL use rollout steps between 5 and 20
2. WHEN using single-process training, THE System SHALL set num_processes to 1
3. WHEN memory allows, THE System SHALL increase rollout steps to improve sample efficiency
4. WHEN training is unstable, THE System SHALL reduce rollout steps to 5 for more frequent updates
5. WHEN rollout configuration changes, THE System SHALL monitor impact on training stability and speed

### Requirement 6: Curriculum Learning Strategy

**User Story:** As a training system, I want to gradually increase task difficulty, so that the agent learns basic skills before tackling complex scenarios.

#### Acceptance Criteria

1. WHEN training starts, THE System SHALL use small containers (5x5x5) and simple item sequences
2. WHEN the agent achieves 30% space utilization on simple tasks, THE System SHALL increase container size to 8x8x8
3. WHEN the agent achieves 35% space utilization on medium tasks, THE System SHALL use full-size containers (10x10x10)
4. WHEN transitioning difficulty levels, THE System SHALL maintain the learned policy weights
5. WHEN curriculum stages complete, THE System SHALL train on the final difficulty until convergence

### Requirement 7: Uncertainty Level Scheduling

**User Story:** As a training system, I want to gradually increase placement uncertainty, so that the agent first learns ideal packing then adapts to noise.

#### Acceptance Criteria

1. WHEN training begins, THE System SHALL use zero placement noise (uncertainty_std = 0.0)
2. WHEN the agent achieves stable performance without noise, THE System SHALL introduce small noise (std = 0.1)
3. WHEN the agent adapts to small noise, THE System SHALL increase to medium noise (std = 0.3)
4. WHEN the agent adapts to medium noise, THE System SHALL increase to target noise level (std = 0.5)
5. WHEN noise level changes, THE System SHALL allow 100,000 timesteps for adaptation before further increases

### Requirement 8: Value Loss Coefficient Tuning

**User Story:** As a training system, I want balanced actor and critic learning, so that both policy and value function improve together.

#### Acceptance Criteria

1. WHEN training begins, THE System SHALL use value_loss_coef of 0.5
2. WHEN value loss consistently exceeds 50.0, THE System SHALL reduce value_loss_coef to 0.25
3. WHEN value loss is consistently below 5.0, THE System SHALL increase value_loss_coef to 0.75
4. WHEN adjusting value_loss_coef, THE System SHALL monitor both value loss and policy performance
5. WHEN optimal balance is achieved, THE System SHALL maintain the coefficient for remaining training

### Requirement 9: Model Checkpointing Strategy

**User Story:** As a training system, I want to save models at key milestones, so that I can recover from failures and compare different training stages.

#### Acceptance Criteria

1. WHEN training progresses, THE System SHALL save checkpoints every 100 updates
2. WHEN a new best performance is achieved, THE System SHALL save a separate "best_model" checkpoint
3. WHEN saving checkpoints, THE System SHALL include training step number, performance metrics, and hyperparameters
4. WHEN disk space is limited, THE System SHALL keep only the 5 most recent checkpoints plus the best model
5. WHEN training is interrupted, THE System SHALL allow resuming from the latest checkpoint

### Requirement 10: Training Monitoring and Diagnostics

**User Story:** As a training operator, I want comprehensive monitoring, so that I can identify issues and make informed adjustments.

#### Acceptance Criteria

1. WHEN training runs, THE System SHALL log mean reward, space utilization, and loss values every 10 updates
2. WHEN logging metrics, THE System SHALL compute rolling statistics over the last 100 episodes
3. WHEN anomalies are detected (NaN losses, reward collapse), THE System SHALL alert the operator
4. WHEN using TensorBoard, THE System SHALL visualize reward curves, loss curves, and space utilization trends
5. WHEN training completes, THE System SHALL generate a summary report with final performance and training statistics

