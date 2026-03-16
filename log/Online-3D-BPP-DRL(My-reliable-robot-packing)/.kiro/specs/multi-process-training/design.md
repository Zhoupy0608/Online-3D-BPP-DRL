# Design Document: Multi-Process Training for Online 3D Bin Packing

## Overview

This document describes the design for implementing multi-process training in the Online 3D Bin Packing with Deep Reinforcement Learning system. The current implementation only supports single-process training, which is significantly slower than the paper's approach. This design will enable parallel environment simulation using 16 processes (as specified in the paper), allowing for faster training and better sample efficiency.

The key challenge is integrating multi-process environment simulation with the KFAC (Kronecker-Factored Approximate Curvature) optimizer while maintaining numerical stability and cross-platform compatibility. The design follows the paper's methodology closely to ensure reproducible results.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop (main.py)                  │
│  - Manages training iterations                               │
│  - Coordinates rollout collection and model updates          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├──────────────────────────────────────────┐
                 │                                          │
┌────────────────▼──────────────┐    ┌────────────────────▼──────────┐
│   Vectorized Environment       │    │   ACKTR Agent                 │
│   (ShmemVecEnv)                │    │   (acktr_pipeline.py)         │
│                                │    │                               │
│  ┌──────────────────────────┐ │    │  ┌─────────────────────────┐ │
│  │ Process 1: PackingGame   │ │    │  │  Actor-Critic Network   │ │
│  └──────────────────────────┘ │    │  └─────────────────────────┘ │
│  ┌──────────────────────────┐ │    │  ┌─────────────────────────┐ │
│  │ Process 2: PackingGame   │ │    │  │  KFAC Optimizer         │ │
│  └──────────────────────────┘ │    │  └─────────────────────────┘ │
│  ┌──────────────────────────┐ │    │  ┌─────────────────────────┐ │
│  │ ...                      │ │    │  │  Rollout Storage        │ │
│  └──────────────────────────┘ │    │  └─────────────────────────┘ │
│  ┌──────────────────────────┐ │    └─────────────────────────────────┘
│  │ Process 16: PackingGame  │ │
│  └──────────────────────────┘ │
│                                │
│  Shared Memory Communication   │
└────────────────────────────────┘
```

### Component Interaction Flow

1. **Initialization Phase**:
   - Main process creates 16 environment instances
   - Each environment is wrapped in a separate process
   - Shared memory is allocated for observations and actions
   - KFAC optimizer initializes Fisher information matrices

2. **Rollout Collection Phase**:
   - All 16 processes run environments in parallel
   - Each process collects num_steps (5) transitions
   - Observations, actions, rewards are stored in shared memory
   - Rollout storage aggregates data from all processes

3. **Update Phase**:
   - Batch of (num_steps × num_processes) transitions is processed
   - KFAC computes Fisher information from all processes
   - Gradients are computed and natural gradient update is applied
   - Model parameters are updated using KFAC optimizer

4. **Synchronization Phase**:
   - All processes synchronize at step boundaries
   - Episode statistics are aggregated across processes
   - Checkpoints are saved periodically

## Components and Interfaces

### 1. Vectorized Environment Wrapper (envs.py)

**Purpose**: Manages multiple environment instances running in parallel processes.

**Key Functions**:

```python
def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, device, 
                  allow_early_resets, num_frame_stack=None, args=None):
    """
    Create vectorized environments with proper multiprocessing context.
    
    Args:
        env_name: Environment identifier ('Bpp-v0' or 'BppReliable-v0')
        seed: Base random seed
        num_processes: Number of parallel processes (default: 16)
        gamma: Discount factor
        log_dir: Directory for logging
        device: Torch device (CPU or CUDA)
        allow_early_resets: Whether to allow early episode resets
        args: Additional arguments including reliability features
        
    Returns:
        VecPyTorch: Vectorized environment wrapper
    """
```

**Implementation Details**:
- Uses `ShmemVecEnv` for multi-process (num_processes > 1)
- Uses `DummyVecEnv` for single-process (num_processes == 1)
- Automatically selects 'spawn' context on Windows, 'fork' on Linux
- Each process gets unique seed: base_seed + process_rank
- Wraps environments with VecNormalize for observation normalization
- Wraps with VecPyTorch for tensor conversion and device management

### 2. KFAC Optimizer (kfac.py)

**Purpose**: Implements Kronecker-Factored Approximate Curvature optimization for efficient second-order updates.

**Key Functions**:

```python
class KFACOptimizer(optim.Optimizer):
    def __init__(self, model, lr=0.25, momentum=0.9, stat_decay=0.99,
                 kl_clip=0.001, damping=1e-2, weight_decay=0,
                 fast_cnn=False, Ts=1, Tf=10):
        """
        Initialize KFAC optimizer.
        
        Args:
            model: Neural network model
            lr: Learning rate (default: 0.25)
            momentum: Momentum coefficient (default: 0.9)
            stat_decay: Decay rate for running statistics (default: 0.99)
            kl_clip: KL divergence clipping threshold (default: 0.001)
            damping: Damping factor for numerical stability (default: 1e-2)
            weight_decay: L2 regularization coefficient (default: 0)
            fast_cnn: Use fast CNN approximation (default: False)
            Ts: Statistics update frequency (default: 1)
            Tf: Fisher matrix update frequency (default: 10)
        """
    
    def step(self):
        """
        Perform KFAC optimization step.
        
        Process:
        1. Add weight decay to gradients
        2. For each module:
           a. Compute natural gradient using Fisher matrices
           b. Apply Kronecker factorization
           c. Handle device mismatches
        3. Compute KL divergence and clip step size
        4. Apply updates to parameters
        """
```

**Critical Implementation Details**:
- Fisher matrices (m_aa, m_gg) accumulate statistics across all processes
- Eigenvalue decomposition is performed every Tf steps
- Eigenvalues below 1e-6 are thresholded to zero for stability
- Device mismatches are automatically handled by moving tensors
- Natural gradient is computed as: v = Q_g @ (Q_g^T @ grad @ Q_a) / (d_g ⊗ d_a + λ) @ Q_a^T

### 3. ACKTR Agent (acktr_pipeline.py)

**Purpose**: Implements the Actor-Critic using Kronecker-Factored Trust Region algorithm.

**Key Functions**:

```python
class ACKTR:
    def update(self, rollouts):
        """
        Update actor-critic model using rollouts from all processes.
        
        Args:
            rollouts: RolloutStorage containing transitions from all processes
            
        Returns:
            tuple: (value_loss, action_loss, dist_entropy, prob_loss, graph_loss)
            
        Process:
        1. Reshape observations and actions from (num_steps, num_processes, ...) 
           to (num_steps * num_processes, ...)
        2. Evaluate actions to get values, log_probs, entropy, and predicted masks
        3. Compute advantages: A = returns - values
        4. Compute losses:
           - value_loss: MSE between returns and values
           - action_loss: -advantages * log_probs (policy gradient)
           - prob_loss: invalid action probability penalty
           - graph_loss: MSE between predicted and true action masks
           - entropy_loss: negative entropy for exploration
        5. If KFAC and at update step:
           - Compute Fisher loss for KFAC statistics
           - Accumulate Fisher information
        6. Compute total loss and backpropagate
        7. Apply KFAC or gradient clipping
        8. Update parameters
        """
```

### 4. Rollout Storage (storage.py)

**Purpose**: Stores transitions collected from all parallel environments.

**Key Attributes**:
```python
class RolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, can_give_up=False,
                 enable_rotation=False, pallet_size=10):
        """
        Initialize rollout storage.
        
        Storage dimensions:
        - obs: (num_steps + 1, num_processes, *obs_shape)
        - actions: (num_steps, num_processes, action_dim)
        - rewards: (num_steps, num_processes, 1)
        - value_preds: (num_steps, num_processes, 1)
        - returns: (num_steps + 1, num_processes, 1)
        - action_log_probs: (num_steps, num_processes, 1)
        - masks: (num_steps + 1, num_processes, 1)
        - location_masks: (num_steps + 1, num_processes, mask_size)
        """
```

**Key Functions**:
```python
def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
           value_preds, rewards, masks, bad_masks, location_masks):
    """Insert new transition into storage."""

def compute_returns(self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits):
    """
    Compute returns using Generalized Advantage Estimation.
    
    For each process independently:
    - Apply discount factor gamma
    - Handle episode boundaries using masks
    - Compute returns: R_t = r_t + gamma * R_{t+1} * mask_{t+1}
    """

def after_update(self):
    """Copy last step to first position for next rollout."""
```

### 5. Training Loop (main.py)

**Purpose**: Orchestrates the training process.

**Key Modifications for Multi-Process**:

```python
def train_model(args):
    # Set num_processes from args (default: 16)
    num_processes = args.num_processes
    
    # Create vectorized environments
    envs = make_vec_envs(env_name, args.seed, num_processes, args.gamma,
                         log_dir, device, False, args=args)
    
    # Initialize rollout storage with num_processes
    rollouts = RolloutStorage(args.num_steps, num_processes,
                              envs.observation_space.shape,
                              envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              can_give_up=False,
                              enable_rotation=args.enable_rotation,
                              pallet_size=args.container_size[0])
    
    # Training loop
    while True:
        # Collect rollouts from all processes
        for step in range(args.num_steps):
            # Sample actions for all processes
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = \
                    actor_critic.act(rollouts.obs[step],
                                    rollouts.recurrent_hidden_states[step],
                                    rollouts.masks[step],
                                    location_masks)
            
            # Step all environments in parallel
            obs, reward, done, infos = envs.step(action)
            
            # Aggregate episode statistics from all processes
            for i in range(num_processes):
                if 'episode' in infos[i].keys():
                    episode_rewards.append(infos[i]['episode']['r'])
                    episode_ratio.append(infos[i]['ratio'])
            
            # Insert transitions into rollout storage
            rollouts.insert(obs, recurrent_hidden_states, action,
                           action_log_prob, value, reward, masks,
                           bad_masks, location_masks)
        
        # Compute returns
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        
        rollouts.compute_returns(next_value, False, args.gamma, 0.95, False)
        
        # Update model using all collected transitions
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = \
            agent.update(rollouts)
        
        # Clear rollout storage for next iteration
        rollouts.after_update()
        
        # Log aggregated statistics
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            print(f"Mean reward: {np.mean(episode_rewards):.1f}")
            print(f"Mean space ratio: {np.mean(episode_ratio):.3f}")
```

## Data Models

### Observation Space

```python
observation_shape = (channels, height, width)
# channels = 4: [height_map, next_box_x, next_box_y, next_box_z]
# height = width = container_size[0] (default: 10)
```

### Action Space

```python
if enable_rotation:
    action_space = Discrete(container_size[0] * container_size[1] * 2)
    # First half: no rotation, Second half: 90° rotation
else:
    action_space = Discrete(container_size[0] * container_size[1])
    # Each action represents a (x, y) placement position
```

### Rollout Data Structure

```python
{
    'obs': Tensor(num_steps+1, num_processes, channels, height, width),
    'actions': Tensor(num_steps, num_processes, 1),
    'rewards': Tensor(num_steps, num_processes, 1),
    'value_preds': Tensor(num_steps, num_processes, 1),
    'returns': Tensor(num_steps+1, num_processes, 1),
    'action_log_probs': Tensor(num_steps, num_processes, 1),
    'masks': Tensor(num_steps+1, num_processes, 1),  # 0 if done, 1 otherwise
    'bad_masks': Tensor(num_steps+1, num_processes, 1),  # 0 if bad transition
    'location_masks': Tensor(num_steps+1, num_processes, mask_size),
    'recurrent_hidden_states': Tensor(num_steps+1, num_processes, hidden_size)
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Rollout storage dimensions match process count

*For any* num_processes and num_steps configuration, the rollout storage tensors should have shape (num_steps, num_processes, ...) or (num_steps+1, num_processes, ...) for observations and masks.

**Validates: Requirements 1.3, 6.1**

### Property 2: Unique seeds across processes

*For any* num_processes value, each environment process should receive a unique seed equal to base_seed + process_rank, ensuring different random sequences.

**Validates: Requirements 4.1**

### Property 3: Gradient batch size equals num_steps × num_processes

*For any* rollout update, the batch size used for gradient computation should equal num_steps × num_processes, ensuring all collected transitions contribute to the update.

**Validates: Requirements 1.4**

### Property 4: Fisher matrix accumulation across processes

*For any* KFAC update step, the Fisher information matrices (m_aa, m_gg) should accumulate statistics from all num_processes environments, not just a single process.

**Validates: Requirements 2.1**

### Property 5: KFAC natural gradient computation

*For any* parameter update, the KFAC optimizer should compute the natural gradient using the formula: v = Q_g @ (Q_g^T @ grad @ Q_a) / (d_g ⊗ d_a + λ) @ Q_a^T, where Q_g, Q_a are eigenvectors and d_g, d_a are eigenvalues of Fisher matrices.

**Validates: Requirements 2.2**

### Property 6: Eigenvalue thresholding for stability

*For any* eigenvalue decomposition in KFAC, eigenvalues below 1e-6 should be set to zero to prevent numerical instability.

**Validates: Requirements 2.3, 8.4**

### Property 7: Device consistency in KFAC

*For any* KFAC update, if Q_a and Q_g are on different devices, the system should automatically move Q_a and d_a to match Q_g's device before matrix multiplication.

**Validates: Requirements 2.4**

### Property 8: No NaN or Inf in gradients

*For any* training step, all gradient tensors and parameter tensors should not contain NaN or Inf values after the update.

**Validates: Requirements 2.5**

### Property 9: Environment parameter serialization

*For any* environment configuration, the parameters (box_set, container_size, reliability settings) should be serializable via pickle for multiprocessing.

**Validates: Requirements 3.3**

### Property 10: Return computation with episode boundaries

*For any* sequence of rewards and done masks, returns should be computed as R_t = r_t + gamma * R_{t+1} * mask_{t+1}, where mask is 0 at episode boundaries and 1 otherwise.

**Validates: Requirements 4.4**

### Property 11: Episode reward aggregation

*For any* training iteration, episode rewards from all processes should be collected and aggregated for statistics computation.

**Validates: Requirements 5.1**

### Property 12: Statistics computation correctness

*For any* list of episode rewards, the computed mean, median, min, and max should match the standard statistical definitions.

**Validates: Requirements 5.2**

### Property 13: Reliability feature tracking across processes

*For any* reliability feature (uncertainty, visual feedback, motion primitive), usage counts should be aggregated across all processes.

**Validates: Requirements 5.5**

## Error Handling

### 1. Process Initialization Errors

**Error**: Environment creation fails in subprocess
**Handling**:
- Log error with process ID and stack trace
- Raise exception to terminate training
- Provide clear error message about environment configuration

### 2. Shared Memory Errors

**Error**: Shared memory allocation fails
**Handling**:
- Fall back to DummyVecEnv (single-process)
- Log warning about performance degradation
- Continue training with reduced parallelism

### 3. KFAC Numerical Errors

**Error**: Eigenvalue decomposition produces NaN or Inf
**Handling**:
- Apply eigenvalue thresholding (values < 1e-6 → 0)
- Add damping factor to Fisher matrices
- Log warning about numerical instability
- Continue training with regularized update

### 4. Device Mismatch Errors

**Error**: Tensors on different devices in KFAC
**Handling**:
- Automatically detect device mismatch
- Move tensors to correct device
- Log warning about device transfer
- Continue training

### 5. GPU Out of Memory

**Error**: CUDA out of memory during training
**Handling**:
- Catch CUDA OOM exception
- Suggest reducing num_processes or batch size
- Provide memory usage statistics
- Terminate training gracefully

### 6. Process Crash

**Error**: Subprocess crashes during training
**Handling**:
- Detect crashed process via exception
- Log error with process ID
- Terminate all processes
- Save emergency checkpoint if possible

## Testing Strategy

### Unit Tests

1. **Test Rollout Storage Dimensions**
   - Create rollout storage with various num_processes values
   - Verify tensor shapes match expected dimensions
   - Test insert and after_update operations

2. **Test Seed Assignment**
   - Create multiple environments with base seed
   - Verify each process gets unique seed
   - Test that different seeds produce different sequences

3. **Test KFAC Eigenvalue Thresholding**
   - Create Fisher matrices with small eigenvalues
   - Run eigenvalue decomposition
   - Verify eigenvalues < 1e-6 are zeroed

4. **Test Device Handling**
   - Create tensors on different devices
   - Run KFAC update
   - Verify tensors are moved to correct device

5. **Test Return Computation**
   - Create reward sequences with episode boundaries
   - Compute returns with different gamma values
   - Verify returns match expected values

6. **Test Statistics Computation**
   - Create list of episode rewards
   - Compute mean, median, min, max
   - Verify against numpy implementations

### Property-Based Tests

Property-based tests will use the Hypothesis library for Python to generate random test cases and verify universal properties.

1. **Property Test: Rollout Storage Dimensions**
   - Generate random num_processes (1-32) and num_steps (1-20)
   - Create rollout storage
   - Verify all tensor dimensions are correct

2. **Property Test: Unique Seeds**
   - Generate random base_seed and num_processes
   - Create environments
   - Verify all seeds are unique

3. **Property Test: Gradient Batch Size**
   - Generate random num_processes and num_steps
   - Create rollouts and compute gradients
   - Verify batch size equals num_processes × num_steps

4. **Property Test: No NaN in Gradients**
   - Generate random model parameters and gradients
   - Run KFAC update
   - Verify no NaN or Inf in results

5. **Property Test: Return Computation**
   - Generate random reward sequences and masks
   - Compute returns
   - Verify returns satisfy recursive formula

6. **Property Test: Statistics Correctness**
   - Generate random reward lists
   - Compute statistics
   - Verify against numpy implementations

### Integration Tests

1. **Test Multi-Process Training Loop**
   - Run training for 100 updates with 4 processes
   - Verify no crashes or errors
   - Check that rewards improve over time

2. **Test Single vs Multi-Process Consistency**
   - Train with 1 process for N steps
   - Train with 4 processes for N/4 steps
   - Verify similar learning progress

3. **Test Checkpoint Saving and Loading**
   - Train for 50 updates
   - Save checkpoint
   - Load checkpoint and continue training
   - Verify training continues correctly

4. **Test Reliability Features with Multi-Process**
   - Enable uncertainty simulation
   - Train with 4 processes
   - Verify feature tracking works across processes

### Performance Tests

1. **Test Scaling with Process Count**
   - Measure training speed with 1, 4, 8, 16 processes
   - Verify approximately linear speedup
   - Check memory usage scales reasonably

2. **Test Memory Stability**
   - Run training for 1000 updates
   - Monitor memory usage
   - Verify no memory leaks

## Implementation Notes

### Platform-Specific Considerations

**Windows**:
- Must use 'spawn' multiprocessing context
- Requires all environment parameters to be picklable
- Shared memory implementation differs from Linux
- May have slower process creation

**Linux**:
- Can use 'fork' multiprocessing context (faster)
- Shared memory is more efficient
- Better support for large numbers of processes

### Hyperparameter Configuration (from Paper)

```python
# Training
num_processes = 16  # Number of parallel environments
num_steps = 5  # Forward steps per update
gamma = 1.0  # Discount factor (no discounting for episodic task)

# Loss coefficients
value_loss_coef = 0.5
entropy_coef = 0.01
invalid_coef = 2.0  # Penalty for invalid actions

# KFAC optimizer
lr = 0.25  # Learning rate
momentum = 0.9
stat_decay = 0.99  # Running statistics decay
kl_clip = 0.001  # KL divergence clipping
damping = 1e-2  # Damping for numerical stability
Ts = 1  # Statistics update frequency
Tf = 10  # Fisher matrix update frequency

# Model
hidden_size = 256  # Hidden layer size

# Training schedule
save_interval = 10  # Save every 10 updates
log_interval = 10  # Log every 10 updates
```

### Expected Performance

Based on the paper's results:
- **CUT-2 dataset**: ~70% space utilization after training
- **Random sequence**: ~73% space utilization after training
- **Training time**: ~1 day on GPU with 16 processes
- **Convergence**: Stable learning curve without divergence

### Debugging Tips

1. **Check process creation**: Verify all 16 processes are created
2. **Monitor memory**: Watch for memory leaks in long runs
3. **Check Fisher matrices**: Verify they're being updated
4. **Inspect gradients**: Check for NaN or Inf values
5. **Compare with single-process**: Verify multi-process gives similar results
6. **Profile performance**: Identify bottlenecks in the pipeline

### Known Issues and Workarounds

1. **Windows spawn context is slow**
   - Workaround: Use fewer processes on Windows (e.g., 8 instead of 16)
   
2. **KFAC can be numerically unstable**
   - Workaround: Eigenvalue thresholding and damping are already implemented
   
3. **GPU memory can be exhausted**
   - Workaround: Reduce num_processes or use CPU for some processes
   
4. **Shared memory cleanup on crash**
   - Workaround: Manually clean up shared memory segments if needed
