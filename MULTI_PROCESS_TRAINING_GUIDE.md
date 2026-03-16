# Multi-Process Training Guide

This guide explains how to use the `multi_process_train.py` script for training the Online 3D Bin Packing model with multiple parallel processes.

## Overview

The `multi_process_train.py` script provides a streamlined interface for multi-process training with:
- **Paper-specified defaults**: All hyperparameters match the original paper (16 processes, KFAC optimizer settings, etc.)
- **Automatic GPU detection**: Automatically detects and configures available GPUs
- **Progress monitoring**: Displays training progress with ETA estimation
- **Comprehensive error handling**: Graceful shutdown and emergency checkpoint saving
- **Reliability features**: Support for uncertainty simulation, visual feedback, and motion primitives

## Quick Start

### Basic Training (16 processes, paper defaults)

```bash
python multi_process_train.py
```

This will:
1. Use 16 parallel processes (as specified in the paper)
2. Automatically detect and use GPU if available
3. Use paper-specified hyperparameters
4. Save models every 10 updates
5. Enable TensorBoard logging

### Custom Process Count

```bash
# Train with 8 processes (faster startup, slower training)
python multi_process_train.py --num-processes 8

# Train with 4 processes (for limited resources)
python multi_process_train.py --num-processes 4
```

### Force CPU Training

```bash
python multi_process_train.py --device cpu
```

### With Reliability Features

```bash
# Enable uncertainty simulation
python multi_process_train.py --uncertainty-enabled

# Enable all reliability features
python multi_process_train.py \
    --uncertainty-enabled \
    --visual-feedback-enabled \
    --parallel-motion-enabled
```

## Command-Line Options

### Core Training Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--num-processes` | 16 | Number of parallel processes (paper default) |
| `--num-steps` | 5 | Forward steps per update (paper default) |
| `--device` | auto | Device: auto, cpu, cuda:0, cuda:1, etc. |

### Dataset Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--item-seq` | cut_2 | Item sequence: cut1, cut2, rs |
| `--container-size` | 10 10 10 | Container dimensions (x, y, z) |
| `--item-size-range` | 2 2 2 5 5 5 | Item size range (min_x, min_y, min_z, max_x, max_y, max_z) |
| `--enable-rotation` | False | Enable box rotation |

### Hyperparameters (Paper-Specified Defaults)

| Option | Default | Description |
|--------|---------|-------------|
| `--gamma` | 1.0 | Discount factor (paper default) |
| `--value-loss-coef` | 0.5 | Value loss coefficient (paper default) |
| `--entropy-coef` | 0.01 | Entropy coefficient (paper default) |
| `--invalid-coef` | 2.0 | Invalid action coefficient (paper default) |

### KFAC Optimizer Parameters (Paper-Specified Defaults)

| Option | Default | Description |
|--------|---------|-------------|
| `--stat-decay` | 0.99 | Statistics decay rate (paper default) |
| `--kl-clip` | 0.001 | KL divergence clipping (paper default) |
| `--damping` | 1e-2 | Damping factor (paper default) |

### Training Schedule

| Option | Default | Description |
|--------|---------|-------------|
| `--save-interval` | 10 | Save model every N updates |
| `--log-interval` | 10 | Log statistics every N updates |
| `--tensorboard` | True | Enable TensorBoard logging |
| `--no-tensorboard` | - | Disable TensorBoard logging |
| `--save-model` | True | Save model checkpoints |
| `--no-save-model` | - | Disable model saving |

### Checkpoint Loading

| Option | Default | Description |
|--------|---------|-------------|
| `--pretrain` | False | Load pretrained model |
| `--load-name` | default_cut_2.pt | Pretrained model filename |
| `--load-dir` | ./pretrained_models/ | Directory with pretrained models |
| `--save-dir` | ./saved_models/ | Directory to save models |

### Reliability Features

| Option | Default | Description |
|--------|---------|-------------|
| `--uncertainty-enabled` | False | Enable placement uncertainty |
| `--uncertainty-std-x` | 0.5 | Uncertainty std in x direction |
| `--uncertainty-std-y` | 0.5 | Uncertainty std in y direction |
| `--uncertainty-std-z` | 0.1 | Uncertainty std in z direction |
| `--visual-feedback-enabled` | False | Enable visual feedback module |
| `--parallel-motion-enabled` | False | Enable parallel entry motion |
| `--buffer-range-x` | 1 | Buffer range in x direction |
| `--buffer-range-y` | 1 | Buffer range in y direction |
| `--camera-config` | None | Path to camera config file |

### Other Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--seed` | 1 | Random seed |
| `--hidden-size` | 256 | Hidden layer size |

## Usage Examples

### Example 1: Basic Training

```bash
python multi_process_train.py
```

When prompted, enter a test name (e.g., "my_first_training").

### Example 2: Quick Test with Fewer Processes

```bash
python multi_process_train.py --num-processes 4 --save-interval 5
```

### Example 3: Training with Uncertainty

```bash
python multi_process_train.py \
    --uncertainty-enabled \
    --uncertainty-std-x 0.3 \
    --uncertainty-std-y 0.3 \
    --uncertainty-std-z 0.1
```

### Example 4: Continue from Checkpoint

```bash
python multi_process_train.py \
    --pretrain \
    --load-name my_model.pt \
    --load-dir ./saved_models/my_first_training/
```

### Example 5: CPU Training with Custom Dataset

```bash
python multi_process_train.py \
    --device cpu \
    --num-processes 4 \
    --item-seq rs \
    --container-size 8 8 8
```

## Training Output

The script displays comprehensive information before training:

```
================================================================================
MULTI-PROCESS TRAINING CONFIGURATION
================================================================================

Process Configuration:
  Number of processes: 16
  Forward steps per update: 5
  Batch size per update: 80

Device Configuration:
  Device: cuda:0
  Available GPUs:
  GPU 0: NVIDIA GeForce RTX 3090 (24.0 GB)

Dataset Configuration:
  Item sequence: cut_2
  Container size: (10, 10, 10)
  Item size range: (2, 2, 2, 5, 5, 5)
  Enable rotation: False

Hyperparameters (Paper-Specified):
  Learning rate: 0.25
  Discount factor (gamma): 1.0
  Value loss coefficient: 0.5
  Entropy coefficient: 0.01
  Invalid action coefficient: 2.0

KFAC Optimizer Parameters:
  Statistics decay: 0.99
  KL clipping: 0.001
  Damping: 0.01

Training Schedule:
  Save interval: every 10 updates
  Log interval: every 10 updates
  TensorBoard: enabled

Estimated Training Time:
  ~1.0 days (for ~10,000 updates)
  Note: Actual time depends on hardware and dataset

Expected Results (from paper):
  CUT-2 dataset: ~70% space utilization
  Random sequence: ~73% space utilization

================================================================================
```

## Monitoring Training

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir=./runs
```

Then open http://localhost:6006 in your browser.

### Log Files

Training logs are saved to:
- `./logs/training_YYYYMMDD_HHMMSS.log`

### Saved Models

Models are saved to:
- `./saved_models/<test_name>/`

### Emergency Checkpoints

If training crashes, emergency checkpoints are saved to:
- `./emergency_checkpoints/`

## Stopping Training

Press `Ctrl+C` to stop training gracefully. The script will:
1. Save an emergency checkpoint
2. Clean up processes
3. Close environments properly

You can resume training using the `--pretrain` option with the saved checkpoint.

## Expected Performance

Based on the paper's results:

| Dataset | Expected Space Utilization | Training Time (16 processes, GPU) |
|---------|---------------------------|-----------------------------------|
| CUT-2 | ~70% | ~1 day |
| Random Sequence | ~73% | ~1 day |

## Troubleshooting

### Out of Memory Error

If you encounter CUDA out of memory errors:

```bash
# Reduce number of processes
python multi_process_train.py --num-processes 8

# Or use CPU
python multi_process_train.py --device cpu
```

### Slow Training

If training is slower than expected:

1. Check GPU utilization: `nvidia-smi`
2. Verify you're using GPU: Check the device configuration output
3. Try reducing logging frequency: `--log-interval 50`

### Process Crashes

If processes crash during training:

1. Check the log files in `./logs/`
2. Look for emergency checkpoints in `./emergency_checkpoints/`
3. Try reducing process count: `--num-processes 8`

## Comparison with main.py

The `multi_process_train.py` script is a wrapper around `main.py` that provides:

- **Easier configuration**: Command-line options with sensible defaults
- **Better documentation**: Clear help messages and examples
- **Progress monitoring**: ETA estimation and progress display
- **GPU detection**: Automatic GPU configuration
- **Error handling**: Graceful shutdown and emergency checkpoints

You can still use `main.py` directly if you prefer, but `multi_process_train.py` is recommended for most use cases.

## Requirements

Ensure you have the following installed:

- Python 3.7+
- PyTorch 1.7+
- NumPy
- TensorboardX (for TensorBoard logging)
- All dependencies from `requirements.txt`

## Additional Resources

- **Design Document**: `.kiro/specs/multi-process-training/design.md`
- **Requirements**: `.kiro/specs/multi-process-training/requirements.md`
- **Task List**: `.kiro/specs/multi-process-training/tasks.md`
- **Main README**: `README.md`

## Support

For issues or questions:

1. Check the log files in `./logs/`
2. Review the design document for implementation details
3. Check emergency checkpoints if training crashed
4. Verify your environment meets the requirements
