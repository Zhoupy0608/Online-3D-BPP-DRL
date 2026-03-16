# Environment Selection Guide

## Overview

The system supports two packing environments:

1. **Bpp-v0**: Original PackingGame environment
2. **BppReliable-v0**: Enhanced ReliablePackingGame with reliability features

## Automatic Environment Selection

The system automatically selects the appropriate environment based on command-line flags:

### Default Behavior (Backward Compatible)
When no reliability features are enabled, the system uses the original `Bpp-v0` environment:

```bash
python main.py --mode train
# Uses: Bpp-v0
```

### With Reliability Features
When any reliability feature is enabled, the system automatically switches to `BppReliable-v0`:

```bash
# Training with uncertainty simulation
python main.py --mode train --uncertainty-enabled
# Uses: BppReliable-v0

# Testing with visual feedback
python main.py --mode test --load-model --visual-feedback-enabled
# Uses: BppReliable-v0

# Training with parallel motion
python main.py --mode train --parallel-motion-enabled
# Uses: BppReliable-v0

# All features enabled
python main.py --mode train --uncertainty-enabled --visual-feedback-enabled --parallel-motion-enabled
# Uses: BppReliable-v0
```

## Environment Registration

Both environments are registered in `main.py` through the `registration_envs()` function:

```python
def registration_envs():
    """Register all available packing environments."""
    # Original environment
    register(
        id='Bpp-v0',
        entry_point='envs.bpp0:PackingGame',
    )
    
    # Enhanced environment with reliability features
    register(
        id='BppReliable-v0',
        entry_point='envs.bpp0:ReliablePackingGame',
    )
```

## Environment Selection Logic

The `select_environment()` function determines which environment to use:

```python
def select_environment(args):
    """
    Select environment based on configuration flags.
    
    Returns:
        'BppReliable-v0' if any reliability feature is enabled
        'Bpp-v0' otherwise (backward compatible)
    """
    if (args.uncertainty_enabled or 
        args.visual_feedback_enabled or 
        args.parallel_motion_enabled):
        return 'BppReliable-v0'
    return 'Bpp-v0'
```

## Reliability Feature Flags

### Uncertainty Simulation
- `--uncertainty-enabled`: Enable placement noise during training
- `--uncertainty-std-x`: Noise standard deviation in x direction (default: 0.5)
- `--uncertainty-std-y`: Noise standard deviation in y direction (default: 0.5)
- `--uncertainty-std-z`: Noise standard deviation in z direction (default: 0.1)

### Visual Feedback
- `--visual-feedback-enabled`: Enable visual feedback module
- `--camera-config`: Path to camera configuration file

### Parallel Entry Motion
- `--parallel-motion-enabled`: Enable parallel entry motion primitive
- `--buffer-range-x`: Buffer space in x direction (default: 1)
- `--buffer-range-y`: Buffer space in y direction (default: 1)

## Examples

### Training with Uncertainty (Recommended)
```bash
python main.py --mode train \
    --uncertainty-enabled \
    --uncertainty-std-x 0.5 \
    --uncertainty-std-y 0.5 \
    --uncertainty-std-z 0.1 \
    --save-model
```

### Testing with Visual Feedback and Parallel Motion
```bash
python main.py --mode test \
    --load-model \
    --load-name my_model.pt \
    --visual-feedback-enabled \
    --parallel-motion-enabled \
    --buffer-range-x 2 \
    --buffer-range-y 2
```

### Backward Compatible Training (No Reliability Features)
```bash
python main.py --mode train --save-model
# Automatically uses Bpp-v0
```

## Logging

When reliability features are enabled, the system logs which features are active:

```
Using environment: BppReliable-v0
Reliability features enabled:
  - Uncertainty simulation (std: (0.5, 0.5, 0.1))
  - Visual feedback module
  - Parallel entry motion (buffer: (1, 1))
```

## Implementation Details

### Training Flow
1. `main()` calls `train_model(args)`
2. `train_model()` calls `select_environment(args)`
3. Environment is registered via `registration_envs()`
4. Selected environment is used for training

### Testing Flow
1. `main()` calls `test_model(args)`
2. `test_model()` calls `select_environment(args)`
3. Environment is registered via `registration_envs()`
4. Selected environment is used for testing

### Backward Compatibility
- Existing scripts work without modification
- Default behavior uses original `Bpp-v0` environment
- No breaking changes to existing functionality
- Reliability features are opt-in through flags

## Troubleshooting

### Environment Not Found
If you see "Environment not found" errors:
1. Ensure `registration_envs()` is called before `main()`
2. Check that both environments are properly imported in `envs/bpp0/__init__.py`

### Wrong Environment Selected
If the wrong environment is being used:
1. Check that reliability feature flags are set correctly
2. Verify `select_environment()` logic in main.py
3. Look for environment selection logs in console output

### Import Errors
If you see import errors for ReliablePackingGame:
1. Ensure `envs/bpp0/bin3D_reliable.py` exists
2. Check that it's imported in `envs/bpp0/__init__.py`
3. Verify all dependencies (uncertainty, motion_primitive, visual_feedback) are available
