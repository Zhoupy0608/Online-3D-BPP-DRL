# Example Scripts for Reliability Features

This directory contains example scripts demonstrating how to use the reliability features for robust robot packing. These scripts show the complete workflow from training with uncertainty simulation to testing with visual feedback and parallel motion.

## Overview

The reliability features enhance the DRL-based bin packing system with:

1. **Uncertainty Simulation** - Adds placement noise during training for robustness
2. **Visual Feedback Module** - Uses 3D cameras to correct height map after placement
3. **Parallel Entry Motion** - Generates motion options with buffer space
4. **Multi-Camera Support** - Fuses point clouds from multiple cameras

## Example Scripts

### 1. `example_train_with_uncertainty.py`

Trains a DRL agent with uncertainty simulation enabled.

**Features:**
- Adds Gaussian noise to placement positions during training
- Makes the learned policy robust to real-world placement errors
- Saves model with uncertainty configuration

**Usage:**
```bash
python example_train_with_uncertainty.py
```

**Configuration:**
- Dataset: CUT-2
- Uncertainty std: (0.5, 0.5, 0.1) for (x, y, z)
- Algorithm: ACKTR
- Processes: 16

**Requirements Validated:** Requirement 8 (Training with simulated uncertainty)

---

### 2. `example_test_with_reliability.py`

Tests a trained model with reliability features enabled.

**Features:**
- Visual feedback module for height map correction
- Parallel entry motion with buffer space
- Multi-camera support (optional)
- Performance comparison with baseline

**Usage:**
```bash
# Test with single camera
python example_test_with_reliability.py

# Test with multi-camera setup
python example_test_with_reliability.py --multi-camera

# Test with custom model
python example_test_with_reliability.py --model-name my_model.pt

# Baseline test (no reliability features)
python example_test_with_reliability.py --no-visual-feedback --no-parallel-motion

# Run comparison
python example_test_with_reliability.py --compare
```

**Options:**
- `--model-name`: Name of trained model (default: default_cut_2.pt)
- `--data-name`: Test dataset name (default: cut_2.pt)
- `--cases`: Number of test cases (default: 100)
- `--multi-camera`: Enable multi-camera setup
- `--buffer-range-x/y`: Buffer space range (default: 1)
- `--enable-rotation`: Enable box rotation
- `--no-visual-feedback`: Disable visual feedback
- `--no-parallel-motion`: Disable parallel motion
- `--compare`: Run baseline vs reliability comparison

**Requirements Validated:** Requirements 1, 2, 9, 10 (Visual feedback, parallel motion, multi-camera, performance evaluation)

---

### 3. `example_full_pipeline.py`

Complete pipeline demonstrating all reliability features.

**Features:**
- Phase 1: Train with uncertainty simulation
- Phase 2: Test with reliability features
- Phase 3: Baseline test for comparison
- Automatic performance comparison

**Usage:**
```bash
# Run complete pipeline
python example_full_pipeline.py

# Quick mode (fewer iterations for testing)
python example_full_pipeline.py --quick

# Train only
python example_full_pipeline.py --train-only

# Test only (requires existing model)
python example_full_pipeline.py --test-only --model-name my_model.pt
```

**Options:**
- `--train-only`: Only run training phase
- `--test-only`: Only run testing phase
- `--quick`: Quick mode with fewer iterations
- `--model-name`: Model name for testing

**Requirements Validated:** All requirements (Complete pipeline)

---

## Camera Configuration

### `camera_config_example.json`

Example camera configuration file for visual feedback module.

**Configurations Included:**

1. **Single Camera Setup**
   - One camera looking down at container
   - Position: (0, 0, 1.5m)
   - Resolution: 640x480
   - Depth range: 0.5-5.0m

2. **Multi-Camera Setup**
   - Three cameras at different angles
   - Front camera: (0, -0.5, 1.5m)
   - Left camera: (-0.7, 0, 1.2m) at 45°
   - Right camera: (0.7, 0, 1.2m) at 45°
   - Provides better coverage and reduces occlusions

**Usage:**
```bash
# Specify camera config in test script
python example_test_with_reliability.py --camera-config camera_config_example.json
```

**Customization:**
Edit the JSON file to match your camera setup:
- Update intrinsic parameters (fx, fy, cx, cy)
- Update extrinsic parameters (position, rotation)
- Adjust resolution and depth range
- Add or remove cameras as needed

---

## Quick Start Guide

### Step 1: Train with Uncertainty

```bash
python example_train_with_uncertainty.py
```

This will:
1. Prompt for a test name (e.g., "my_test")
2. Train the agent with uncertainty simulation
3. Save the model in `./saved_models/my_test/`

### Step 2: Test with Reliability Features

```bash
# Update the model name to match your trained model
python example_test_with_reliability.py --model-name BppReliable-v02024.11.27-14-30.pt
```

This will:
1. Load your trained model
2. Test with visual feedback and parallel motion
3. Display performance metrics

### Step 3: Compare with Baseline

```bash
python example_test_with_reliability.py --compare
```

This will:
1. Run baseline test (no reliability features)
2. Run test with reliability features
3. Display comparison of performance metrics

---

## Complete Pipeline Example

For a fully automated workflow:

```bash
# Quick test of the pipeline
python example_full_pipeline.py --quick

# Full pipeline (takes longer)
python example_full_pipeline.py
```

This runs all three phases automatically and provides a summary.

---

## Configuration Parameters

### Uncertainty Simulation

- `--uncertainty-enabled`: Enable placement noise
- `--uncertainty-std-x`: Noise std in x direction (default: 0.5)
- `--uncertainty-std-y`: Noise std in y direction (default: 0.5)
- `--uncertainty-std-z`: Noise std in z direction (default: 0.1)

### Visual Feedback

- `--visual-feedback-enabled`: Enable visual feedback module
- `--camera-config`: Path to camera configuration file

### Parallel Motion

- `--parallel-motion-enabled`: Enable parallel entry motion
- `--buffer-range-x`: Buffer space in x direction (default: 1)
- `--buffer-range-y`: Buffer space in y direction (default: 1)

### General

- `--container-size`: Container dimensions (default: 10 10 10)
- `--item-size-range`: Item size range (default: 2 2 2 5 5 5)
- `--enable-rotation`: Enable box rotation
- `--use-cuda`: Enable GPU acceleration
- `--device`: GPU device ID (default: 0)

---

## Performance Metrics

The test scripts report the following metrics:

1. **Space Utilization**: Ratio of packed volume to container volume
2. **Mean Utilization**: Average across all test cases
3. **Std Utilization**: Standard deviation of utilization
4. **Success Rate**: Percentage of successful packing episodes
5. **Execution Time**: Time per packing episode

Compare these metrics between baseline and reliability-enhanced testing to see the improvement.

---

## Troubleshooting

### Issue: "No trained model chosen"

**Solution:** Make sure to use `--load-model` flag when testing:
```bash
python main.py --mode test --load-model --load-name your_model.pt
```

### Issue: "Camera configuration file not found"

**Solution:** Ensure `camera_config_example.json` exists in the current directory, or specify the full path:
```bash
python example_test_with_reliability.py --camera-config /path/to/camera_config.json
```

### Issue: CUDA out of memory

**Solution:** Reduce the number of processes or use CPU:
```bash
# Use fewer processes
python example_train_with_uncertainty.py  # Edit script to reduce --num-processes

# Or use CPU
python example_train_with_uncertainty.py  # Remove --use-cuda flag
```

### Issue: Training takes too long

**Solution:** Use quick mode for testing:
```bash
python example_full_pipeline.py --quick
```

---

## Next Steps

After running the example scripts:

1. **Analyze Results**: Compare performance metrics between baseline and reliability-enhanced testing
2. **Tune Parameters**: Adjust uncertainty std, buffer range, etc. based on your robot's characteristics
3. **Calibrate Cameras**: Update camera_config_example.json with your actual camera parameters
4. **Deploy on Robot**: Use the trained model with reliability features on your physical robot

---

## Additional Resources

- **Main Documentation**: See `README.md` for overall system documentation
- **Reliability Features Guide**: See `RELIABILITY_FEATURES_GUIDE.md` for detailed feature descriptions
- **Multi-Camera Guide**: See `MULTI_CAMERA_USAGE_GUIDE.md` for camera setup instructions
- **Environment Selection**: See `ENVIRONMENT_SELECTION_GUIDE.md` for choosing the right environment

---

## Requirements Mapping

These example scripts validate the following requirements:

- **Requirement 1**: Visual Feedback System Integration
- **Requirement 2**: Parallel Entry Motion Primitive
- **Requirement 8**: Training with Simulated Uncertainty
- **Requirement 9**: Multi-Camera Point Cloud Fusion
- **Requirement 10**: Performance Evaluation and Metrics

All scripts are designed to be easily customizable for your specific use case.
