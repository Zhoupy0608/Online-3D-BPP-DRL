#!/usr/bin/env python3
"""
Example Testing Script with Reliability Features

This script demonstrates how to test a trained DRL agent with reliability features enabled:
- Visual feedback module for height map correction
- Parallel entry motion primitive with buffer space
- Multi-camera point cloud fusion (optional)

Usage:
    # Test with single camera visual feedback and parallel motion
    python example_test_with_reliability.py
    
    # Test with multi-camera setup
    python example_test_with_reliability.py --multi-camera

Features enabled:
    - Visual feedback module
    - Parallel entry motion with buffer space
    - Performance comparison with baseline

Requirements: 1, 2, 9, 10 (Visual feedback, parallel motion, multi-camera, performance evaluation)
"""

import subprocess
import sys
import argparse

def parse_args():
    """Parse command-line arguments for the test script."""
    parser = argparse.ArgumentParser(
        description='Test trained model with reliability features'
    )
    parser.add_argument(
        '--model-name',
        default='default_cut_2.pt',
        help='Name of the trained model to test (default: default_cut_2.pt)'
    )
    parser.add_argument(
        '--data-name',
        default='cut_2.pt',
        help='Name of the test dataset (default: cut_2.pt)'
    )
    parser.add_argument(
        '--cases',
        type=int,
        default=100,
        help='Number of test cases to run (default: 100)'
    )
    parser.add_argument(
        '--multi-camera',
        action='store_true',
        help='Enable multi-camera setup for visual feedback'
    )
    parser.add_argument(
        '--buffer-range-x',
        type=int,
        default=1,
        help='Buffer space range in x direction (default: 1)'
    )
    parser.add_argument(
        '--buffer-range-y',
        type=int,
        default=1,
        help='Buffer space range in y direction (default: 1)'
    )
    parser.add_argument(
        '--enable-rotation',
        action='store_true',
        help='Enable box rotation during packing'
    )
    parser.add_argument(
        '--no-visual-feedback',
        action='store_true',
        help='Disable visual feedback (for baseline comparison)'
    )
    parser.add_argument(
        '--no-parallel-motion',
        action='store_true',
        help='Disable parallel motion (for baseline comparison)'
    )
    return parser.parse_args()

def main():
    """
    Test a trained DRL agent with reliability features enabled.
    
    This example demonstrates testing with:
    1. Visual feedback module for height map correction
    2. Parallel entry motion primitive with buffer space
    3. Optional multi-camera point cloud fusion
    """
    
    args = parse_args()
    
    print("=" * 70)
    print("Testing DRL Agent with Reliability Features")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.data_name}")
    print(f"  - Test cases: {args.cases}")
    print(f"  - Visual feedback: {'Disabled' if args.no_visual_feedback else 'Enabled'}")
    print(f"  - Parallel motion: {'Disabled' if args.no_parallel_motion else 'Enabled'}")
    
    if not args.no_parallel_motion:
        print(f"  - Buffer range: ({args.buffer_range_x}, {args.buffer_range_y})")
    
    if not args.no_visual_feedback:
        camera_config = "camera_config_example.json"
        if args.multi_camera:
            print(f"  - Camera setup: Multi-camera (3 cameras)")
            print(f"  - Camera config: {camera_config} (multi_camera section)")
        else:
            print(f"  - Camera setup: Single camera")
            print(f"  - Camera config: {camera_config} (single_camera section)")
    
    print(f"  - Rotation: {'Enabled' if args.enable_rotation else 'Disabled'}")
    print()
    
    if not args.no_visual_feedback or not args.no_parallel_motion:
        print("Reliability features enabled:")
        if not args.no_visual_feedback:
            print("  1. Visual Feedback Module:")
            print("     - Captures point cloud data after each placement")
            print("     - Extracts actual box positions using region growing")
            print("     - Updates height map based on measured positions")
        if not args.no_parallel_motion:
            print("  2. Parallel Entry Motion:")
            print("     - Generates motion options with buffer space")
            print("     - Selects best option based on height map")
            print("     - Mitigates placement uncertainty")
        print()
    
    print("=" * 70)
    print()
    
    # Build command with all necessary arguments
    cmd = [
        sys.executable,  # Use current Python interpreter
        "main.py",
        
        # Test mode
        "--mode", "test",
        "--load-model",
        "--load-name", args.model_name,
        "--data-name", args.data_name,
        
        # Test configuration
        "--cases", str(args.cases),
        "--container-size", "10", "10", "10",
        "--item-size-range", "2", "2", "2", "5", "5", "5",
        
        # Directories
        "--load-dir", "./pretrained_models/",
        
        # Random seed for reproducibility
        "--seed", "42",
    ]
    
    # Add reliability features if not disabled
    if not args.no_visual_feedback:
        cmd.append("--visual-feedback-enabled")
        camera_config = "camera_config_example.json"
        cmd.extend(["--camera-config", camera_config])
    
    if not args.no_parallel_motion:
        cmd.append("--parallel-motion-enabled")
        cmd.extend(["--buffer-range-x", str(args.buffer_range_x)])
        cmd.extend(["--buffer-range-y", str(args.buffer_range_y)])
    
    # Add rotation if enabled
    if args.enable_rotation:
        cmd.append("--enable-rotation")
    
    # Optional: Add CUDA support if available
    try:
        import torch
        if torch.cuda.is_available():
            cmd.extend(["--use-cuda", "--device", "0"])
            print("CUDA detected - GPU testing enabled")
            print()
    except ImportError:
        print("PyTorch not found - using CPU testing")
        print()
    
    print("Starting testing...")
    print("Command:", " ".join(cmd))
    print()
    
    # Run the test
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTesting failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(0)
    
    print()
    print("=" * 70)
    print("Testing completed!")
    print("=" * 70)
    print()
    print("Performance metrics have been calculated and displayed above.")
    print()
    print("To compare with baseline (no reliability features), run:")
    print(f"  python example_test_with_reliability.py --no-visual-feedback --no-parallel-motion")
    print()


def run_comparison():
    """
    Run a comparison between baseline and reliability-enhanced testing.
    
    This function runs the test twice:
    1. With reliability features disabled (baseline)
    2. With reliability features enabled
    
    Then compares the performance metrics.
    """
    print("=" * 70)
    print("Running Performance Comparison")
    print("=" * 70)
    print()
    print("This will run two tests:")
    print("  1. Baseline (no reliability features)")
    print("  2. With reliability features (visual feedback + parallel motion)")
    print()
    
    # Test 1: Baseline
    print("Running baseline test...")
    baseline_cmd = [
        sys.executable,
        __file__,
        "--no-visual-feedback",
        "--no-parallel-motion",
        "--cases", "50"  # Fewer cases for faster comparison
    ]
    subprocess.run(baseline_cmd)
    
    print()
    print("-" * 70)
    print()
    
    # Test 2: With reliability features
    print("Running test with reliability features...")
    reliability_cmd = [
        sys.executable,
        __file__,
        "--cases", "50"
    ]
    subprocess.run(reliability_cmd)
    
    print()
    print("=" * 70)
    print("Comparison completed!")
    print("=" * 70)
    print()
    print("Compare the space utilization metrics from both runs above.")
    print()


if __name__ == "__main__":
    # Check if running in comparison mode
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        run_comparison()
    else:
        main()
