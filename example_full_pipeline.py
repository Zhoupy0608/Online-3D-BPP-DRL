#!/usr/bin/env python3
"""
Complete Pipeline Example: Train and Test with All Reliability Features

This script demonstrates the complete workflow:
1. Train with uncertainty simulation
2. Test with visual feedback and parallel motion
3. Compare performance with baseline

Usage:
    # Run complete pipeline
    python example_full_pipeline.py
    
    # Run only training
    python example_full_pipeline.py --train-only
    
    # Run only testing
    python example_full_pipeline.py --test-only

Requirements: All (Complete reliability feature pipeline)
"""

import subprocess
import sys
import argparse
import os
import time

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Complete pipeline for training and testing with reliability features'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only run training phase'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run testing phase'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode with fewer iterations (for testing the pipeline)'
    )
    parser.add_argument(
        '--model-name',
        default=None,
        help='Model name for testing (auto-generated if training)'
    )
    return parser.parse_args()


def print_header(title):
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(title.center(70))
    print("=" * 70)
    print()


def train_with_uncertainty(quick_mode=False):
    """
    Phase 1: Train with uncertainty simulation.
    
    Args:
        quick_mode: If True, use fewer iterations for quick testing
    """
    print_header("PHASE 1: Training with Uncertainty Simulation")
    
    print("This phase will:")
    print("  1. Initialize the DRL agent")
    print("  2. Add Gaussian noise to placements during training")
    print("  3. Train the agent to be robust to placement errors")
    print("  4. Save the trained model with uncertainty configuration")
    print()
    
    if quick_mode:
        print("QUICK MODE: Using reduced training iterations")
        print()
    
    # Training configuration
    cmd = [
        sys.executable,
        "main.py",
        "--mode", "train",
        "--item-seq", "cut2",
        "--container-size", "10", "10", "10",
        "--item-size-range", "2", "2", "2", "5", "5", "5",
        "--algorithm", "acktr",
        "--uncertainty-enabled",
        "--uncertainty-std-x", "0.5",
        "--uncertainty-std-y", "0.5",
        "--uncertainty-std-z", "0.1",
        "--save-model",
        "--tensorboard",
        "--seed", "42",
    ]
    
    # Adjust parameters for quick mode
    if quick_mode:
        cmd.extend([
            "--num-processes", "4",
            "--save-interval", "5",
            "--log-interval", "5",
        ])
    else:
        cmd.extend([
            "--num-processes", "16",
            "--save-interval", "10",
            "--log-interval", "10",
        ])
    
    # Add CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            cmd.extend(["--use-cuda", "--device", "0"])
            print("GPU training enabled")
    except ImportError:
        print("CPU training enabled")
    
    print()
    print("Starting training...")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
        return False


def test_with_reliability(model_name, quick_mode=False):
    """
    Phase 2: Test with reliability features.
    
    Args:
        model_name: Name of the trained model to test
        quick_mode: If True, use fewer test cases
    """
    print_header("PHASE 2: Testing with Reliability Features")
    
    print("This phase will:")
    print("  1. Load the trained model")
    print("  2. Enable visual feedback module")
    print("  3. Enable parallel entry motion")
    print("  4. Test on the dataset and measure performance")
    print()
    
    if quick_mode:
        print("QUICK MODE: Using fewer test cases")
        print()
    
    cases = 20 if quick_mode else 100
    
    # Testing configuration
    cmd = [
        sys.executable,
        "main.py",
        "--mode", "test",
        "--load-model",
        "--load-name", model_name,
        "--data-name", "cut_2.pt",
        "--cases", str(cases),
        "--container-size", "10", "10", "10",
        "--item-size-range", "2", "2", "2", "5", "5", "5",
        "--visual-feedback-enabled",
        "--parallel-motion-enabled",
        "--buffer-range-x", "1",
        "--buffer-range-y", "1",
        "--camera-config", "camera_config_example.json",
        "--seed", "42",
    ]
    
    # Add CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            cmd.extend(["--use-cuda", "--device", "0"])
    except ImportError:
        pass
    
    print("Starting testing with reliability features...")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("✓ Testing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Testing failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n✗ Testing interrupted by user")
        return False


def test_baseline(model_name, quick_mode=False):
    """
    Phase 3: Test baseline (no reliability features) for comparison.
    
    Args:
        model_name: Name of the trained model to test
        quick_mode: If True, use fewer test cases
    """
    print_header("PHASE 3: Baseline Testing (No Reliability Features)")
    
    print("This phase will:")
    print("  1. Load the same trained model")
    print("  2. Test WITHOUT reliability features")
    print("  3. Provide baseline performance for comparison")
    print()
    
    if quick_mode:
        print("QUICK MODE: Using fewer test cases")
        print()
    
    cases = 20 if quick_mode else 100
    
    # Baseline testing configuration (no reliability features)
    cmd = [
        sys.executable,
        "main.py",
        "--mode", "test",
        "--load-model",
        "--load-name", model_name,
        "--data-name", "cut_2.pt",
        "--cases", str(cases),
        "--container-size", "10", "10", "10",
        "--item-size-range", "2", "2", "2", "5", "5", "5",
        "--seed", "42",
    ]
    
    # Add CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            cmd.extend(["--use-cuda", "--device", "0"])
    except ImportError:
        pass
    
    print("Starting baseline testing...")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("✓ Baseline testing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Baseline testing failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n✗ Baseline testing interrupted by user")
        return False


def main():
    """Run the complete pipeline."""
    args = parse_args()
    
    print_header("Complete Reliability Features Pipeline")
    
    print("This pipeline demonstrates:")
    print("  • Training with uncertainty simulation (Requirement 8)")
    print("  • Visual feedback module (Requirement 1)")
    print("  • Parallel entry motion (Requirement 2)")
    print("  • Multi-camera support (Requirement 9)")
    print("  • Performance evaluation (Requirement 10)")
    print()
    
    if args.quick:
        print("⚡ QUICK MODE ENABLED")
        print("   Using reduced iterations for faster execution")
        print()
    
    # Determine model name
    if args.test_only:
        if args.model_name is None:
            print("Error: --model-name required when using --test-only")
            sys.exit(1)
        model_name = args.model_name
    else:
        # Generate model name based on timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        model_name = f"reliable_model_{timestamp}.pt"
    
    # Track success of each phase
    results = {}
    
    # Phase 1: Training (unless test-only)
    if not args.test_only:
        success = train_with_uncertainty(quick_mode=args.quick)
        results['training'] = success
        
        if not success:
            print()
            print("Pipeline stopped due to training failure.")
            sys.exit(1)
        
        if args.train_only:
            print()
            print_header("Pipeline Complete (Training Only)")
            print(f"Model saved. Use --test-only --model-name {model_name} to test.")
            return
    
    # Phase 2: Testing with reliability features (unless train-only)
    if not args.train_only:
        time.sleep(2)  # Brief pause between phases
        success = test_with_reliability(model_name, quick_mode=args.quick)
        results['reliability_test'] = success
        
        if not success:
            print()
            print("Pipeline stopped due to testing failure.")
            sys.exit(1)
    
    # Phase 3: Baseline testing for comparison (unless train-only)
    if not args.train_only:
        time.sleep(2)  # Brief pause between phases
        success = test_baseline(model_name, quick_mode=args.quick)
        results['baseline_test'] = success
    
    # Summary
    print()
    print_header("Pipeline Summary")
    
    if not args.test_only:
        status = "✓" if results.get('training', False) else "✗"
        print(f"{status} Phase 1: Training with uncertainty simulation")
    
    if not args.train_only:
        status = "✓" if results.get('reliability_test', False) else "✗"
        print(f"{status} Phase 2: Testing with reliability features")
        
        status = "✓" if results.get('baseline_test', False) else "✗"
        print(f"{status} Phase 3: Baseline testing")
    
    print()
    print("Compare the performance metrics from Phase 2 and Phase 3 above")
    print("to see the improvement from reliability features.")
    print()
    
    # Check if all phases succeeded
    if all(results.values()):
        print("✓ All phases completed successfully!")
        print()
        print("Next steps:")
        print("  1. Review the performance comparison")
        print("  2. Adjust parameters in the example scripts as needed")
        print("  3. Deploy on physical robot with real cameras")
        print()
    else:
        print("✗ Some phases failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
