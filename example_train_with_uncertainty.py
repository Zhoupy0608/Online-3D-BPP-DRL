#!/usr/bin/env python3
"""
Example Training Script with Uncertainty Simulation

This script demonstrates how to train a DRL agent with uncertainty simulation enabled.
The uncertainty simulation adds random noise to placement positions during training,
making the learned policy more robust to real-world placement errors.

Usage:
    python example_train_with_uncertainty.py

Features enabled:
    - Uncertainty simulation with Gaussian noise
    - Standard deviation: (0.5, 0.5, 0.1) for (x, y, z)
    - Training on CUT-2 dataset
    - Model saving enabled

Requirements: All (Training with simulated uncertainty - Requirement 8)
"""

import subprocess
import sys

def main():
    """
    Train a DRL agent with uncertainty simulation enabled.
    
    This example trains on the CUT-2 dataset with uncertainty simulation,
    which adds Gaussian noise to placement positions to simulate real-world
    placement errors. The trained model will be more robust when deployed
    on physical robots.
    """
    
    print("=" * 70)
    print("Training DRL Agent with Uncertainty Simulation")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - Dataset: CUT-2")
    print("  - Uncertainty enabled: Yes")
    print("  - Uncertainty std: (0.5, 0.5, 0.1) for (x, y, z)")
    print("  - Save interval: 10 updates")
    print()
    print("This training will:")
    print("  1. Add random Gaussian noise to placement positions")
    print("  2. Adjust positions if noise causes collisions")
    print("  3. Train the agent to be robust to placement errors")
    print("  4. Save models with uncertainty configuration")
    print()
    print("=" * 70)
    print()
    
    # Build command with all necessary arguments
    cmd = [
        sys.executable,  # Use current Python interpreter
        "main.py",
        
        # Basic training configuration
        "--mode", "train",
        "--item-seq", "cut2",
        
        # Reliability features - UNCERTAINTY ENABLED
        "--uncertainty-enabled",
        "--uncertainty-std-x", "0.5",
        "--uncertainty-std-y", "0.5",
        "--uncertainty-std-z", "0.1",
        
        # Model saving
        "--save_model",  # 注意：使用下划线
        "--save_interval", "10",
        "--log_interval", "10",
        
        # Optional: Enable tensorboard for monitoring
        "--tensorboard",
        
        # Random seed for reproducibility
        "--seed", "42",
    ]
    
    # Optional: Add CUDA support if available
    try:
        import torch
        if torch.cuda.is_available():
            cmd.extend(["--use-cuda", "--device", "0"])
            print("CUDA detected - GPU training enabled")
            print()
    except ImportError:
        print("PyTorch not found - using CPU training")
        print()
    
    print("Starting training...")
    print("Command:", " ".join(cmd))
    print()
    print("Note: You will be prompted to enter a test name for this training run.")
    print()
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    
    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)
    print()
    print("The trained model has been saved with uncertainty configuration.")
    print("You can now test it using example_test_with_reliability.py")
    print()


if __name__ == "__main__":
    main()
