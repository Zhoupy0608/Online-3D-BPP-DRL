"""
Improved training script with better hyperparameters and curriculum learning.
This script implements the training improvements from TRAINING_IMPROVEMENT_GUIDE.md
"""

import subprocess
import sys
import os
import time

def run_training_stage(stage_name, args, expected_steps=300000):
    """
    Run a training stage with specified arguments.
    
    Args:
        stage_name: Name of the training stage for logging
        args: List of command-line arguments
        expected_steps: Expected number of training steps
    """
    print(f"\n{'='*60}")
    print(f"Starting Training Stage: {stage_name}")
    print(f"Expected duration: ~{expected_steps // 10 // 60} minutes at 10 FPS")
    print(f"{'='*60}\n")
    
    cmd = [sys.executable, "main.py"] + args
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n{stage_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{stage_name} failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n{stage_name} interrupted by user")
        return False

def main():
    """
    Run the complete improved training pipeline with curriculum learning.
    """
    print("="*60)
    print("Improved Training Pipeline for Reliable Robot Packing")
    print("="*60)
    print("\nThis script will run 4 training stages:")
    print("1. Basic training (no noise, simple task)")
    print("2. Full task (no noise, full complexity)")
    print("3. Small noise (reliability features)")
    print("4. Target noise (full reliability)")
    print("\nTotal expected time: ~18-24 hours")
    print("="*60)
    
    response = input("\nDo you want to start the full pipeline? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Training cancelled.")
        return
    
    # Stage 1: Basic training (no noise, simpler task)
    print("\n\nSTAGE 1: Basic Training")
    print("Goal: Learn basic packing without noise")
    print("Target: Space ratio > 0.30")
    
    stage1_args = [
        "--env_name", "Bpp-v0",
        "--container_size", "8", "8", "8",
        "--item-size-range", "2", "2", "2", "4", "4", "4",
        "--learning_rate", "1e-4",
        "--entropy_coef", "0.05",
        "--value_loss_coef", "0.25",
        "--save_model",
        "--num_processes", "1",
        "--num_steps", "10",
        "--save_interval", "100",
        "--log_interval", "10",
        "--tensorboard"
    ]
    
    if not run_training_stage("Stage 1: Basic Training", stage1_args, 300000):
        print("\nStage 1 failed. Stopping pipeline.")
        return
    
    # Find the latest saved model from stage 1
    save_dir = "./saved_models/"
    stage1_model = find_latest_model(save_dir)
    if not stage1_model:
        print("\nCould not find Stage 1 model. Stopping pipeline.")
        return
    
    print(f"\nStage 1 model saved as: {stage1_model}")
    time.sleep(3)
    
    # Stage 2: Full task (no noise, full complexity)
    print("\n\nSTAGE 2: Full Task Training")
    print("Goal: Scale to full container size")
    print("Target: Space ratio > 0.35")
    
    stage2_args = [
        "--env_name", "Bpp-v0",
        "--container_size", "10", "10", "10",
        "--item-size-range", "2", "2", "2", "5", "5", "5",
        "--learning_rate", "1e-4",
        "--entropy_coef", "0.03",
        "--value_loss_coef", "0.25",
        "--load-model",
        "--load-name", stage1_model,
        "--save_model",
        "--num_processes", "1",
        "--num_steps", "10",
        "--save_interval", "100",
        "--log_interval", "10",
        "--tensorboard"
    ]
    
    if not run_training_stage("Stage 2: Full Task", stage2_args, 300000):
        print("\nStage 2 failed. Stopping pipeline.")
        return
    
    stage2_model = find_latest_model(save_dir)
    print(f"\nStage 2 model saved as: {stage2_model}")
    time.sleep(3)
    
    # Stage 3: Small noise (reliability features)
    print("\n\nSTAGE 3: Small Noise Training")
    print("Goal: Adapt to small placement uncertainty")
    print("Target: Space ratio > 0.30")
    
    stage3_args = [
        "--env_name", "BppReliable-v0",
        "--uncertainty-enabled",
        "--uncertainty-std-x", "0.2",
        "--uncertainty-std-y", "0.2",
        "--uncertainty-std-z", "0.05",
        "--learning_rate", "5e-5",
        "--entropy_coef", "0.02",
        "--value_loss_coef", "0.25",
        "--load-model",
        "--load-name", stage2_model,
        "--save_model",
        "--num_processes", "1",
        "--num_steps", "10",
        "--save_interval", "100",
        "--log_interval", "10",
        "--tensorboard"
    ]
    
    if not run_training_stage("Stage 3: Small Noise", stage3_args, 200000):
        print("\nStage 3 failed. Stopping pipeline.")
        return
    
    stage3_model = find_latest_model(save_dir)
    print(f"\nStage 3 model saved as: {stage3_model}")
    time.sleep(3)
    
    # Stage 4: Target noise (full reliability)
    print("\n\nSTAGE 4: Full Reliability Training")
    print("Goal: Handle target noise level with all features")
    print("Target: Space ratio > 0.25")
    
    stage4_args = [
        "--env_name", "BppReliable-v0",
        "--uncertainty-enabled",
        "--uncertainty-std-x", "0.5",
        "--uncertainty-std-y", "0.5",
        "--uncertainty-std-z", "0.1",
        "--visual-feedback-enabled",
        "--parallel-motion-enabled",
        "--learning_rate", "5e-5",
        "--entropy_coef", "0.01",
        "--value_loss_coef", "0.25",
        "--load-model",
        "--load-name", stage3_model,
        "--save_model",
        "--num_processes", "1",
        "--num_steps", "10",
        "--save_interval", "100",
        "--log_interval", "10",
        "--tensorboard"
    ]
    
    if not run_training_stage("Stage 4: Full Reliability", stage4_args, 200000):
        print("\nStage 4 failed. Stopping pipeline.")
        return
    
    stage4_model = find_latest_model(save_dir)
    print(f"\nStage 4 model saved as: {stage4_model}")
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED!")
    print("="*60)
    print(f"\nFinal model: {stage4_model}")
    print("\nYou can now test the model with:")
    print(f"python main.py --mode test --load-model --load-name {stage4_model}")
    print("\nOr view training progress with:")
    print("tensorboard --logdir=./runs")

def find_latest_model(save_dir):
    """
    Find the most recently saved model in the save directory.
    
    Args:
        save_dir: Directory containing saved models
        
    Returns:
        Filename of the latest model, or None if no models found
    """
    if not os.path.exists(save_dir):
        return None
    
    # Get all .pt files
    models = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    if not models:
        return None
    
    # Sort by modification time
    models.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)), reverse=True)
    
    return models[0]

if __name__ == "__main__":
    main()
