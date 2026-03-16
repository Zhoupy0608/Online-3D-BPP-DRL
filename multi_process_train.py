"""
Multi-Process Training Script for Online 3D Bin Packing

This script provides a streamlined interface for training the bin packing model
using multiple parallel processes as specified in the paper (default: 16 processes).

Features:
- Automatic GPU detection and configuration
- Progress monitoring with ETA estimation
- Configurable process count
- Paper-specified hyperparameters as defaults
- Comprehensive error handling and logging

Usage:
    # Basic training with 16 processes (paper default)
    python multi_process_train.py
    
    # Custom process count
    python multi_process_train.py --num-processes 8
    
    # Force CPU training
    python multi_process_train.py --device cpu
    
    # With reliability features
    python multi_process_train.py --uncertainty-enabled --visual-feedback-enabled

Requirements: 1.1, 7.1
"""

import sys
import os
import time
import argparse
import torch
from datetime import datetime, timedelta


def detect_gpu():
    """
    Detect available GPUs and provide configuration recommendations.
    
    Returns:
        tuple: (has_gpu, gpu_count, gpu_info, recommended_device)
    """
    if not torch.cuda.is_available():
        return False, 0, "No GPU detected", "cpu"
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        name = torch.cuda.get_device_name(i)
        memory_gb = props.total_memory / 1e9
        gpu_info.append(f"GPU {i}: {name} ({memory_gb:.1f} GB)")
    
    # Recommend first GPU by default
    recommended_device = "cuda:0"
    
    return True, gpu_count, "\n  ".join(gpu_info), recommended_device


def estimate_training_time(num_processes, num_updates=10000):
    """
    Estimate training time based on process count.
    
    According to the paper, training takes ~1 day on GPU with 16 processes.
    This provides a rough estimate for different configurations.
    
    Args:
        num_processes: Number of parallel processes
        num_updates: Total number of updates (default: 10000)
        
    Returns:
        str: Estimated training time
    """
    # Base estimate: 1 day for 16 processes, 10000 updates
    base_hours = 24
    base_processes = 16
    base_updates = 10000
    
    # Scale by process count (more processes = faster)
    # Scale by update count
    estimated_hours = (base_hours * base_processes / num_processes) * (num_updates / base_updates)
    
    if estimated_hours < 1:
        return f"~{int(estimated_hours * 60)} minutes"
    elif estimated_hours < 24:
        return f"~{estimated_hours:.1f} hours"
    else:
        days = estimated_hours / 24
        return f"~{days:.1f} days"


def print_training_info(args):
    """
    Print comprehensive training configuration information.
    
    Args:
        args: Parsed command-line arguments
    """
    print("=" * 80)
    print("MULTI-PROCESS TRAINING CONFIGURATION")
    print("=" * 80)
    print()
    
    # Process configuration
    print("Process Configuration:")
    print(f"  Number of processes: {args.num_processes}")
    print(f"  Forward steps per update: {args.num_steps}")
    print(f"  Batch size per update: {args.num_processes * args.num_steps}")
    print()
    
    # Device configuration
    has_gpu, gpu_count, gpu_info, _ = detect_gpu()
    print("Device Configuration:")
    if args.device == "cpu":
        print(f"  Device: CPU (forced)")
        if has_gpu:
            print(f"  Note: {gpu_count} GPU(s) available but not used")
    else:
        if has_gpu:
            print(f"  Device: {args.device}")
            print(f"  Available GPUs:\n  {gpu_info}")
        else:
            print(f"  Device: CPU (no GPU available)")
    print()
    
    # Dataset configuration
    print("Dataset Configuration:")
    print(f"  Item sequence: {args.item_seq}")
    print(f"  Container size: {args.container_size}")
    print(f"  Item size range: {args.item_size_range}")
    print(f"  Enable rotation: {args.enable_rotation}")
    print()
    
    # Hyperparameters (paper-specified)
    print("Hyperparameters (Paper-Specified):")
    print(f"  Learning rate: {args.lr if hasattr(args, 'lr') else 0.25}")
    print(f"  Discount factor (gamma): {args.gamma}")
    print(f"  Value loss coefficient: {args.value_loss_coef}")
    print(f"  Entropy coefficient: {args.entropy_coef}")
    print(f"  Invalid action coefficient: {args.invalid_coef}")
    print()
    
    # KFAC optimizer parameters
    print("KFAC Optimizer Parameters:")
    print(f"  Statistics decay: {args.stat_decay}")
    print(f"  KL clipping: {args.kl_clip}")
    print(f"  Damping: {args.damping}")
    print()
    
    # Reliability features
    if args.uncertainty_enabled or args.visual_feedback_enabled or args.parallel_motion_enabled:
        print("Reliability Features:")
        if args.uncertainty_enabled:
            print(f"  ✓ Uncertainty simulation (std: {args.uncertainty_std})")
        if args.visual_feedback_enabled:
            print(f"  ✓ Visual feedback module")
        if args.parallel_motion_enabled:
            print(f"  ✓ Parallel entry motion (buffer: {args.buffer_range})")
        print()
    
    # Training schedule
    print("Training Schedule:")
    print(f"  Save interval: every {args.save_interval} updates")
    print(f"  Log interval: every {args.log_interval} updates")
    print(f"  TensorBoard: {'enabled' if args.tensorboard else 'disabled'}")
    print()
    
    # Time estimation
    estimated_time = estimate_training_time(args.num_processes)
    print("Estimated Training Time:")
    print(f"  {estimated_time} (for ~10,000 updates)")
    print(f"  Note: Actual time depends on hardware and dataset")
    print()
    
    # Expected results
    print("Expected Results (from paper):")
    print(f"  CUT-2 dataset: ~70% space utilization")
    print(f"  Random sequence: ~73% space utilization")
    print()
    
    print("=" * 80)
    print()


def monitor_progress(start_time, current_update, total_updates, recent_fps):
    """
    Display training progress with ETA estimation.
    
    Args:
        start_time: Training start time
        current_update: Current update number
        total_updates: Total number of updates (if known)
        recent_fps: Recent frames per second
        
    Returns:
        str: Progress string
    """
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    
    if total_updates and current_update > 0:
        progress_pct = (current_update / total_updates) * 100
        remaining = (elapsed / current_update) * (total_updates - current_update)
        eta_str = str(timedelta(seconds=int(remaining)))
        
        return (
            f"Progress: {current_update}/{total_updates} ({progress_pct:.1f}%) | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str} | FPS: {recent_fps}"
        )
    else:
        return f"Updates: {current_update} | Elapsed: {elapsed_str} | FPS: {recent_fps}"


def parse_arguments():
    """
    Parse command-line arguments with paper-specified defaults.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Multi-Process Training for Online 3D Bin Packing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with 16 processes (paper default)
  python multi_process_train.py
  
  # Custom process count
  python multi_process_train.py --num-processes 8
  
  # Force CPU training
  python multi_process_train.py --device cpu
  
  # With reliability features
  python multi_process_train.py --uncertainty-enabled --visual-feedback-enabled
  
  # Continue from checkpoint
  python multi_process_train.py --pretrain --load-name my_model.pt
        """
    )
    
    # Core training parameters
    parser.add_argument(
        '--num-processes', type=int, default=16,
        help='Number of parallel processes (default: 16, as specified in paper)'
    )
    parser.add_argument(
        '--num-steps', type=int, default=5,
        help='Forward steps per update (default: 5, as specified in paper)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device to use: auto, cpu, cuda:0, cuda:1, etc. (default: auto)'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--item-seq', type=str, default='cut_2', choices=['cut1', 'cut2', 'rs'],
        help='Item sequence generator (default: cut_2)'
    )
    parser.add_argument(
        '--container-size', type=int, nargs=3, default=[10, 10, 10],
        help='Container size (x, y, z) (default: 10 10 10)'
    )
    parser.add_argument(
        '--item-size-range', type=int, nargs=6, default=[2, 2, 2, 5, 5, 5],
        help='Item size range (min_x, min_y, min_z, max_x, max_y, max_z) (default: 2 2 2 5 5 5)'
    )
    parser.add_argument(
        '--enable-rotation', action='store_true', default=False,
        help='Enable box rotation'
    )
    
    # Hyperparameters (paper-specified defaults)
    parser.add_argument(
        '--gamma', type=float, default=1.0,
        help='Discount factor (default: 1.0, as specified in paper)'
    )
    parser.add_argument(
        '--value-loss-coef', type=float, default=0.5,
        help='Value loss coefficient (default: 0.5, as specified in paper)'
    )
    parser.add_argument(
        '--entropy-coef', type=float, default=0.01,
        help='Entropy coefficient (default: 0.01, as specified in paper)'
    )
    parser.add_argument(
        '--invalid-coef', type=float, default=2.0,
        help='Invalid action coefficient (default: 2.0, as specified in paper)'
    )
    
    # KFAC optimizer parameters (paper-specified defaults)
    parser.add_argument(
        '--stat-decay', type=float, default=0.99,
        help='KFAC statistics decay (default: 0.99, as specified in paper)'
    )
    parser.add_argument(
        '--kl-clip', type=float, default=0.001,
        help='KFAC KL clipping (default: 0.001, as specified in paper)'
    )
    parser.add_argument(
        '--damping', type=float, default=1e-2,
        help='KFAC damping (default: 1e-2, as specified in paper)'
    )
    
    # Training schedule
    parser.add_argument(
        '--save-interval', type=int, default=10,
        help='Save model every N updates (default: 10)'
    )
    parser.add_argument(
        '--log-interval', type=int, default=10,
        help='Log statistics every N updates (default: 10)'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=True,
        help='Enable TensorBoard logging (default: enabled)'
    )
    parser.add_argument(
        '--no-tensorboard', dest='tensorboard', action='store_false',
        help='Disable TensorBoard logging'
    )
    
    # Model saving
    parser.add_argument(
        '--save-model', action='store_true', default=True,
        help='Save model checkpoints (default: enabled)'
    )
    parser.add_argument(
        '--no-save-model', dest='save_model', action='store_false',
        help='Disable model saving'
    )
    
    # Checkpoint loading
    parser.add_argument(
        '--pretrain', action='store_true', default=False,
        help='Load pretrained model'
    )
    parser.add_argument(
        '--load-name', type=str, default='default_cut_2.pt',
        help='Pretrained model filename (default: default_cut_2.pt)'
    )
    parser.add_argument(
        '--load-dir', type=str, default='./pretrained_models/',
        help='Directory containing pretrained models (default: ./pretrained_models/)'
    )
    parser.add_argument(
        '--save-dir', type=str, default='./saved_models/',
        help='Directory to save models (default: ./saved_models/)'
    )
    
    # Reliability features
    parser.add_argument(
        '--uncertainty-enabled', action='store_true', default=False,
        help='Enable placement uncertainty simulation'
    )
    parser.add_argument(
        '--uncertainty-std-x', type=float, default=0.5,
        help='Uncertainty std in x direction (default: 0.5)'
    )
    parser.add_argument(
        '--uncertainty-std-y', type=float, default=0.5,
        help='Uncertainty std in y direction (default: 0.5)'
    )
    parser.add_argument(
        '--uncertainty-std-z', type=float, default=0.1,
        help='Uncertainty std in z direction (default: 0.1)'
    )
    parser.add_argument(
        '--visual-feedback-enabled', action='store_true', default=False,
        help='Enable visual feedback module'
    )
    parser.add_argument(
        '--parallel-motion-enabled', action='store_true', default=False,
        help='Enable parallel entry motion primitive'
    )
    parser.add_argument(
        '--buffer-range-x', type=int, default=1,
        help='Buffer range in x direction (default: 1)'
    )
    parser.add_argument(
        '--buffer-range-y', type=int, default=1,
        help='Buffer range in y direction (default: 1)'
    )
    parser.add_argument(
        '--camera-config', type=str, default=None,
        help='Path to camera configuration file'
    )
    
    # Other parameters
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Random seed (default: 1)'
    )
    parser.add_argument(
        '--hidden-size', type=int, default=256,
        help='Hidden layer size (default: 256)'
    )
    
    args = parser.parse_args()
    
    # Post-process arguments
    args.container_size = tuple(args.container_size)
    args.item_size_range = tuple(args.item_size_range)
    args.uncertainty_std = (args.uncertainty_std_x, args.uncertainty_std_y, args.uncertainty_std_z)
    args.buffer_range = (args.buffer_range_x, args.buffer_range_y)
    
    # Auto-detect device
    if args.device == 'auto':
        has_gpu, _, _, recommended_device = detect_gpu()
        args.device = recommended_device if has_gpu else 'cpu'
    
    # Convert device string for compatibility with main.py
    if args.device != 'cpu':
        args.use_cuda = True
        # Extract device number from cuda:N format
        if ':' in args.device:
            device_num = int(args.device.split(':')[1])
        else:
            device_num = 0
        args.device = device_num  # main.py expects integer for CUDA device
    else:
        args.use_cuda = False
        args.device = 'cpu'
    
    return args


def main():
    """
    Main entry point for multi-process training.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Print training configuration
    print_training_info(args)
    
    # Prompt for test name
    print("Please enter a name for this training run:")
    print("(This will be used for saving models and TensorBoard logs)")
    test_name = input("Test name: ").strip()
    
    if not test_name:
        test_name = f"multi_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Using default name: {test_name}")
    
    print()
    print("=" * 80)
    print("STARTING MULTI-PROCESS TRAINING")
    print("=" * 80)
    print()
    print("Training will begin shortly...")
    print("Press Ctrl+C to stop training and save an emergency checkpoint")
    print()
    
    # Prepare arguments for main.py
    sys.argv = [
        'main.py',
        '--mode', 'train',
        '--algorithm', 'acktr',
        '--num-processes', str(args.num_processes),
        '--num-steps', str(args.num_steps),
        '--item-seq', args.item_seq,
        '--container-size', str(args.container_size[0]), str(args.container_size[1]), str(args.container_size[2]),
        '--item-size-range', *[str(x) for x in args.item_size_range],
        '--gamma', str(args.gamma),
        '--value-loss-coef', str(args.value_loss_coef),
        '--entropy-coef', str(args.entropy_coef),
        '--invalid-coef', str(args.invalid_coef),
        '--stat-decay', str(args.stat_decay),
        '--kl-clip', str(args.kl_clip),
        '--damping', str(args.damping),
        '--save-interval', str(args.save_interval),
        '--log-interval', str(args.log_interval),
        '--seed', str(args.seed),
        '--hidden-size', str(args.hidden_size),
        '--load-dir', args.load_dir,
        '--save-dir', args.save_dir,
    ]
    
    # Add device argument
    if args.use_cuda:
        sys.argv.extend(['--use-cuda', '--device', str(args.device)])
    
    # Add optional flags
    if args.enable_rotation:
        sys.argv.append('--enable-rotation')
    if args.tensorboard:
        sys.argv.append('--tensorboard')
    if args.save_model:
        sys.argv.append('--save-model')
    if args.pretrain:
        sys.argv.extend(['--pretrain', '--load-name', args.load_name])
    
    # Add reliability features
    if args.uncertainty_enabled:
        sys.argv.extend([
            '--uncertainty-enabled',
            '--uncertainty-std-x', str(args.uncertainty_std[0]),
            '--uncertainty-std-y', str(args.uncertainty_std[1]),
            '--uncertainty-std-z', str(args.uncertainty_std[2])
        ])
    if args.visual_feedback_enabled:
        sys.argv.append('--visual-feedback-enabled')
        if args.camera_config:
            sys.argv.extend(['--camera-config', args.camera_config])
    if args.parallel_motion_enabled:
        sys.argv.extend([
            '--parallel-motion-enabled',
            '--buffer-range-x', str(args.buffer_range[0]),
            '--buffer-range-y', str(args.buffer_range[1])
        ])
    
    # Import and run main training
    try:
        # Store test name for main.py to use
        os.environ['TRAINING_TEST_NAME'] = test_name
        
        # Import main module
        import main as training_main
        
        # Register environments
        training_main.registration_envs()
        
        # Get arguments from main.py's argument parser
        from acktr.arguments import get_args
        training_args = get_args()
        
        # Start training
        start_time = time.time()
        training_main.main(training_args)
        
        # Training completed
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print()
        print("=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total training time: {elapsed_str}")
        print(f"Models saved to: {args.save_dir}/{test_name}/")
        if args.tensorboard:
            print(f"TensorBoard logs: ./runs/Bpp-v0/{test_name}/")
            print(f"View with: tensorboard --logdir=./runs")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 80)
        print("Emergency checkpoint saved (if applicable)")
        print("You can resume training by using --pretrain with the saved checkpoint")
        print("=" * 80)
        sys.exit(0)
        
    except Exception as e:
        print()
        print("=" * 80)
        print("TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print()
        print("Check the log files for detailed error information:")
        print("  - ./logs/training_*.log")
        print("  - Emergency checkpoints (if any): ./emergency_checkpoints/")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
