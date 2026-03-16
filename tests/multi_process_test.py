import sys
import os
import argparse
import multiprocessing
import numpy as np
import torch
from unified_test import unified_test
from acktr.arguments import get_args as get_base_args

# Import gym compatibility layer to fix gym/gymnasium issues
import gym_compatibility
import gymnasium as gym

def worker_process(args, start_idx, end_idx, result_queue):
    """
    Worker process function to run a subset of test cases
    
    Args:
        args: Command-line arguments
        start_idx: Start index for test cases
        end_idx: End index for test cases
        result_queue: Queue to store results
    """
    # Set a unique seed for each process
    torch.manual_seed(args.seed + start_idx)
    np.random.seed(args.seed + start_idx)
    
    # Create a copy of args with the specific cases for this process
    process_args = argparse.Namespace(**vars(args))
    process_args.cases = end_idx - start_idx
    
    try:
        print(f"Process {multiprocessing.current_process().name}: Running test cases {start_idx}-{end_idx-1} ({process_args.cases} cases)")
        mean_ratio, mean_reward = unified_test(process_args.load_dir + process_args.load_name, process_args)
        result_queue.put((mean_ratio, mean_reward, process_args.cases))
        print(f"Process {multiprocessing.current_process().name}: Completed with mean ratio {mean_ratio:.3f}")
    except Exception as e:
        print(f"Process {multiprocessing.current_process().name}: Error occurred: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((None, None, process_args.cases))

def main():
    """
    Main function for multi-process testing
    """
    # Get base arguments
    args = get_base_args()
    
    # Set test mode explicitly
    args.mode = 'test'
    args.load_model = True
    args.test = True
    
    # Validate arguments
    if not args.load_model:
        print("Error: --load-model must be specified for testing")
        sys.exit(1)
    
    if args.num_processes <= 0:
        print("Error: --num_processes must be greater than 0")
        sys.exit(1)
    
    if args.cases <= 0:
        print("Error: --cases must be greater than 0")
        sys.exit(1)
    
    print(f"=== 进入多进程测试模式 ===")
    print(f"测试模型: {args.load_dir + args.load_name}")
    print(f"数据集: {args.data_name}")
    print(f"容器大小: {args.container_size}")
    print(f"测试用例总数: {args.cases}")
    print(f"进程数: {args.num_processes}")
    print(f"设备: {args.device}")
    print(f"是否启用旋转: {args.enable_rotation}")
    
    # Calculate cases per process
    cases_per_process = args.cases // args.num_processes
    remaining_cases = args.cases % args.num_processes
    
    # Create result queue
    result_queue = multiprocessing.Queue()
    
    # Create and start processes
    processes = []
    start_idx = 0
    
    for i in range(args.num_processes):
        # Distribute remaining cases to the first few processes
        current_cases = cases_per_process + (1 if i < remaining_cases else 0)
        end_idx = start_idx + current_cases
        
        # Skip if no cases for this process
        if current_cases <= 0:
            break
        
        # Create process
        p = multiprocessing.Process(
            target=worker_process,
            name=f"TestWorker-{i+1}",
            args=(args, start_idx, end_idx, result_queue)
        )
        
        processes.append(p)
        p.start()
        start_idx = end_idx
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results
    all_ratios = []
    all_rewards = []
    total_cases = 0
    
    while not result_queue.empty():
        mean_ratio, mean_reward, cases = result_queue.get()
        if mean_ratio is not None and mean_reward is not None:
            all_ratios.append(mean_ratio)
            all_rewards.append(mean_reward)
            total_cases += cases
    
    # Calculate and display aggregated results
    print("\n" + "="*50)
    print("多进程测试结果汇总")
    print("="*50)
    
    if all_ratios:
        overall_mean_ratio = np.mean(all_ratios)
        overall_mean_reward = np.mean(all_rewards)
        overall_std_ratio = np.std(all_ratios)
        
        print(f"总测试用例数: {total_cases}")
        print(f"平均空间利用率: {overall_mean_ratio:.3f} ± {overall_std_ratio:.3f} ({overall_mean_ratio * 100:.1f}%)")
        print(f"平均奖励: {overall_mean_reward:.2f}")
        print(f"最佳进程空间利用率: {max(all_ratios):.3f} ({max(all_ratios) * 100:.1f}%)")
        print(f"最差进程空间利用率: {min(all_ratios):.3f} ({min(all_ratios) * 100:.1f}%)")
        
        # Check if target is achieved
        target_ratio = 0.75
        if overall_mean_ratio >= target_ratio:
            print(f"✅ TARGET ACHIEVED! Mean utilization {overall_mean_ratio * 100:.1f}% >= {target_ratio * 100:.1f}%")
        else:
            improvement_needed = target_ratio - overall_mean_ratio
            print(f"❌ Target not reached. Need {improvement_needed * 100:.1f}% more utilization to reach {target_ratio * 100:.1f}%")
    else:
        print("Error: No valid results collected from any process")
        sys.exit(1)

if __name__ == '__main__':
    # Set start method for Windows compatibility
    if sys.platform == 'win32':
        multiprocessing.set_start_method('spawn')
    main()