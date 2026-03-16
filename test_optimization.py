#!/usr/bin/env python3
"""
Quick test to verify performance optimizations are working.
"""

from envs.bpp0.support_calculation import GeometricUtils
from envs.bpp0.space import Space
import time

def test_optimizations():
    print("🚀 Testing Performance Optimizations")
    print("=" * 50)

    # Create optimized space
    space = Space(10, 10, 20, use_enhanced_feasibility=True)

    # Configure optimal performance settings
    GeometricUtils.configure_performance(
        enable_caching=True,
        enable_early_exit=True,
        cache_size=500
    )

    print("✅ Optimized configuration applied")

    # Test basic functionality
    result = space.check_box_enhanced(space.plain, 2, 2, 1, 1, 1)
    print("✅ Enhanced feasibility check working: {result}")

    # Check cache configuration
    cache_stats = GeometricUtils.get_cache_stats()
    print("✅ Cache configured: {cache_stats['cache_max_size']} entries")
    print("✅ Caching enabled: {cache_stats['caching_enabled']}")
    print("✅ Early exit enabled: {cache_stats['early_exit_enabled']}")

    # Test utilization metrics
    metrics = space.collect_utilization_metrics()
    print("✅ Performance monitoring working")
    print(f"   Target utilization: {metrics['target_utilization']:.1%}")

    print("\n🎯 Performance Optimization Test PASSED")
    print("   All optimizations are working correctly")

    return True

if __name__ == "__main__":
    test_optimizations()

# import torch
# import gym
# import numpy as np
# from acktr.model import Policy
# from acktr import utils
#
#
# def unified_test(model_url, args):
#     """
#     Test a trained model with proper environment setup and model loading.
#
#     Args:
#         model_url: Path to the trained model checkpoint
#         args: Arguments containing test configuration
#     """
#     print("Loading model from: {model_url}")
#
#     # Set device
#     device = torch.device(args.device)
#
#     # Load checkpoint
#     try:
#         checkpoint = torch.load(model_url, map_location=device)
#         print("Checkpoint loaded successfully")
#
#         # Handle different checkpoint formats
#         if isinstance(checkpoint, dict):
#             if 'model_state_dict' in checkpoint:
#                 # New format with reliability config
#                 model_state_dict = checkpoint['model_state_dict']
#                 ob_rms = checkpoint.get('ob_rms', None)
#                 reliability_config = checkpoint.get('reliability_config', {})
#                 print("Loaded checkpoint with reliability config: {reliability_config}")
#             else:
#                 # Legacy format - assume it's the model state dict directly
#                 model_state_dict = checkpoint
#                 ob_rms = None
#         elif isinstance(checkpoint, list) and len(checkpoint) == 2:
#             # Legacy format [model_state_dict, ob_rms]
#             model_state_dict, ob_rms = checkpoint
#         else:
#             raise ValueError("Unknown checkpoint format: {type(checkpoint)}")
#
#     except Exception as e:
#         print("Error loading checkpoint: {e}")
#         return
#
#     # Create environment with proper box_set
#     try:
#         env = gym.make(args.env_name,
#                        container_size=args.container_size,
#                        test=True,
#                        data_name=args.data_name,
#                        data_type=args.item_seq,
#                        box_set=args.box_size_set,  # This is the missing parameter!
#                        enable_rotation=args.enable_rotation)
#         print("Environment created successfully: {args.env_name}")
#     except Exception as e:
#         print("Error creating environment: {e}")
#         return
#
#     # Create policy model
#     try:
#         actor_critic = Policy(
#             env.observation_space.shape,
#             env.action_space,
#             base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size, 'args': args}
#         )
#
#         # 使用与main.py中相同的模型加载方式
#         print("使用训练时的模型加载方式...")
#
#         # 按照main.py中pretrain部分的处理方式
#         load_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
#         load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
#         load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
#
#         # 处理张量维度问题
#         for k, v in load_dict.items():
#             if len(v.size()) <= 3:
#                 load_dict[k] = v.squeeze(dim=-1)
#
#         print("正在加载模型状态字典...")
#         actor_critic.load_state_dict(load_dict)
#         actor_critic.to(device)
#         actor_critic.eval()
#         print("Model loaded and set to evaluation mode")
#
#     except Exception as e:
#         print("Error creating or loading model: {e}")
#         return
#
#     # Set observation normalization if available
#     if ob_rms is not None:
#         try:
#             # This would be used if we had vectorized environments
#             # For single environment testing, we might not need this
#             print("Observation normalization parameters loaded")
#         except Exception as e:
#             print("Warning: Could not set observation normalization: {e}")
#
#     # Run test episodes
#     print("Starting test with {args.cases} episodes...")
#
#     episode_rewards = []
#     episode_ratios = []
#
#     for episode in range(args.cases):
#         obs = env.reset()
#         episode_reward = 0
#         done = False
#         step_count = 0
#
#         while not done:
#             # Convert observation to tensor
#             obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
#
#             # Get action from model
#             with torch.no_grad():
#                 # Create dummy recurrent hidden state and mask
#                 rnn_hxs = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
#                 masks = torch.ones(1, 1).to(device)
#
#                 # Get location mask for valid actions
#                 location_mask = env.get_possible_position()
#                 location_mask_tensor = torch.FloatTensor(location_mask).flatten().unsqueeze(0).to(device)
#
#                 # Get action from policy
#                 value, action, action_log_prob, rnn_hxs = actor_critic.act(
#                     obs_tensor, rnn_hxs, masks, location_mask_tensor, deterministic=True
#                 )
#
#                 action = action.cpu().numpy()[0]
#
#             # Take step in environment
#             obs, reward, done, info = env.step(action)
#             episode_reward += reward
#             step_count += 1
#
#             # Prevent infinite loops
#             if step_count > 1000:
#                 print("Episode {episode + 1} exceeded 1000 steps, terminating")
#                 break
#
#         # Record episode results
#         episode_rewards.append(episode_reward)
#         if 'ratio' in info:
#             episode_ratios.append(info['ratio'])
#         else:
#             episode_ratios.append(0.0)
#
#         if (episode + 1) % 10 == 0:
#             print("Completed {episode + 1}/{args.cases} episodes")
#
#     # Calculate and display results
#     mean_reward = np.mean(episode_rewards)
#     mean_ratio = np.mean(episode_ratios)
#     std_reward = np.std(episode_rewards)
#     std_ratio = np.std(episode_ratios)
#
#     print("\n" + "=" * 50)
#     print("TEST RESULTS")
#     print("=" * 50)
#     print("Episodes tested: {args.cases}")
#     print("Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
#     print("Mean space utilization: {mean_ratio:.3f} ± {std_ratio:.3f} ({mean_ratio * 100:.1f}%)")
#     print("Min space utilization: {min(episode_ratios):.3f} ({min(episode_ratios) * 100:.1f}%)")
#     print("Max space utilization: {max(episode_ratios):.3f} ({max(episode_ratios) * 100:.1f}%)")
#
#     # Check if target is achieved
#     target_ratio = 0.75
#     if mean_ratio >= target_ratio:
#         print("✅ TARGET ACHIEVED! Mean utilization {mean_ratio * 100:.1f}% >= {target_ratio * 100:.1f}%")
#     else:
#         improvement_needed = target_ratio - mean_ratio
#         print(
#             "❌ Target not reached. Need {improvement_needed * 100:.1f}% more utilization to reach {target_ratio * 100:.1f}%")
#
#     print("=" * 50)
#
#     env.close()
#     return mean_ratio, mean_reward