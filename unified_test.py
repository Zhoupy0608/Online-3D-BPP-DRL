# # -*- coding: utf-8 -*-
# import torch
# import numpy as np
# from acktr.model import Policy
# import gym
#
# def unified_test(model_path, args):
#     print("Loading model from:", model_path)
#     device = torch.device("cuda" if args.use_cuda else "cpu")
#
#     try:
#         # Load checkpoint
#         checkpoint = torch.load(model_path, map_location=device)
#         print("Checkpoint loaded successfully")
#
#         # Create model with proper box_set
#         box_set = args.box_size_set if hasattr(args, 'box_size_set') else [(2,2,2), (5,5,5)]
#         env = gym.make(args.env_name,
#                       container_size=args.container_size,
#                       box_set=box_set,
#                       test=False,
#                       data_type=args.item_seq)
#
#         # Create model with args parameter
#         actor_critic = Policy(env.observation_space.shape, env.action_space, base_kwargs={'recurrent': False, 'args': args})
#         actor_critic.to(device)
#
#         # Load state dict
#         actor_critic.load_state_dict(checkpoint['model_state_dict'])
#         actor_critic.eval()
#         print("Model loaded and ready")
#
#         # Test environment
#         if hasattr(args, 'data_name') and args.data_name:
#             test_env = gym.make(args.env_name,
#                               container_size=args.container_size,
#                               box_set=box_set,
#                               test=True,
#                               data_name=args.data_name,
#                               data_type=args.item_seq)
#         else:
#             test_env = gym.make(args.env_name,
#                               container_size=args.container_size,
#                               box_set=box_set,
#                               test=False,
#                               data_type=args.item_seq)
#
#         print("Test environment created")
#
#         # Run test episodes
#         results = []
#         num_episodes = 10
#         print("Running", num_episodes, "test episodes...")
#
#         for episode in range(num_episodes):
#             obs = test_env.reset()
#             done = False
#             steps = 0
#             episode_reward = 0
#
#             while not done and steps < 200:
#                 with torch.no_grad():
#                     obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
#                     _, action, _, _ = actor_critic.act(obs_tensor, None, None, deterministic=True)
#                     action = action.cpu().numpy()[0]
#
#                 obs, reward, done, info = test_env.step(action)
#                 episode_reward += reward
#                 steps += 1
#
#             ratio = info.get('ratio', 0)
#             counter = info.get('counter', 0)
#             results.append(ratio)
#             print("Episode", episode+1, "- Ratio:", round(ratio, 3), "Items:", counter, "Steps:", steps)
#
#         test_env.close()
#         env.close()
#
#         # Calculate results
#         avg_ratio = np.mean(results)
#         max_ratio = np.max(results)
#         min_ratio = np.min(results)
#
#         print()
#         print("=== TEST RESULTS ===")
#         print("Episodes:", len(results))
#         print("Average Ratio:", round(avg_ratio, 3), "(" + str(round(avg_ratio*100, 1)) + "%)")
#         print("Best Ratio:", round(max_ratio, 3), "(" + str(round(max_ratio*100, 1)) + "%)")
#         print("Worst Ratio:", round(min_ratio, 3), "(" + str(round(min_ratio*100, 1)) + "%)")
#
#         if avg_ratio >= 0.75:
#             print("SUCCESS! Target achieved (75%+)")
#         elif avg_ratio >= 0.68:
#             improvement = round(((avg_ratio - 0.68) / 0.68) * 100, 1)
#             print("GOOD! Above baseline, improvement:", str(improvement) + "%")
#         else:
#             print("Below baseline 68%")
#
#     except Exception as e:
#         print("Error:", e)
#         import traceback
#         traceback.print_exc()


import torch
import numpy as np
from acktr.model import Policy
from acktr import utils

# Import gym compatibility layer to fix gym/gymnasium issues
import gym_compatibility
import gymnasium as gym

# Register custom environments
import register_env


def unified_test(model_url, args):
    """
    Test a trained model with proper environment setup and model loading.

    Args:
        model_url: Path to the trained model checkpoint
        args: Arguments containing test configuration
    """
    print("Loading model from:", model_url)

    # Set device
    device = torch.device(args.device)

    # Load checkpoint
    try:
        checkpoint = torch.load(model_url, map_location=device)
        print("Checkpoint loaded successfully")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # New format with reliability config
                model_state_dict = checkpoint['model_state_dict']
                ob_rms = checkpoint.get('ob_rms', None)
                reliability_config = checkpoint.get('reliability_config', {})
                print("Loaded checkpoint with reliability config:", reliability_config)
            else:
                # Legacy format - assume it's the model state dict directly
                model_state_dict = checkpoint
                ob_rms = None
        elif isinstance(checkpoint, list) and len(checkpoint) == 2:
            # Legacy format [model_state_dict, ob_rms]
            model_state_dict, ob_rms = checkpoint
        else:
            raise ValueError("Unknown checkpoint format: " + str(type(checkpoint)))

    except Exception as e:
        print("Error loading checkpoint:", str(e))
        return

    # Create environment with proper box_set
    try:
        env = gym.make(args.env_name,
                       container_size=args.container_size,
                       test=True,
                       data_name=args.data_name,
                       data_type=args.item_seq,
                       box_set=args.box_size_set,  # This is the missing parameter!
                       enable_rotation=args.enable_rotation)
        print("Environment created successfully:", args.env_name)
    except Exception as e:
        print("Error creating environment:", str(e))
        print("Debug info:")
        print("  - env_name:", args.env_name)
        print("  - container_size:", args.container_size)
        print("  - data_name:", args.data_name)
        print("  - item_seq:", args.item_seq)
        print("  - box_size_set length:", len(args.box_size_set) if hasattr(args, 'box_size_set') else "NOT FOUND")
        print("  - enable_rotation:", args.enable_rotation)
        return

    # Create policy model
    try:
        actor_critic = Policy(
            env.observation_space.shape,
            env.action_space,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size, 'args': args}
        )

        # Load model state with parameter name conversion
        # Handle DataParallel and add_bias naming differences
        load_dict = {}
        for k, v in model_state_dict.items():
            # Remove 'module.' prefix if present (from DataParallel)
            new_key = k.replace('module.', '')
            # Convert add_bias naming
            new_key = new_key.replace('add_bias.', '')
            new_key = new_key.replace('_bias', 'bias')

            # Handle shape mismatch for bias parameters
            # Training used [N, 1] shape, but testing expects [N] shape
            if 'bias' in new_key and len(v.shape) > 1:
                v = v.squeeze(-1)  # Remove the last dimension if it's 1

            load_dict[new_key] = v

        actor_critic.load_state_dict(load_dict)
        actor_critic.to(device)
        actor_critic.eval()
        print("Model loaded and set to evaluation mode")

    except Exception as e:
        print("Error creating or loading model:", str(e))
        return

    # Set observation normalization if available
    if ob_rms is not None:
        try:
            # This would be used if we had vectorized environments
            # For single environment testing, we might not need this
            print("Observation normalization parameters loaded")
        except Exception as e:
            print("Warning: Could not set observation normalization:", str(e))

    # Run test episodes
    print("Starting test with", args.cases, "episodes...")

    episode_rewards = []
    episode_ratios = []

    for episode in range(args.cases):
        # Handle both gymnasium and gym reset APIs
        try:
            obs, info = env.reset()
        except ValueError:
            # Fallback for gym API
            obs = env.reset()
            info = {}
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            # Get action from model
            with torch.no_grad():
                # Create dummy recurrent hidden state and mask
                rnn_hxs = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
                masks = torch.ones(1, 1).to(device)

                # Get location mask for valid actions
                location_mask = env.unwrapped.get_possible_position()
                location_mask_tensor = torch.FloatTensor(location_mask).flatten().unsqueeze(0).to(device)

                # Get action from policy
                value, action, action_log_prob, rnn_hxs = actor_critic.act(
                    obs_tensor, rnn_hxs, masks, location_mask_tensor, deterministic=True
                )

                action = action.cpu().numpy()[0]

            # Take step in environment (gymnasium API)
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # Fallback for gym API
                obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Prevent infinite loops
            if step_count > 1000:
                print("Episode", episode + 1, "exceeded 1000 steps, terminating")
                break

        # Record episode results
        episode_rewards.append(episode_reward)
        if 'ratio' in info:
            episode_ratios.append(info['ratio'])
        else:
            episode_ratios.append(0.0)

        if (episode + 1) % 10 == 0:
            print("Completed", episode + 1, "/", args.cases, "episodes")

    # Calculate and display results
    mean_reward = np.mean(episode_rewards)
    mean_ratio = np.mean(episode_ratios)
    std_reward = np.std(episode_rewards)
    std_ratio = np.std(episode_ratios)

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print("Episodes tested:", args.cases)
    print("Mean reward: {:.2f} ± {:.2f}".format(mean_reward, std_reward))
    print("Mean space utilization: {:.3f} ± {:.3f} ({:.1f}%)".format(mean_ratio, std_ratio, mean_ratio * 100))
    print("Min space utilization: {:.3f} ({:.1f}%)".format(min(episode_ratios), min(episode_ratios) * 100))
    print("Max space utilization: {:.3f} ({:.1f}%)".format(max(episode_ratios), max(episode_ratios) * 100))

    # Check if target is achieved
    target_ratio = 0.75
    if mean_ratio >= target_ratio:
        print("✅ TARGET ACHIEVED! Mean utilization {:.1f}% >= {:.1f}%".format(mean_ratio * 100, target_ratio * 100))
    else:
        improvement_needed = target_ratio - mean_ratio
        print("❌ Target not reached. Need {:.1f}% more utilization to reach {:.1f}%".format(improvement_needed * 100,
                                                                                            target_ratio * 100))

    print("=" * 50)

    env.close()
    return mean_ratio, mean_reward




