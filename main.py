import sys
import os
import time
from collections import deque
import numpy as np
import torch
from shutil import copyfile
from acktr import algo, utils
from acktr.utils import get_possible_position, get_rotation_mask
from acktr.envs import make_vec_envs
from acktr.arguments import get_args
from acktr.model import Policy
from acktr.storage import RolloutStorage
from evaluation import evaluate
from tensorboardX import SummaryWriter
from unified_test import unified_test
from gym.envs.registration import register

# Register custom environments in both gym and gymnasium
import register_env
from acktr.error_handler import (
    MultiProcessLogger, DeviceErrorDetector, ErrorCheckpoint,
    GracefulShutdown, handle_training_error
)

# def main(args):
#     # input arguments about environment
#     if args.test:
#         test_model(args)
#     else:
#         train_model(args)
def main(args):
    # 修改判断逻辑
    if hasattr(args, 'load_model') and args.load_model:
        print("=== 进入测试模式 ===")
        # 移除强制单进程限制，允许使用用户指定的进程数
        test_model(args)
    else:
        print("=== 进入训练模式 ===")
        train_model(args)

def test_model(args):
    # assert args.test is True
    assert hasattr(args, 'load_model') and args.load_model  # 修改后
    
    # Select environment based on reliability feature flags
    env_name = select_environment(args)
    print(f"Testing with environment: {env_name}")
    
    # Log which reliability features are enabled for testing
    if env_name == 'BppReliable-v0':
        print("Reliability features enabled for testing:")
        if args.uncertainty_enabled:
            print(f"  - Uncertainty simulation (std: {args.uncertainty_std})")
        if args.visual_feedback_enabled:
            print(f"  - Visual feedback module")
        if args.parallel_motion_enabled:
            print(f"  - Parallel entry motion (buffer: {args.buffer_range})")
    
    # Update args.env_name to use selected environment
    args.env_name = env_name
    
    model_url = args.load_dir + args.load_name
    unified_test(model_url, args)

def select_environment(args):
    """
    Select the appropriate environment based on configuration flags.
    
    Returns the environment name to use based on whether reliability features
    are enabled. This ensures backward compatibility while supporting the
    enhanced ReliablePackingGame when needed.
    
    Args:
        args: Command-line arguments containing feature flags
        
    Returns:
        str: Environment name ('Bpp-v0' or 'BppReliable-v0')
    """
    # Use ReliablePackingGame if any reliability feature is enabled
    if (args.uncertainty_enabled or 
        args.visual_feedback_enabled or 
        args.parallel_motion_enabled):
        return 'BppReliable-v0'
    
    # Otherwise use original PackingGame for backward compatibility
    return 'Bpp-v0'

def train_model(args):
    # Requirement 8.1: Initialize multi-process logger with process ID tracking
    logger = MultiProcessLogger(name='training')
    logger.info("Starting training initialization...")
    
    custom = input('please input the test name: ')
    time_now = time.strftime('%Y.%m.%d-%H-%M', time.localtime(time.time()))

    # Select environment based on reliability feature flags
    env_name = select_environment(args)
    logger.info(f"Using environment: {env_name}")
    
    # Log which reliability features are enabled
    if env_name == 'BppReliable-v0':
        logger.info("Reliability features enabled:")
        if args.uncertainty_enabled:
            logger.info(f"  - Uncertainty simulation (std: {args.uncertainty_std})")
        if args.visual_feedback_enabled:
            logger.info(f"  - Visual feedback module")
        if args.parallel_motion_enabled:
            logger.info(f"  - Parallel entry motion (buffer: {args.buffer_range})")
    
    # Requirement 8.5: Check device compatibility before starting
    if args.device != 'cpu':
        is_compatible, error_msg = DeviceErrorDetector.check_device_compatibility(args.device)
        if not is_compatible:
            logger.critical(f"Device compatibility check failed:\n{error_msg}")
            sys.exit(1)
        elif error_msg:  # Warning message
            logger.warning(error_msg)
        
        try:
            torch.cuda.set_device(torch.device(args.device))
            logger.info(f"Using device: {args.device}")
            logger.info(f"GPU: {torch.cuda.get_device_name(args.device)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(args.device).total_memory / 1e9:.2f} GB")
        except Exception as e:
            logger.critical(f"Failed to set CUDA device: {e}")
            sys.exit(1)
    else:
        logger.info("Using CPU for training")
    
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    save_path = args.save_dir
    load_path = args.load_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    data_path = os.path.join(save_path, custom)
    try:
        os.makedirs(data_path)
    except OSError:
        pass

    log_dir = './log'  # directory to save agent logs (default: ./log)
    log_dir = os.path.expanduser(log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device(args.device)
    
    # Requirement 8.5: Initialize error handling components
    checkpoint_handler = ErrorCheckpoint(save_dir=os.path.join(data_path, 'emergency_checkpoints'), logger=logger)
    shutdown_handler = GracefulShutdown(logger=logger)
    
    # Create environments with error handling
    try:
        logger.info(f"Creating {args.num_processes} parallel environments...")
        envs = make_vec_envs(env_name, args.seed, args.num_processes, args.gamma, log_dir, device, False, args = args)
        logger.info(f"Successfully created {args.num_processes} environments")
    except Exception as e:
        logger.critical(f"Failed to create environments: {e}", exc_info=True)
        sys.exit(1)

    if args.pretrain:
        checkpoint = torch.load(os.path.join(load_path, args.load_name))
        model_pretrained = checkpoint['model_state_dict']
        ob_rms = checkpoint['ob_rms']
        actor_critic = Policy(
            envs.observation_space.shape, envs.action_space,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size, 'args': args})
        load_dict = {k.replace('module.', ''): v for k, v in model_pretrained.items()}
        load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
        load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
        for k, v in load_dict.items():
            if len(v.size()) <= 3:
                load_dict[k] = v.squeeze(dim=-1)
        actor_critic.load_state_dict(load_dict)
        setattr(utils.get_vec_normalize(envs), 'ob_rms', ob_rms)
    else:
        actor_critic = Policy(
            envs.observation_space.shape, envs.action_space,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size,'args': args})
    print(actor_critic)
    print("Rotation:", args.enable_rotation)
    actor_critic.to(device)

    # leave a backup for parameter tuning
    copyfile('main.py', os.path.join(data_path, 'main.py'))
    copyfile('./acktr/envs.py', os.path.join(data_path, 'envs.py'))
    copyfile('./acktr/distributions.py', os.path.join(data_path, 'distributions.py'))
    copyfile('./acktr/storage.py', os.path.join(data_path, 'storage.py'))
    copyfile('./acktr/model.py', os.path.join(data_path, 'model.py'))
    copyfile('./acktr/algo/acktr_pipeline.py', os.path.join(data_path, 'acktr_pipeline.py'))

    if args.algorithm == 'a2c':
        agent = algo.ACKTR(actor_critic,
                       args.value_loss_coef,
                       args.entropy_coef,
                       args.invalid_coef,
                       args.lr,
                       args.eps,
                       args.alpha,
                       max_grad_norm = 0.5
                           )
    elif args.algorithm == 'acktr':
        agent = algo.ACKTR(actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            args.invalid_coef,
            acktr=True,
            args=args)

    rollouts = RolloutStorage(args.num_steps,  # forward steps
                              args.num_processes,  # agent processes
                              envs.observation_space.shape,
                              envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              can_give_up=False,
                              enable_rotation=args.enable_rotation,
                              pallet_size=args.container_size[0])

    obs = envs.reset()
    location_masks = []
    for observation in obs:
        if not args.enable_rotation:
            box_mask = get_possible_position(observation, args.container_size)
        else:
            box_mask = get_rotation_mask(observation, args.container_size)
        location_masks.append(box_mask)
    location_masks = torch.FloatTensor(location_masks).to(device)

    rollouts.obs[0].copy_(obs)
    rollouts.location_masks[0].copy_(location_masks)
    rollouts.to(device)

    # Multi-process statistics tracking (Requirement 5.1, 5.2)
    episode_rewards = deque(maxlen=10)
    episode_ratio = deque(maxlen=10)
    
    # Per-process episode tracking for detailed statistics
    process_episode_rewards = [[] for _ in range(args.num_processes)]
    process_episode_ratios = [[] for _ in range(args.num_processes)]
    
    # Initialize reliability feature tracking across all processes (Requirement 5.5)
    if env_name == 'BppReliable-v0':
        # Track reliability features aggregated across all processes
        noise_applied_count = [0] * args.num_processes
        visual_feedback_updates = [0] * args.num_processes
        motion_options_used = [0] * args.num_processes

    start = time.time()

    tbx_dir = './runs'
    if not os.path.exists('{}/{}/{}'.format(tbx_dir, env_name, custom)):
        os.makedirs('{}/{}/{}'.format(tbx_dir, env_name, custom))
    if args.tensorboard:
        writer = SummaryWriter(logdir='{}/{}/{}'.format(tbx_dir, env_name, custom))

    j = 0
    index = 0
    
    # Requirement 8.5: Main training loop with error handling
    logger.info("Starting main training loop...")
    
    try:
        while not shutdown_handler.is_shutdown_requested():
            j += 1
            
            # Check for shutdown request periodically
            if shutdown_handler.is_shutdown_requested():
                logger.info("Shutdown requested, exiting training loop...")
                break
            
            for step in range(args.num_steps):
                # Sample actions with error handling
                try:
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step], location_masks)
                except Exception as e:
                    logger.error(f"Error during action sampling at step {step}: {e}", exc_info=True)
                    # Try to recover or request shutdown
                    if not handle_training_error(e, actor_critic, agent.optimizer, j, envs, 
                                                checkpoint_handler, shutdown_handler, logger):
                        break

                try:
                    obs, reward, terminated, truncated, infos = envs.step(action)
                    done = np.logical_or(terminated, truncated)
                except Exception as e:
                    logger.error(f"Error during environment step at step {step}: {e}", exc_info=True)
                    # Try to recover or request shutdown
                    infos = [{} for _ in range(args.num_processes)]  # Initialize with empty dicts for each process
                    done = np.ones(args.num_processes, dtype=bool)  # Assume all processes are done on error
                    if not handle_training_error(e, actor_critic, agent.optimizer, j, envs,
                                                checkpoint_handler, shutdown_handler, logger):
                        break
                
                # Aggregate episode statistics from all processes (Requirement 5.1)
                for i in range(len(infos)):
                    if 'episode' in infos[i].keys():
                        # Collect rewards from all processes
                        episode_rewards.append(infos[i]['episode']['r'])
                        episode_ratio.append(infos[i]['ratio'])
                        
                        # Track per-process statistics for detailed analysis
                        process_episode_rewards[i].append(infos[i]['episode']['r'])
                        process_episode_ratios[i].append(infos[i]['ratio'])
                    
                    # Track reliability features usage across all processes (Requirement 5.5)
                    if env_name == 'BppReliable-v0':
                        if 'noise_applied' in infos[i]:
                            noise_applied_count[i] += 1
                        if 'visual_feedback_update' in infos[i]:
                            visual_feedback_updates[i] += 1
                        if 'motion_option_used' in infos[i]:
                            motion_options_used[i] += 1
                
                # Compute location masks for next step
                location_masks = []
                for observation in obs:
                    if not args.enable_rotation:
                        box_mask = get_possible_position(observation, args.container_size)
                    else:
                        box_mask = get_rotation_mask(observation, args.container_size)
                    location_masks.append(box_mask)
                location_masks = torch.FloatTensor(location_masks).to(device)

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks, location_masks)

            # Compute returns and update model with error handling
            try:
                with torch.no_grad():
                    next_value = actor_critic.get_value(
                        rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1]).detach()

                rollouts.compute_returns(next_value, False, args.gamma, 0.95, False)
                value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
                rollouts.after_update()
                
            except RuntimeError as e:
                # Handle CUDA out of memory and other runtime errors
                if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                    logger.critical(
                        f"CUDA error at update {j}: {e}\n"
                        f"Suggestions:\n"
                        f"  1. Reduce --num-processes (current: {args.num_processes})\n"
                        f"  2. Reduce --num-steps (current: {args.num_steps})\n"
                        f"  3. Use --device cpu\n"
                        f"  4. Close other GPU-using applications"
                    )
                else:
                    logger.error(f"Runtime error during update at step {j}: {e}", exc_info=True)
                
                # Save emergency checkpoint and request shutdown
                if not handle_training_error(e, actor_critic, agent.optimizer, j, envs,
                                            checkpoint_handler, shutdown_handler, logger):
                    break
            except Exception as e:
                logger.error(f"Unexpected error during update at step {j}: {e}", exc_info=True)
                if not handle_training_error(e, actor_critic, agent.optimizer, j, envs,
                                            checkpoint_handler, shutdown_handler, logger):
                    break
            # Requirement 8.5: Save checkpoints with error handling
            if args.save_model:
                if (j % args.save_interval == 0) and args.save_dir != "":
                    try:
                        # Save model with reliability configuration
                        save_dict = {
                            'model_state_dict': actor_critic.state_dict(),
                            'ob_rms': getattr(utils.get_vec_normalize(envs), 'ob_rms', None),
                            'reliability_config': {
                                'uncertainty_enabled': args.uncertainty_enabled,
                                'uncertainty_std': args.uncertainty_std,
                                'visual_feedback_enabled': args.visual_feedback_enabled,
                                'parallel_motion_enabled': args.parallel_motion_enabled,
                                'buffer_range': args.buffer_range,
                                'env_name': env_name
                            },
                            'training_step': j,
                            'optimizer_stats': agent.optimizer.get_statistics() if hasattr(agent.optimizer, 'get_statistics') else {}
                        }
                        checkpoint_path = os.path.join(data_path, env_name + time_now + ".pt")
                        torch.save(save_dict, checkpoint_path)
                        logger.info(f"Checkpoint saved to: {checkpoint_path}")
                        
                        # Also save in legacy format for backward compatibility
                        legacy_path = os.path.join(data_path, env_name + time_now + "_legacy.pt")
                        torch.save([
                            actor_critic.state_dict(),
                            getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                        ], legacy_path)
                        logger.debug(f"Legacy checkpoint saved to: {legacy_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to save checkpoint at step {j}: {e}", exc_info=True)

            # Print useful information of training with multi-process statistics (Requirement 5.2, 5.3, 5.4)
            # Requirement 8.1: Use logger with process ID
            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                end = time.time()
                index += 1
                
                # Compute statistics aggregated from all processes (Requirement 5.2)
                mean_reward = np.mean(episode_rewards)
                median_reward = np.median(episode_rewards)
                min_reward = np.min(episode_rewards)
                max_reward = np.max(episode_rewards)
                mean_ratio = np.mean(episode_ratio)
                median_ratio = np.median(episode_ratio)
                min_ratio = np.min(episode_ratio)
                max_ratio = np.max(episode_ratio)
                
                logger.info(
                    f"Algorithm: {args.algorithm}, Recurrent: False\n"
                    f"Environment: {env_name}, Version: {custom}"
                )
                logger.info(
                    f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}\n"
                    f"Last {len(episode_rewards)} training episodes (aggregated from {args.num_processes} processes):\n"
                    f"  Reward - mean/median: {mean_reward:.1f}/{median_reward:.1f}, min/max: {min_reward:.1f}/{max_reward:.1f}\n"
                    f"  Space ratio - mean/median: {mean_ratio:.3f}/{median_ratio:.3f}, min/max: {min_ratio:.3f}/{max_ratio:.3f}\n"
                    f"Loss - entropy: {dist_entropy:.5f}, value: {value_loss:.5f}, action: {action_loss:.5f}"
                )
            
                # Log per-process statistics for debugging (Requirement 5.1)
                active_processes = sum(1 for p in process_episode_rewards if len(p) > 0)
                if active_processes > 0:
                    logger.info(f"Active processes: {active_processes}/{args.num_processes}")
                
                # Log optimizer statistics if available (Requirement 8.4)
                if hasattr(agent.optimizer, 'get_statistics'):
                    opt_stats = agent.optimizer.get_statistics()
                    if opt_stats.get('nan_inf_count', 0) > 0 or opt_stats.get('device_mismatch_count', 0) > 0:
                        logger.warning(
                            f"Optimizer statistics:\n"
                            f"  NaN/Inf fallbacks: {opt_stats.get('nan_inf_count', 0)}\n"
                            f"  Device mismatches: {opt_stats.get('device_mismatch_count', 0)}\n"
                            f"  Total NaN detections: {opt_stats.get('total_nan_detections', 0)}\n"
                            f"  Total Inf detections: {opt_stats.get('total_inf_detections', 0)}"
                        )
            
                # Log reliability features usage aggregated across all processes (Requirement 5.5)
                if env_name == 'BppReliable-v0':
                    total_noise = sum(noise_applied_count)
                    total_visual = sum(visual_feedback_updates)
                    total_motion = sum(motion_options_used)
                    
                    logger.info("Reliability features usage (aggregated across all processes):")
                    if args.uncertainty_enabled:
                        logger.info(f"  - Noise applied: {total_noise} times (avg {total_noise/args.num_processes:.1f} per process)")
                    if args.visual_feedback_enabled:
                        logger.info(f"  - Visual feedback updates: {total_visual} times (avg {total_visual/args.num_processes:.1f} per process)")
                    if args.parallel_motion_enabled:
                        logger.info(f"  - Motion options used: {total_motion} times (avg {total_motion/args.num_processes:.1f} per process)")

                # TensorBoard logging with multi-process aggregated metrics (Requirement 5.3)
                if args.tensorboard:
                    # Log aggregated reward statistics (Requirement 5.2)
                    writer.add_scalar('Rewards/Mean', mean_reward, j)
                    writer.add_scalar('Rewards/Median', median_reward, j)
                    writer.add_scalar('Rewards/Min', min_reward, j)
                    writer.add_scalar('Rewards/Max', max_reward, j)
                    
                    # Log aggregated space utilization statistics (Requirement 5.4)
                    writer.add_scalar('SpaceRatio/Mean', mean_ratio, j)
                    writer.add_scalar('SpaceRatio/Median', median_ratio, j)
                    writer.add_scalar('SpaceRatio/Min', min_ratio, j)
                    writer.add_scalar('SpaceRatio/Max', max_ratio, j)
                    
                    # Log loss components
                    writer.add_scalar('Loss/Entropy', dist_entropy, j)
                    writer.add_scalar('Loss/Value', value_loss, j)
                    writer.add_scalar('Loss/Action', action_loss, j)
                    writer.add_scalar('Loss/Probability', prob_loss, j)
                    writer.add_scalar('Loss/Mask', graph_loss, j)
                    
                    # Log multi-process training metrics
                    writer.add_scalar('Training/FPS', int(total_num_steps / (end - start)), j)
                    writer.add_scalar('Training/TotalSteps', total_num_steps, j)
                    writer.add_scalar('Training/ActiveProcesses', active_processes, j)
                    
                    # Log reliability features aggregated across all processes (Requirement 5.5)
                    if env_name == 'BppReliable-v0':
                        if args.uncertainty_enabled:
                            writer.add_scalar('Reliability/Noise_Applied_Total', total_noise, j)
                            writer.add_scalar('Reliability/Noise_Applied_PerProcess', total_noise/args.num_processes, j)
                        if args.visual_feedback_enabled:
                            writer.add_scalar('Reliability/Visual_Feedback_Total', total_visual, j)
                            writer.add_scalar('Reliability/Visual_Feedback_PerProcess', total_visual/args.num_processes, j)
                        if args.parallel_motion_enabled:
                            writer.add_scalar('Reliability/Motion_Options_Total', total_motion, j)
                            writer.add_scalar('Reliability/Motion_Options_PerProcess', total_motion/args.num_processes, j)
    
    except KeyboardInterrupt:
        # Requirement 8.5: Handle keyboard interrupt gracefully
        logger.info("Training interrupted by user (Ctrl+C)")
        logger.info("Saving emergency checkpoint...")
        checkpoint_handler.save_emergency_checkpoint(
            actor_critic, agent.optimizer, j,
            {'error_type': 'KeyboardInterrupt', 'error_message': 'User interrupted training'},
            envs
        )
        shutdown_handler.request_shutdown("User interrupt")
        shutdown_handler.cleanup_processes(envs)
        logger.info("Graceful shutdown complete")
        
    except Exception as e:
        # Requirement 8.5: Handle unexpected errors
        logger.critical(f"Unexpected error in training loop: {e}", exc_info=True)
        handle_training_error(e, actor_critic, agent.optimizer, j, envs,
                            checkpoint_handler, shutdown_handler, logger)
        shutdown_handler.cleanup_processes(envs)
        raise
    
    finally:
        # Requirement 8.5: Ensure cleanup always happens
        logger.info("Cleaning up resources...")
        try:
            if 'envs' in locals():
                envs.close()
            if args.tensorboard and 'writer' in locals():
                writer.close()
            logger.info("Training session ended")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)


def registration_envs():
    """
    Register all available packing environments.
    
    Registers both the original PackingGame and the enhanced ReliablePackingGame.
    The environment selection is handled through configuration flags in args.
    """
    # Register original PackingGame environment
    register(
        id='Bpp-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='envs.bpp0:PackingGame',   # Explained in envs/__init__.py
    )
    
    # Register ReliablePackingGame environment with reliability features
    register(
        id='BppReliable-v0',                          # Enhanced environment with reliability features
        entry_point='envs.bpp0:ReliablePackingGame',  # Reliable packing environment
    )


if __name__ == "__main__":
    registration_envs()
    args = get_args()
    main(args)
