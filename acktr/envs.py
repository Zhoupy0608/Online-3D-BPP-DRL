import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.box import Box
from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
import sys
sys.path.append('../')

from acktr.error_handler import MultiProcessLogger

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

def make_env(env_id, seed, rank, log_dir, allow_early_resets, args):
    def _thunk():
        # Register environments in subprocess (required for Windows spawn mode)
        # This ensures each subprocess has access to the custom environments
        from gym.envs.registration import register
        
        # Check if environment is already registered to avoid duplicate registration
        try:
            gym.envs.registry.spec('Bpp-v0')
        except:
            # Register original PackingGame environment
            register(
                id='Bpp-v0',
                entry_point='envs.bpp0:PackingGame',
            )
        
        try:
            gym.envs.registry.spec('BppReliable-v0')
        except:
            # Register ReliablePackingGame environment
            register(
                id='BppReliable-v0',
                entry_point='envs.bpp0:ReliablePackingGame',
            )
        
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            # Base parameters for all environments
            env_kwargs = {
                'enable_rotation': args.enable_rotation,
                'box_set': args.box_size_set,
                'container_size': args.container_size,
                'test': False,
                'data_name': None,
                'data_type': args.data_type
            }
            
            # Add reliability parameters if using ReliablePackingGame
            if env_id == 'BppReliable-v0':
                env_kwargs.update({
                    'uncertainty_enabled': args.uncertainty_enabled,
                    'visual_feedback_enabled': args.visual_feedback_enabled,
                    'parallel_motion_enabled': args.parallel_motion_enabled,
                    'noise_std': args.uncertainty_std,
                    'buffer_range': args.buffer_range,
                    'camera_config': args.camera_config
                })
            
            env = gym.make(env_id, **env_kwargs)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        # if is_atari:
        #     env = make_atari(env_id)
        
        # Assign unique seed to each process: base_seed + rank
        # Gymnasium doesn't have seed() method, seed in reset instead
        env.reset(seed=seed + rank)

        obs_shape = env.observation_space.shape

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        # if is_atari:
        #     if len(env.observation_space.shape) == 3:
        #         env = wrap_deepmind(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env
    return _thunk

def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack = None,
                  args = None):
    """
    Create vectorized environments with proper multiprocessing context.
    
    Args:
        env_name: Environment identifier ('Bpp-v0' or 'BppReliable-v0')
        seed: Base random seed
        num_processes: Number of parallel processes (default: 16)
        gamma: Discount factor
        log_dir: Directory for logging
        device: Torch device (CPU or CUDA)
        allow_early_resets: Whether to allow early episode resets
        num_frame_stack: Number of frames to stack (optional)
        args: Additional arguments including reliability features
        
    Returns:
        VecPyTorch: Vectorized environment wrapper
    """
    # Requirement 8.1: Initialize logger with process ID tracking
    logger = MultiProcessLogger(name='envs')
    
    # Create environment factory functions for each process
    # Each process gets unique seed: base_seed + rank
    logger.info(f"Creating {num_processes} environment factory functions...")
    try:
        envs = [
            make_env(env_name, seed, i, log_dir, allow_early_resets, args)
            for i in range(num_processes)
        ]
        logger.info(f"Successfully created {num_processes} environment factories")
    except Exception as e:
        logger.error(f"Failed to create environment factories: {e}", exc_info=True)
        raise

    if len(envs) > 1:
        # Multi-process mode: use ShmemVecEnv for efficient shared memory communication
        # Requirement 8.1, 8.5: Error handling for environment creation
        logger.info("Multi-process mode: Creating ShmemVecEnv...")
        
        # Create a dummy environment to get observation and action spaces
        # Base parameters for dummy environment
        env_kwargs = {
            'enable_rotation': args.enable_rotation,
            'box_set': args.box_size_set,
            'container_size': args.container_size,
            'test': False,
            'data_name': None,
            'data_type': args.data_type
        }
        
        # Add reliability parameters if using ReliablePackingGame
        if env_name == 'BppReliable-v0':
            env_kwargs.update({
                'uncertainty_enabled': args.uncertainty_enabled,
                'visual_feedback_enabled': args.visual_feedback_enabled,
                'parallel_motion_enabled': args.parallel_motion_enabled,
                'noise_std': args.uncertainty_std,
                'buffer_range': args.buffer_range,
                'camera_config': args.camera_config
            })
        
        # Create dummy environment to get spaces
        try:
            logger.debug("Creating dummy environment to get observation/action spaces...")
            env = gym.make(env_name, **env_kwargs)
            spaces = [env.observation_space, env.action_space]
            env.close()
            del env
            logger.debug("Dummy environment created and closed successfully")
        except Exception as e:
            logger.error(f"Failed to create dummy environment: {e}", exc_info=True)
            raise
        
        # Platform-specific multiprocessing context
        # Windows requires 'spawn' context, Linux can use 'fork' (faster)
        # Requirement 3.1, 3.2: Platform detection
        import sys
        if sys.platform == 'win32':
            context = 'spawn'
            logger.info("Platform: Windows - using 'spawn' multiprocessing context")
        else:
            context = 'fork'
            logger.info("Platform: Linux/Unix - using 'fork' multiprocessing context")
        
        # Requirement 8.5: Try to create ShmemVecEnv with error handling
        try:
            logger.info(f"Creating ShmemVecEnv with {num_processes} processes...")
            envs = ShmemVecEnv(envs, spaces, context=context)
            logger.info(f"Successfully created ShmemVecEnv with {num_processes} processes")
        except Exception as e:
            logger.warning(
                f"Failed to create ShmemVecEnv with context '{context}': {e}\n"
                f"Falling back to DummyVecEnv (single-process mode)\n"
                f"This will significantly reduce training speed.\n"
                f"Possible causes:\n"
                f"  1. Insufficient shared memory\n"
                f"  2. Platform-specific multiprocessing issues\n"
                f"  3. Environment serialization problems"
            )
            logger.info("Creating DummyVecEnv as fallback...")
            envs = DummyVecEnv(envs)
            logger.info("DummyVecEnv created successfully")
    else:
        # Single-process mode: use DummyVecEnv
        logger.info("Single-process mode: Creating DummyVecEnv...")
        try:
            envs = DummyVecEnv(envs)
            logger.info("DummyVecEnv created successfully")
        except Exception as e:
            logger.error(f"Failed to create DummyVecEnv: {e}", exc_info=True)
            raise

    # Apply observation normalization if needed
    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma = gamma, ob = False, ret = False)
    
    # Wrap with PyTorch tensor conversion
    envs = VecPyTorch(envs, device)
    
    # Apply frame stacking if needed
    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)
    
    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if done and hasattr(self.env, '_max_episode_steps') and hasattr(self.env, '_elapsed_steps') and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        # Handle Gymnasium's (obs, reward, terminated, truncated, info) format
        obs, reward, terminated, truncated, info = self.venv.step_wait()
        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, terminated, truncated, info

class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, terminated, truncated, infos = self.venv.step_wait()
        news = np.logical_or(terminated, truncated)
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, terminated, truncated, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
