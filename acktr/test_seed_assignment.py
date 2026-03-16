"""
Property-based tests for seed assignment in multi-process training

These tests verify that each environment process receives a unique seed
to ensure different random sequences across parallel environments.
"""

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from unittest.mock import Mock, patch, MagicMock
import gym

from acktr.envs import make_env, make_vec_envs


# Hypothesis strategies for generating test data
@st.composite
def num_processes_strategy(draw):
    """Generate valid number of processes (1-32)."""
    return draw(st.integers(min_value=1, max_value=32))


@st.composite
def base_seed_strategy(draw):
    """Generate valid base seeds."""
    return draw(st.integers(min_value=0, max_value=100000))


# Property 2: Unique seeds across processes
# Feature: multi-process-training, Property 2: Unique seeds across processes
# Validates: Requirements 4.1
@settings(max_examples=100, deadline=None)
@given(
    base_seed=base_seed_strategy(),
    num_processes=num_processes_strategy()
)
def test_unique_seeds_across_processes(base_seed, num_processes):
    """
    Property 2: For any num_processes value, each environment process 
    should receive a unique seed equal to base_seed + process_rank, 
    ensuring different random sequences.
    
    Validates: Requirements 4.1
    """
    # Track seeds that were set on environments
    seeds_set = []
    
    # Create a mock args object with minimal required attributes
    mock_args = Mock()
    mock_args.enable_rotation = False
    mock_args.box_size_set = [(2, 2, 2), (3, 3, 3)]
    mock_args.container_size = (10, 10, 10)
    mock_args.data_type = 'cut2'
    mock_args.uncertainty_enabled = False
    mock_args.visual_feedback_enabled = False
    mock_args.parallel_motion_enabled = False
    mock_args.uncertainty_std = (0.5, 0.5, 0.1)
    mock_args.buffer_range = (1, 1)
    mock_args.camera_config = None
    
    # Mock gym.make to capture seed calls
    original_gym_make = gym.make
    
    def mock_gym_make(env_id, **kwargs):
        """Mock gym.make to return a mock environment that tracks seed calls."""
        mock_env = MagicMock()
        # Use 1D observation space to avoid the CNN check in make_env
        mock_env.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(400,), dtype=np.float32
        )
        mock_env.action_space = gym.spaces.Discrete(100)
        mock_env._max_episode_steps = 100
        mock_env._elapsed_steps = 0
        mock_env.__class__.__name__ = 'PackingGame'
        
        # Track seed calls
        def track_seed(seed):
            seeds_set.append(seed)
            return [seed]
        
        mock_env.seed = track_seed
        return mock_env
    
    # Patch gym.make
    with patch('gym.make', side_effect=mock_gym_make):
        # Create environment factory functions for each process
        env_fns = [
            make_env('Bpp-v0', base_seed, rank, None, False, mock_args)
            for rank in range(num_processes)
        ]
        
        # Call each environment factory to trigger seed assignment
        for env_fn in env_fns:
            env = env_fn()
            # Clean up
            if hasattr(env, 'close'):
                try:
                    env.close()
                except:
                    pass
    
    # Verify we collected the expected number of seeds
    assert len(seeds_set) == num_processes, \
        f"Expected {num_processes} seeds, but got {len(seeds_set)}"
    
    # Property 1: All seeds should be unique
    unique_seeds = set(seeds_set)
    assert len(unique_seeds) == num_processes, \
        f"Seeds are not unique: {seeds_set}. Found {len(unique_seeds)} unique seeds out of {num_processes}"
    
    # Property 2: Each seed should equal base_seed + rank
    expected_seeds = [base_seed + rank for rank in range(num_processes)]
    assert sorted(seeds_set) == sorted(expected_seeds), \
        f"Seeds {sorted(seeds_set)} do not match expected {sorted(expected_seeds)}"
    
    # Property 3: Seeds should be in sequential order (base_seed, base_seed+1, ..., base_seed+n-1)
    for rank in range(num_processes):
        expected_seed = base_seed + rank
        assert expected_seed in seeds_set, \
            f"Expected seed {expected_seed} for rank {rank} not found in {seeds_set}"


# Unit test: Verify seed assignment with small number of processes
def test_seed_assignment_small_processes():
    """Unit test to verify seed assignment with 4 processes."""
    base_seed = 42
    num_processes = 4
    
    seeds_set = []
    
    mock_args = Mock()
    mock_args.enable_rotation = False
    mock_args.box_size_set = [(2, 2, 2)]
    mock_args.container_size = (10, 10, 10)
    mock_args.data_type = 'cut2'
    mock_args.uncertainty_enabled = False
    mock_args.visual_feedback_enabled = False
    mock_args.parallel_motion_enabled = False
    mock_args.uncertainty_std = (0.5, 0.5, 0.1)
    mock_args.buffer_range = (1, 1)
    mock_args.camera_config = None
    
    def mock_gym_make(env_id, **kwargs):
        mock_env = MagicMock()
        # Use 1D observation space to avoid the CNN check in make_env
        mock_env.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(400,), dtype=np.float32
        )
        mock_env.action_space = gym.spaces.Discrete(100)
        mock_env._max_episode_steps = 100
        mock_env._elapsed_steps = 0
        mock_env.__class__.__name__ = 'PackingGame'
        
        def track_seed(seed):
            seeds_set.append(seed)
            return [seed]
        
        mock_env.seed = track_seed
        return mock_env
    
    with patch('gym.make', side_effect=mock_gym_make):
        env_fns = [
            make_env('Bpp-v0', base_seed, rank, None, False, mock_args)
            for rank in range(num_processes)
        ]
        
        for env_fn in env_fns:
            env = env_fn()
            if hasattr(env, 'close'):
                try:
                    env.close()
                except:
                    pass
    
    # Verify seeds are [42, 43, 44, 45]
    assert seeds_set == [42, 43, 44, 45], \
        f"Expected seeds [42, 43, 44, 45], got {seeds_set}"


# Unit test: Verify single process gets base seed
def test_seed_assignment_single_process():
    """Unit test to verify seed assignment with 1 process."""
    base_seed = 100
    
    seeds_set = []
    
    mock_args = Mock()
    mock_args.enable_rotation = False
    mock_args.box_size_set = [(2, 2, 2)]
    mock_args.container_size = (10, 10, 10)
    mock_args.data_type = 'cut2'
    mock_args.uncertainty_enabled = False
    mock_args.visual_feedback_enabled = False
    mock_args.parallel_motion_enabled = False
    mock_args.uncertainty_std = (0.5, 0.5, 0.1)
    mock_args.buffer_range = (1, 1)
    mock_args.camera_config = None
    
    def mock_gym_make(env_id, **kwargs):
        mock_env = MagicMock()
        # Use 1D observation space to avoid the CNN check in make_env
        mock_env.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(400,), dtype=np.float32
        )
        mock_env.action_space = gym.spaces.Discrete(100)
        mock_env._max_episode_steps = 100
        mock_env._elapsed_steps = 0
        mock_env.__class__.__name__ = 'PackingGame'
        
        def track_seed(seed):
            seeds_set.append(seed)
            return [seed]
        
        mock_env.seed = track_seed
        return mock_env
    
    with patch('gym.make', side_effect=mock_gym_make):
        env_fn = make_env('Bpp-v0', base_seed, 0, None, False, mock_args)
        env = env_fn()
        if hasattr(env, 'close'):
            try:
                env.close()
            except:
                pass
    
    # Verify seed is exactly base_seed (100)
    assert seeds_set == [100], \
        f"Expected seed [100], got {seeds_set}"


# Unit test: Verify different base seeds produce different sequences
def test_different_base_seeds_produce_different_sequences():
    """Unit test to verify different base seeds result in different seed sequences."""
    num_processes = 4
    
    # Test with base_seed = 10
    seeds_set_1 = []
    mock_args = Mock()
    mock_args.enable_rotation = False
    mock_args.box_size_set = [(2, 2, 2)]
    mock_args.container_size = (10, 10, 10)
    mock_args.data_type = 'cut2'
    mock_args.uncertainty_enabled = False
    mock_args.visual_feedback_enabled = False
    mock_args.parallel_motion_enabled = False
    mock_args.uncertainty_std = (0.5, 0.5, 0.1)
    mock_args.buffer_range = (1, 1)
    mock_args.camera_config = None
    
    def mock_gym_make_1(env_id, **kwargs):
        mock_env = MagicMock()
        # Use 1D observation space to avoid the CNN check in make_env
        mock_env.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(400,), dtype=np.float32
        )
        mock_env.action_space = gym.spaces.Discrete(100)
        mock_env._max_episode_steps = 100
        mock_env._elapsed_steps = 0
        mock_env.__class__.__name__ = 'PackingGame'
        
        def track_seed(seed):
            seeds_set_1.append(seed)
            return [seed]
        
        mock_env.seed = track_seed
        return mock_env
    
    with patch('gym.make', side_effect=mock_gym_make_1):
        env_fns = [
            make_env('Bpp-v0', 10, rank, None, False, mock_args)
            for rank in range(num_processes)
        ]
        for env_fn in env_fns:
            env = env_fn()
            if hasattr(env, 'close'):
                try:
                    env.close()
                except:
                    pass
    
    # Test with base_seed = 50
    seeds_set_2 = []
    
    def mock_gym_make_2(env_id, **kwargs):
        mock_env = MagicMock()
        # Use 1D observation space to avoid the CNN check in make_env
        mock_env.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(400,), dtype=np.float32
        )
        mock_env.action_space = gym.spaces.Discrete(100)
        mock_env._max_episode_steps = 100
        mock_env._elapsed_steps = 0
        mock_env.__class__.__name__ = 'PackingGame'
        
        def track_seed(seed):
            seeds_set_2.append(seed)
            return [seed]
        
        mock_env.seed = track_seed
        return mock_env
    
    with patch('gym.make', side_effect=mock_gym_make_2):
        env_fns = [
            make_env('Bpp-v0', 50, rank, None, False, mock_args)
            for rank in range(num_processes)
        ]
        for env_fn in env_fns:
            env = env_fn()
            if hasattr(env, 'close'):
                try:
                    env.close()
                except:
                    pass
    
    # Verify the two seed sequences are different
    assert seeds_set_1 != seeds_set_2, \
        f"Different base seeds should produce different sequences: {seeds_set_1} vs {seeds_set_2}"
    
    # Verify first sequence is [10, 11, 12, 13]
    assert seeds_set_1 == [10, 11, 12, 13], \
        f"Expected [10, 11, 12, 13], got {seeds_set_1}"
    
    # Verify second sequence is [50, 51, 52, 53]
    assert seeds_set_2 == [50, 51, 52, 53], \
        f"Expected [50, 51, 52, 53], got {seeds_set_2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
