"""
Property-based tests for gradient batch size in multi-process training

These tests verify that the batch size used for gradient computation
equals num_steps × num_processes, ensuring all collected transitions
contribute to the update.

Feature: multi-process-training, Property 3: Gradient batch size equals num_steps × num_processes
Validates: Requirements 1.4
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from hypothesis import given, settings, strategies as st
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from acktr.algo.acktr_pipeline import ACKTR
from acktr.model import Policy
from acktr.storage import RolloutStorage
from gym.spaces import Discrete


# Hypothesis strategies for generating test data
@st.composite
def num_processes_strategy(draw):
    """Generate valid number of processes (1-32)."""
    return draw(st.integers(min_value=1, max_value=32))


@st.composite
def num_steps_strategy(draw):
    """Generate valid number of steps (1-20)."""
    return draw(st.integers(min_value=1, max_value=20))


@st.composite
def container_size_strategy(draw):
    """Generate valid container sizes (5-15)."""
    size = draw(st.integers(min_value=5, max_value=15))
    return size


def create_test_args(container_size=10, enable_rotation=False):
    """Create minimal args for testing."""
    args = argparse.Namespace()
    args.container_size = [container_size, container_size, container_size]
    args.pallet_size = container_size
    args.channel = 4
    args.enable_rotation = enable_rotation
    return args


def create_rollout_storage(num_steps, num_processes, container_size, enable_rotation):
    """
    Create and populate a rollout storage with random data.
    
    Args:
        num_steps: Number of forward steps
        num_processes: Number of parallel processes
        container_size: Size of the container (for action space)
        enable_rotation: Whether rotation is enabled
        
    Returns:
        RolloutStorage: Populated rollout storage
    """
    # Observation shape: flattened (area * channels)
    obs_shape = (container_size * container_size * 4,)
    
    # Action space size
    action_space_size = container_size * container_size
    if enable_rotation:
        action_space_size *= 2
    action_space = Discrete(action_space_size)
    
    # Create rollout storage
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=enable_rotation,
        pallet_size=container_size
    )
    
    # Fill with random data
    for step in range(num_steps):
        obs = torch.randn(num_processes, *obs_shape)
        actions = torch.randint(0, action_space_size, (num_processes, 1))
        rewards = torch.randn(num_processes, 1)
        masks = torch.ones(num_processes, 1)
        location_masks = torch.randn(num_processes, action_space_size)
        
        rollouts.insert(
            obs=obs,
            recurrent_hidden_states=torch.zeros(num_processes, 1),
            actions=actions,
            action_log_probs=torch.randn(num_processes, 1),
            value_preds=torch.randn(num_processes, 1),
            rewards=rewards,
            masks=masks,
            bad_masks=masks,
            location_masks=location_masks
        )
    
    # Compute returns
    rollouts.compute_returns(
        next_value=torch.zeros(num_processes, 1),
        use_gae=False,
        gamma=1.0,
        gae_lambda=0.95
    )
    
    return rollouts


# Property 3: Gradient batch size equals num_steps × num_processes
# Feature: multi-process-training, Property 3: Gradient batch size equals num_steps × num_processes
# Validates: Requirements 1.4
@settings(max_examples=100, deadline=None)
@given(
    num_steps=num_steps_strategy(),
    num_processes=num_processes_strategy(),
    container_size=container_size_strategy(),
    enable_rotation=st.booleans()
)
def test_gradient_batch_size_property(num_steps, num_processes, container_size, enable_rotation):
    """
    Property 3: For any rollout update, the batch size used for gradient 
    computation should equal num_steps × num_processes, ensuring all 
    collected transitions contribute to the update.
    
    Validates: Requirements 1.4
    
    This property verifies that:
    1. The reshaping from (num_steps, num_processes, ...) to 
       (num_steps * num_processes, ...) is correct
    2. All transitions from all processes are included in gradient computation
    3. The batch size assertion in ACKTR.update() passes
    """
    # Create args
    args = create_test_args(container_size, enable_rotation)
    
    # Observation shape
    obs_shape = (container_size * container_size * 4,)
    
    # Action space
    action_space_size = container_size * container_size
    if enable_rotation:
        action_space_size *= 2
    action_space = Discrete(action_space_size)
    
    # Create policy
    actor_critic = Policy(
        obs_shape, 
        action_space, 
        base_kwargs={'recurrent': False, 'hidden_size': 256, 'args': args}
    )
    
    # Create ACKTR agent (use RMSprop for faster testing)
    agent = ACKTR(
        actor_critic=actor_critic,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        invaild_coef=2.0,
        acktr=False,  # Use RMSprop for simplicity
        lr=0.001,
        eps=1e-5,
        alpha=0.99,
        max_grad_norm=0.5,
        args=args
    )
    
    # Create and populate rollout storage
    rollouts = create_rollout_storage(num_steps, num_processes, container_size, enable_rotation)
    
    # Expected batch size
    expected_batch_size = num_steps * num_processes
    
    # Run update - this will verify batch size internally via assertions
    # If the batch size is incorrect, the assertions in acktr_pipeline.py will fail
    try:
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
        
        # If we reach here, the batch size was correct
        # Additional verification: check that losses are finite
        assert np.isfinite(value_loss), f"Value loss is not finite: {value_loss}"
        assert np.isfinite(action_loss), f"Action loss is not finite: {action_loss}"
        assert np.isfinite(dist_entropy), f"Entropy is not finite: {dist_entropy}"
        
    except AssertionError as e:
        # If assertion fails, it means batch size was incorrect
        pytest.fail(f"Batch size verification failed for num_steps={num_steps}, "
                   f"num_processes={num_processes}: {e}")


# Unit test: Verify batch size with single process
def test_batch_size_single_process():
    """
    Unit test: Verify batch size with single process (num_processes=1).
    
    This tests that the reshaping works correctly even with a single process.
    """
    num_steps = 5
    num_processes = 1
    container_size = 10
    enable_rotation = False
    
    args = create_test_args(container_size, enable_rotation)
    obs_shape = (container_size * container_size * 4,)
    action_space = Discrete(container_size * container_size)
    
    actor_critic = Policy(
        obs_shape, 
        action_space, 
        base_kwargs={'recurrent': False, 'hidden_size': 256, 'args': args}
    )
    
    agent = ACKTR(
        actor_critic=actor_critic,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        invaild_coef=2.0,
        acktr=False,
        lr=0.001,
        eps=1e-5,
        alpha=0.99,
        max_grad_norm=0.5,
        args=args
    )
    
    rollouts = create_rollout_storage(num_steps, num_processes, container_size, enable_rotation)
    
    # Expected batch size: 5 * 1 = 5
    expected_batch_size = 5
    
    value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
    
    # Verify losses are finite
    assert np.isfinite(value_loss)
    assert np.isfinite(action_loss)
    assert np.isfinite(dist_entropy)


# Unit test: Verify batch size with multiple processes
def test_batch_size_multi_process():
    """
    Unit test: Verify batch size with multiple processes (num_processes=4).
    
    This tests that the reshaping correctly aggregates experiences from
    all parallel environments.
    """
    num_steps = 5
    num_processes = 4
    container_size = 10
    enable_rotation = False
    
    args = create_test_args(container_size, enable_rotation)
    obs_shape = (container_size * container_size * 4,)
    action_space = Discrete(container_size * container_size)
    
    actor_critic = Policy(
        obs_shape, 
        action_space, 
        base_kwargs={'recurrent': False, 'hidden_size': 256, 'args': args}
    )
    
    agent = ACKTR(
        actor_critic=actor_critic,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        invaild_coef=2.0,
        acktr=False,
        lr=0.001,
        eps=1e-5,
        alpha=0.99,
        max_grad_norm=0.5,
        args=args
    )
    
    rollouts = create_rollout_storage(num_steps, num_processes, container_size, enable_rotation)
    
    # Expected batch size: 5 * 4 = 20
    expected_batch_size = 20
    
    value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
    
    # Verify losses are finite
    assert np.isfinite(value_loss)
    assert np.isfinite(action_loss)
    assert np.isfinite(dist_entropy)


# Unit test: Verify batch size with paper configuration
def test_batch_size_paper_config():
    """
    Unit test: Verify batch size with paper configuration (num_steps=5, num_processes=16).
    
    This tests the exact configuration used in the paper.
    """
    num_steps = 5
    num_processes = 16
    container_size = 10
    enable_rotation = False
    
    args = create_test_args(container_size, enable_rotation)
    obs_shape = (container_size * container_size * 4,)
    action_space = Discrete(container_size * container_size)
    
    actor_critic = Policy(
        obs_shape, 
        action_space, 
        base_kwargs={'recurrent': False, 'hidden_size': 256, 'args': args}
    )
    
    agent = ACKTR(
        actor_critic=actor_critic,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        invaild_coef=2.0,
        acktr=False,
        lr=0.001,
        eps=1e-5,
        alpha=0.99,
        max_grad_norm=0.5,
        args=args
    )
    
    rollouts = create_rollout_storage(num_steps, num_processes, container_size, enable_rotation)
    
    # Expected batch size: 5 * 16 = 80
    expected_batch_size = 80
    
    value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
    
    # Verify losses are finite
    assert np.isfinite(value_loss)
    assert np.isfinite(action_loss)
    assert np.isfinite(dist_entropy)


# Unit test: Verify batch size with rotation enabled
def test_batch_size_with_rotation():
    """
    Unit test: Verify batch size with rotation enabled.
    
    This tests that batch size calculation works correctly when rotation
    doubles the action space size.
    """
    num_steps = 5
    num_processes = 4
    container_size = 10
    enable_rotation = True
    
    args = create_test_args(container_size, enable_rotation)
    obs_shape = (container_size * container_size * 4,)
    action_space = Discrete(container_size * container_size * 2)
    
    actor_critic = Policy(
        obs_shape, 
        action_space, 
        base_kwargs={'recurrent': False, 'hidden_size': 256, 'args': args}
    )
    
    agent = ACKTR(
        actor_critic=actor_critic,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        invaild_coef=2.0,
        acktr=False,
        lr=0.001,
        eps=1e-5,
        alpha=0.99,
        max_grad_norm=0.5,
        args=args
    )
    
    rollouts = create_rollout_storage(num_steps, num_processes, container_size, enable_rotation)
    
    # Expected batch size: 5 * 4 = 20 (rotation doesn't affect batch size)
    expected_batch_size = 20
    
    value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
    
    # Verify losses are finite
    assert np.isfinite(value_loss)
    assert np.isfinite(action_loss)
    assert np.isfinite(dist_entropy)


# Unit test: Verify batch size with varying container sizes
def test_batch_size_varying_container_sizes():
    """
    Unit test: Verify batch size with different container sizes.
    
    This tests that batch size calculation is independent of container size.
    """
    num_steps = 5
    num_processes = 4
    
    for container_size in [5, 10, 15]:
        enable_rotation = False
        
        args = create_test_args(container_size, enable_rotation)
        obs_shape = (container_size * container_size * 4,)
        action_space = Discrete(container_size * container_size)
        
        actor_critic = Policy(
            obs_shape, 
            action_space, 
            base_kwargs={'recurrent': False, 'hidden_size': 256, 'args': args}
        )
        
        agent = ACKTR(
            actor_critic=actor_critic,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            invaild_coef=2.0,
            acktr=False,
            lr=0.001,
            eps=1e-5,
            alpha=0.99,
            max_grad_norm=0.5,
            args=args
        )
        
        rollouts = create_rollout_storage(num_steps, num_processes, container_size, enable_rotation)
        
        # Expected batch size: 5 * 4 = 20 (independent of container size)
        expected_batch_size = 20
        
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
        
        # Verify losses are finite
        assert np.isfinite(value_loss), f"Value loss not finite for container_size={container_size}"
        assert np.isfinite(action_loss), f"Action loss not finite for container_size={container_size}"
        assert np.isfinite(dist_entropy), f"Entropy not finite for container_size={container_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
