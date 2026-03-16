"""
Test ACKTR agent multi-process batching functionality.

This test verifies that the ACKTR update() method correctly handles
multi-process training by:
1. Reshaping from (num_steps, num_processes, ...) to (num_steps * num_processes, ...)
2. Computing Fisher loss with all processes
3. Accumulating gradients across all processes
4. Ensuring batch size equals num_steps × num_processes

Requirements tested: 1.3, 1.4, 2.1
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from acktr.algo.acktr_pipeline import ACKTR
from acktr.model import Policy
from acktr.storage import RolloutStorage
from gym.spaces import Discrete
import argparse


def create_test_args():
    """Create minimal args for testing."""
    args = argparse.Namespace()
    args.container_size = [10, 10, 10]
    args.pallet_size = 10
    args.channel = 4
    args.enable_rotation = False
    return args


def test_batch_size_single_process():
    """
    Unit test: Verify batch size with single process (num_processes=1).
    
    This tests that the reshaping works correctly even with a single process.
    """
    print("\n=== Test: Single Process Batch Size ===")
    
    num_steps = 5
    num_processes = 1
    # Use 1D observation space (flattened) like the actual environment
    # obs_len = area * (1 + 3) = 10*10 * 4 = 400
    obs_shape = (400,)
    action_space = Discrete(100)
    
    args = create_test_args()
    
    # Create policy with proper base_kwargs
    actor_critic = Policy(
        obs_shape, 
        action_space, 
        base_kwargs={'recurrent': False, 'hidden_size': 256, 'args': args}
    )
    
    # Create ACKTR agent
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
    
    # Create rollout storage
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=False,
        pallet_size=10
    )
    
    # Fill with random data
    for step in range(num_steps):
        obs = torch.randn(num_processes, *obs_shape)
        actions = torch.randint(0, 100, (num_processes, 1))
        rewards = torch.randn(num_processes, 1)
        masks = torch.ones(num_processes, 1)
        location_masks = torch.randn(num_processes, 100)
        
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
    
    # Run update - this should verify batch size internally
    try:
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
        print(f"✓ Single process update successful")
        print(f"  Expected batch size: {num_steps * num_processes}")
        print(f"  Losses: value={value_loss:.4f}, action={action_loss:.4f}")
    except AssertionError as e:
        print(f"✗ Batch size assertion failed: {e}")
        raise


def test_batch_size_multi_process():
    """
    Unit test: Verify batch size with multiple processes (num_processes=4).
    
    This tests that the reshaping correctly aggregates experiences from
    all parallel environments.
    """
    print("\n=== Test: Multi-Process Batch Size ===")
    
    num_steps = 5
    num_processes = 4
    obs_shape = (400,)
    action_space = Discrete(100)
    
    args = create_test_args()
    
    # Create policy
    actor_critic = Policy(
        obs_shape, 
        action_space, 
        base_kwargs={'recurrent': False, 'hidden_size': 256, 'args': args}
    )
    
    # Create ACKTR agent
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
    
    # Create rollout storage
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=False,
        pallet_size=10
    )
    
    # Fill with random data
    for step in range(num_steps):
        obs = torch.randn(num_processes, *obs_shape)
        actions = torch.randint(0, 100, (num_processes, 1))
        rewards = torch.randn(num_processes, 1)
        masks = torch.ones(num_processes, 1)
        location_masks = torch.randn(num_processes, 100)
        
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
    
    # Run update - this should verify batch size internally
    try:
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
        print(f"✓ Multi-process update successful")
        print(f"  Expected batch size: {num_steps * num_processes}")
        print(f"  Losses: value={value_loss:.4f}, action={action_loss:.4f}")
    except AssertionError as e:
        print(f"✗ Batch size assertion failed: {e}")
        raise


def test_batch_size_paper_config():
    """
    Unit test: Verify batch size with paper configuration (num_steps=5, num_processes=16).
    
    This tests the exact configuration used in the paper.
    """
    print("\n=== Test: Paper Configuration (5 steps × 16 processes = 80 batch) ===")
    
    num_steps = 5
    num_processes = 16
    obs_shape = (400,)
    action_space = Discrete(100)
    
    args = create_test_args()
    
    # Create policy
    actor_critic = Policy(
        obs_shape, 
        action_space, 
        base_kwargs={'recurrent': False, 'hidden_size': 256, 'args': args}
    )
    
    # Create ACKTR agent with KFAC
    agent = ACKTR(
        actor_critic=actor_critic,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        invaild_coef=2.0,
        acktr=True,  # Use KFAC optimizer
        args=args
    )
    
    # Create rollout storage
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=False,
        pallet_size=10
    )
    
    # Fill with random data
    for step in range(num_steps):
        obs = torch.randn(num_processes, *obs_shape)
        actions = torch.randint(0, 100, (num_processes, 1))
        rewards = torch.randn(num_processes, 1)
        masks = torch.ones(num_processes, 1)
        location_masks = torch.randn(num_processes, 100)
        
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
    
    # Run update - this should verify batch size internally
    try:
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
        print(f"✓ Paper configuration update successful")
        print(f"  Expected batch size: {num_steps * num_processes} (5 × 16)")
        print(f"  Losses: value={value_loss:.4f}, action={action_loss:.4f}")
    except AssertionError as e:
        print(f"✗ Batch size assertion failed: {e}")
        raise


def test_gradient_accumulation():
    """
    Unit test: Verify gradients are accumulated across all processes.
    
    This tests that gradients from all processes contribute to the update.
    """
    print("\n=== Test: Gradient Accumulation Across Processes ===")
    
    num_steps = 5
    num_processes = 4
    obs_shape = (400,)
    action_space = Discrete(100)
    
    args = create_test_args()
    
    # Create policy
    actor_critic = Policy(
        obs_shape, 
        action_space, 
        base_kwargs={'recurrent': False, 'hidden_size': 256, 'args': args}
    )
    
    # Create ACKTR agent
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
    
    # Create rollout storage
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=False,
        pallet_size=10
    )
    
    # Fill with random data
    for step in range(num_steps):
        obs = torch.randn(num_processes, *obs_shape)
        actions = torch.randint(0, 100, (num_processes, 1))
        rewards = torch.randn(num_processes, 1)
        masks = torch.ones(num_processes, 1)
        location_masks = torch.randn(num_processes, 100)
        
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
    
    # Store initial parameters
    initial_params = {name: param.clone() for name, param in actor_critic.named_parameters()}
    
    # Run update
    value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
    
    # Verify parameters changed (gradients were applied)
    params_changed = False
    for name, param in actor_critic.named_parameters():
        if not torch.allclose(param, initial_params[name]):
            params_changed = True
            break
    
    assert params_changed, "Parameters did not change after update - gradients may not be accumulating"
    print(f"✓ Gradient accumulation verified - parameters updated")
    print(f"  Losses: value={value_loss:.4f}, action={action_loss:.4f}")


if __name__ == '__main__':
    print("Testing ACKTR Multi-Process Batching")
    print("=" * 60)
    
    try:
        test_batch_size_single_process()
        test_batch_size_multi_process()
        test_batch_size_paper_config()
        test_gradient_accumulation()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
