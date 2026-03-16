"""
Property-based tests for RolloutStorage in multi-process training

These tests verify that rollout storage correctly handles multi-process data,
maintains proper tensor dimensions, and correctly computes returns with episode boundaries.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st
from gym.spaces import Discrete, Box

from acktr.storage import RolloutStorage


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
def obs_shape_strategy(draw):
    """Generate valid observation shapes."""
    channels = draw(st.integers(min_value=1, max_value=8))
    height = draw(st.integers(min_value=4, max_value=16))
    width = draw(st.integers(min_value=4, max_value=16))
    return (channels, height, width)


@st.composite
def pallet_size_strategy(draw):
    """Generate valid pallet sizes."""
    return draw(st.integers(min_value=5, max_value=15))


@st.composite
def gamma_strategy(draw):
    """Generate valid discount factors."""
    return draw(st.floats(min_value=0.9, max_value=1.0))


# Property 1: Rollout storage dimensions match process count
# Feature: multi-process-training, Property 1: Rollout storage dimensions match process count
# Validates: Requirements 1.3, 6.1
@settings(max_examples=100, deadline=None)
@given(
    num_processes=num_processes_strategy(),
    num_steps=num_steps_strategy(),
    obs_shape=obs_shape_strategy(),
    pallet_size=pallet_size_strategy()
)
def test_rollout_storage_dimensions(num_processes, num_steps, obs_shape, pallet_size):
    """
    Property 1: For any num_processes and num_steps configuration, 
    the rollout storage tensors should have shape (num_steps, num_processes, ...) 
    or (num_steps+1, num_processes, ...) for observations and masks.
    
    Validates: Requirements 1.3, 6.1
    """
    # Create action space
    action_space = Discrete(pallet_size ** 2)
    
    # Create rollout storage
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=False,
        pallet_size=pallet_size
    )
    
    # Verify observations have shape (num_steps+1, num_processes, *obs_shape)
    assert rollouts.obs.shape == (num_steps + 1, num_processes, *obs_shape), \
        f"obs shape {rollouts.obs.shape} != expected {(num_steps + 1, num_processes, *obs_shape)}"
    
    # Verify recurrent hidden states have shape (num_steps+1, num_processes, hidden_size)
    assert rollouts.recurrent_hidden_states.shape == (num_steps + 1, num_processes, 1), \
        f"recurrent_hidden_states shape {rollouts.recurrent_hidden_states.shape} != expected {(num_steps + 1, num_processes, 1)}"
    
    # Verify rewards have shape (num_steps, num_processes, 1)
    assert rollouts.rewards.shape == (num_steps, num_processes, 1), \
        f"rewards shape {rollouts.rewards.shape} != expected {(num_steps, num_processes, 1)}"
    
    # Verify value predictions have shape (num_steps+1, num_processes, 1)
    assert rollouts.value_preds.shape == (num_steps + 1, num_processes, 1), \
        f"value_preds shape {rollouts.value_preds.shape} != expected {(num_steps + 1, num_processes, 1)}"
    
    # Verify returns have shape (num_steps+1, num_processes, 1)
    assert rollouts.returns.shape == (num_steps + 1, num_processes, 1), \
        f"returns shape {rollouts.returns.shape} != expected {(num_steps + 1, num_processes, 1)}"
    
    # Verify action log probs have shape (num_steps, num_processes, 1)
    assert rollouts.action_log_probs.shape == (num_steps, num_processes, 1), \
        f"action_log_probs shape {rollouts.action_log_probs.shape} != expected {(num_steps, num_processes, 1)}"
    
    # Verify actions have shape (num_steps, num_processes, 1)
    assert rollouts.actions.shape == (num_steps, num_processes, 1), \
        f"actions shape {rollouts.actions.shape} != expected {(num_steps, num_processes, 1)}"
    
    # Verify masks have shape (num_steps+1, num_processes, 1)
    assert rollouts.masks.shape == (num_steps + 1, num_processes, 1), \
        f"masks shape {rollouts.masks.shape} != expected {(num_steps + 1, num_processes, 1)}"
    
    # Verify bad_masks have shape (num_steps+1, num_processes, 1)
    assert rollouts.bad_masks.shape == (num_steps + 1, num_processes, 1), \
        f"bad_masks shape {rollouts.bad_masks.shape} != expected {(num_steps + 1, num_processes, 1)}"
    
    # Verify location_masks have shape (num_steps+1, num_processes, pallet_size^2)
    assert rollouts.location_masks.shape == (num_steps + 1, num_processes, pallet_size ** 2), \
        f"location_masks shape {rollouts.location_masks.shape} != expected {(num_steps + 1, num_processes, pallet_size ** 2)}"


# Test insert() method with multi-process data
@settings(max_examples=50, deadline=None)
@given(
    num_processes=num_processes_strategy(),
    num_steps=num_steps_strategy(),
    pallet_size=pallet_size_strategy()
)
def test_insert_method_multiprocess(num_processes, num_steps, pallet_size):
    """
    Test that insert() method correctly handles multi-process data.
    """
    obs_shape = (4, 10, 10)
    action_space = Discrete(pallet_size ** 2)
    
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=False,
        pallet_size=pallet_size
    )
    
    # Insert data for each step
    for step in range(num_steps):
        # Create random data for all processes
        obs = torch.randn(num_processes, *obs_shape)
        recurrent_hidden_states = torch.randn(num_processes, 1)
        actions = torch.randint(0, pallet_size ** 2, (num_processes, 1))
        action_log_probs = torch.randn(num_processes, 1)
        value_preds = torch.randn(num_processes, 1)
        rewards = torch.randn(num_processes, 1)
        masks = torch.ones(num_processes, 1)
        bad_masks = torch.ones(num_processes, 1)
        location_masks = torch.randint(0, 2, (num_processes, pallet_size ** 2)).float()
        
        # Insert data
        rollouts.insert(
            obs, recurrent_hidden_states, actions, action_log_probs,
            value_preds, rewards, masks, bad_masks, location_masks
        )
        
        # Verify data was inserted at correct position
        assert torch.allclose(rollouts.obs[step + 1], obs), \
            f"obs not inserted correctly at step {step}"
        assert torch.equal(rollouts.actions[step], actions), \
            f"actions not inserted correctly at step {step}"
        assert torch.allclose(rollouts.rewards[step], rewards), \
            f"rewards not inserted correctly at step {step}"
    
    # Verify step counter wrapped correctly
    assert rollouts.step == 0, \
        f"step counter should wrap to 0, got {rollouts.step}"


# Test after_update() method
@settings(max_examples=50, deadline=None)
@given(
    num_processes=num_processes_strategy(),
    num_steps=num_steps_strategy(),
    pallet_size=pallet_size_strategy()
)
def test_after_update_copies_last_to_first(num_processes, num_steps, pallet_size):
    """
    Test that after_update() correctly copies last step to first position.
    """
    obs_shape = (4, 10, 10)
    action_space = Discrete(pallet_size ** 2)
    
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=False,
        pallet_size=pallet_size
    )
    
    # Set unique values at last position
    last_obs = torch.randn(num_processes, *obs_shape)
    last_hidden = torch.randn(num_processes, 1)
    last_masks = torch.ones(num_processes, 1) * 0.5
    last_bad_masks = torch.ones(num_processes, 1) * 0.7
    last_location_masks = torch.randn(num_processes, pallet_size ** 2)
    
    rollouts.obs[-1] = last_obs
    rollouts.recurrent_hidden_states[-1] = last_hidden
    rollouts.masks[-1] = last_masks
    rollouts.bad_masks[-1] = last_bad_masks
    rollouts.location_masks[-1] = last_location_masks
    
    # Call after_update
    rollouts.after_update()
    
    # Verify last step was copied to first
    assert torch.allclose(rollouts.obs[0], last_obs), \
        "obs[0] should equal obs[-1] after update"
    assert torch.allclose(rollouts.recurrent_hidden_states[0], last_hidden), \
        "recurrent_hidden_states[0] should equal recurrent_hidden_states[-1] after update"
    assert torch.allclose(rollouts.masks[0], last_masks), \
        "masks[0] should equal masks[-1] after update"
    assert torch.allclose(rollouts.bad_masks[0], last_bad_masks), \
        "bad_masks[0] should equal bad_masks[-1] after update"
    assert torch.allclose(rollouts.location_masks[0], last_location_masks), \
        "location_masks[0] should equal location_masks[-1] after update"


# Property 10: Return computation with episode boundaries
# Feature: multi-process-training, Property 10: Return computation with episode boundaries
# Validates: Requirements 4.4
@settings(max_examples=100, deadline=None)
@given(
    num_processes=num_processes_strategy(),
    num_steps=num_steps_strategy(),
    gamma=gamma_strategy(),
    pallet_size=pallet_size_strategy()
)
def test_return_computation_with_episode_boundaries(num_processes, num_steps, gamma, pallet_size):
    """
    Property 10: For any sequence of rewards and done masks, returns should be 
    computed as R_t = r_t + gamma * R_{t+1} * mask_{t+1}, where mask is 0 at 
    episode boundaries and 1 otherwise.
    
    Validates: Requirements 4.4
    """
    obs_shape = (4, 10, 10)
    action_space = Discrete(pallet_size ** 2)
    
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=False,
        pallet_size=pallet_size
    )
    
    # Generate random rewards
    rewards = torch.randn(num_steps, num_processes, 1)
    rollouts.rewards = rewards
    
    # Generate random masks (0 or 1)
    # 0 means episode boundary, 1 means continue
    masks = torch.randint(0, 2, (num_steps + 1, num_processes, 1)).float()
    rollouts.masks = masks
    
    # Set bad_masks to all 1s (no time limit issues)
    rollouts.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
    
    # Set next_value
    next_value = torch.randn(num_processes, 1)
    
    # Compute returns
    rollouts.compute_returns(next_value, use_gae=False, gamma=gamma, 
                            gae_lambda=0.95, use_proper_time_limits=False)
    
    # Manually compute expected returns and verify
    expected_returns = torch.zeros(num_steps + 1, num_processes, 1)
    expected_returns[-1] = next_value
    
    for step in reversed(range(num_steps)):
        expected_returns[step] = (
            rewards[step] + gamma * expected_returns[step + 1] * masks[step + 1]
        )
    
    # Verify computed returns match expected
    assert torch.allclose(rollouts.returns, expected_returns, rtol=1e-5, atol=1e-5), \
        f"Returns computation incorrect. Max diff: {(rollouts.returns - expected_returns).abs().max()}"
    
    # Verify that when mask is 0 (episode boundary), the return doesn't include future rewards
    for step in range(num_steps):
        for proc in range(num_processes):
            if masks[step + 1, proc, 0] == 0:
                # At episode boundary, return should just be the reward
                expected = rewards[step, proc, 0]
                actual = rollouts.returns[step, proc, 0]
                assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5), \
                    f"At episode boundary (step={step}, proc={proc}), return should be reward only"


# Unit test: Test with specific episode boundary scenario
def test_return_computation_with_specific_boundary():
    """
    Unit test with a specific scenario to verify episode boundary handling.
    """
    num_steps = 5
    num_processes = 2
    obs_shape = (4, 10, 10)
    pallet_size = 10
    action_space = Discrete(pallet_size ** 2)
    gamma = 0.99
    
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=False,
        pallet_size=pallet_size
    )
    
    # Set specific rewards: all 1.0
    rollouts.rewards = torch.ones(num_steps, num_processes, 1)
    
    # Set masks: episode ends at step 2 for process 0
    rollouts.masks = torch.ones(num_steps + 1, num_processes, 1)
    rollouts.masks[3, 0, 0] = 0  # Episode boundary after step 2 for process 0
    
    rollouts.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
    
    # Set next_value to 0
    next_value = torch.zeros(num_processes, 1)
    
    # Compute returns
    rollouts.compute_returns(next_value, use_gae=False, gamma=gamma, 
                            gae_lambda=0.95, use_proper_time_limits=False)
    
    # For process 0:
    # Step 4: R[4] = r[4] + gamma * 0 * 1 = 1.0
    # Step 3: R[3] = r[3] + gamma * R[4] * 1 = 1.0 + 0.99 * 1.0 = 1.99
    # Step 2: R[2] = r[2] + gamma * R[3] * 0 = 1.0 (episode boundary)
    # Step 1: R[1] = r[1] + gamma * R[2] * 1 = 1.0 + 0.99 * 1.0 = 1.99
    # Step 0: R[0] = r[0] + gamma * R[1] * 1 = 1.0 + 0.99 * 1.99 ≈ 2.9701
    
    expected_returns_proc0 = torch.tensor([
        [2.9701],  # step 0
        [1.99],    # step 1
        [1.0],     # step 2 (boundary)
        [1.99],    # step 3
        [1.0],     # step 4
        [0.0]      # step 5 (next_value)
    ])
    
    # For process 1 (no boundaries):
    # Step 4: R[4] = 1.0
    # Step 3: R[3] = 1.0 + 0.99 * 1.0 = 1.99
    # Step 2: R[2] = 1.0 + 0.99 * 1.99 ≈ 2.9701
    # Step 1: R[1] = 1.0 + 0.99 * 2.9701 ≈ 3.9404
    # Step 0: R[0] = 1.0 + 0.99 * 3.9404 ≈ 4.901
    
    expected_returns_proc1 = torch.tensor([
        [4.901],   # step 0
        [3.9404],  # step 1
        [2.9701],  # step 2
        [1.99],    # step 3
        [1.0],     # step 4
        [0.0]      # step 5 (next_value)
    ])
    
    # Verify returns for process 0
    assert torch.allclose(rollouts.returns[:, 0, :], expected_returns_proc0, rtol=1e-3, atol=1e-3), \
        f"Returns for process 0 incorrect.\nExpected:\n{expected_returns_proc0}\nGot:\n{rollouts.returns[:, 0, :]}"
    
    # Verify returns for process 1
    assert torch.allclose(rollouts.returns[:, 1, :], expected_returns_proc1, rtol=1e-3, atol=1e-3), \
        f"Returns for process 1 incorrect.\nExpected:\n{expected_returns_proc1}\nGot:\n{rollouts.returns[:, 1, :]}"


# Unit test: Test with rotation enabled
def test_rollout_storage_with_rotation():
    """
    Unit test to verify storage dimensions with rotation enabled.
    """
    num_steps = 5
    num_processes = 4
    obs_shape = (4, 10, 10)
    pallet_size = 10
    action_space = Discrete(2 * pallet_size ** 2)  # Double for rotation
    
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=True,
        pallet_size=pallet_size
    )
    
    # Verify location_masks has double the size for rotation
    expected_mask_size = 2 * pallet_size ** 2
    assert rollouts.location_masks.shape == (num_steps + 1, num_processes, expected_mask_size), \
        f"location_masks shape with rotation {rollouts.location_masks.shape} != expected {(num_steps + 1, num_processes, expected_mask_size)}"


# Unit test: Test with give_up enabled
def test_rollout_storage_with_give_up():
    """
    Unit test to verify storage dimensions with give_up enabled.
    """
    num_steps = 5
    num_processes = 4
    obs_shape = (4, 10, 10)
    pallet_size = 10
    action_space = Discrete(pallet_size ** 2 + 1)  # +1 for give up action
    
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=True,
        enable_rotation=False,
        pallet_size=pallet_size
    )
    
    # Verify location_masks has +1 size for give_up action
    expected_mask_size = pallet_size ** 2 + 1
    assert rollouts.location_masks.shape == (num_steps + 1, num_processes, expected_mask_size), \
        f"location_masks shape with give_up {rollouts.location_masks.shape} != expected {(num_steps + 1, num_processes, expected_mask_size)}"


# Unit test: Test device transfer
def test_rollout_storage_device_transfer():
    """
    Unit test to verify storage can be moved to different devices.
    """
    num_steps = 3
    num_processes = 2
    obs_shape = (4, 10, 10)
    pallet_size = 10
    action_space = Discrete(pallet_size ** 2)
    
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=obs_shape,
        action_space=action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=False,
        pallet_size=pallet_size
    )
    
    # Initially on CPU
    assert rollouts.obs.device.type == 'cpu'
    
    # Move to CPU explicitly (should work)
    rollouts.to(torch.device('cpu'))
    assert rollouts.obs.device.type == 'cpu'
    assert rollouts.rewards.device.type == 'cpu'
    assert rollouts.masks.device.type == 'cpu'
    
    # If CUDA is available, test GPU transfer
    if torch.cuda.is_available():
        rollouts.to(torch.device('cuda:0'))
        assert rollouts.obs.device.type == 'cuda'
        assert rollouts.rewards.device.type == 'cuda'
        assert rollouts.masks.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
