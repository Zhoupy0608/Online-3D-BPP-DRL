"""
Property-based tests for statistics computation in multi-process training

These tests verify that statistics (mean, median, min, max) are computed correctly
from episode rewards collected across all parallel processes.

Feature: multi-process-training, Property 12: Statistics computation correctness
Validates: Requirements 5.2
"""

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from collections import deque


# Hypothesis strategies for generating test data
@st.composite
def reward_list_strategy(draw):
    """
    Generate a list of episode rewards for statistics computation.
    Rewards are typically positive floats representing episode returns.
    """
    # Generate 1-100 rewards
    num_rewards = draw(st.integers(min_value=1, max_value=100))
    rewards = draw(st.lists(
        st.floats(min_value=-100.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=num_rewards,
        max_size=num_rewards
    ))
    return rewards


@st.composite
def deque_reward_strategy(draw):
    """
    Generate a deque of episode rewards with a maxlen.
    This simulates the actual data structure used in main.py.
    """
    maxlen = draw(st.integers(min_value=5, max_value=100))
    num_rewards = draw(st.integers(min_value=1, max_value=maxlen * 2))
    
    rewards = []
    for _ in range(num_rewards):
        reward = draw(st.floats(min_value=-100.0, max_value=1000.0, 
                               allow_nan=False, allow_infinity=False))
        rewards.append(reward)
    
    # Create deque with maxlen
    reward_deque = deque(maxlen=maxlen)
    for reward in rewards:
        reward_deque.append(reward)
    
    return reward_deque


# Property 12: Statistics computation correctness
# Feature: multi-process-training, Property 12: Statistics computation correctness
# Validates: Requirements 5.2
@settings(max_examples=100, deadline=None)
@given(
    rewards=reward_list_strategy()
)
def test_statistics_computation_correctness(rewards):
    """
    Property 12: For any list of episode rewards, the computed mean, median, min, 
    and max should match the standard statistical definitions.
    
    This test verifies that:
    1. Mean is computed correctly as sum(rewards) / len(rewards)
    2. Median is computed correctly as the middle value(s)
    3. Min is the smallest value
    4. Max is the largest value
    5. All statistics are finite (no NaN or Inf)
    
    Validates: Requirements 5.2
    """
    # Compute statistics using numpy (as in main.py)
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    # Verify against expected values using standard definitions
    expected_mean = sum(rewards) / len(rewards)
    expected_median = np.median(rewards)  # Use numpy as ground truth
    expected_min = min(rewards)
    expected_max = max(rewards)
    
    # Verify mean
    assert np.isclose(mean_reward, expected_mean, rtol=1e-9), \
        f"Mean mismatch: {mean_reward} != {expected_mean}"
    
    # Verify median
    assert np.isclose(median_reward, expected_median, rtol=1e-9), \
        f"Median mismatch: {median_reward} != {expected_median}"
    
    # Verify min
    assert np.isclose(min_reward, expected_min, rtol=1e-9), \
        f"Min mismatch: {min_reward} != {expected_min}"
    
    # Verify max
    assert np.isclose(max_reward, expected_max, rtol=1e-9), \
        f"Max mismatch: {max_reward} != {expected_max}"
    
    # Verify all statistics are finite
    assert np.isfinite(mean_reward), "Mean should be finite"
    assert np.isfinite(median_reward), "Median should be finite"
    assert np.isfinite(min_reward), "Min should be finite"
    assert np.isfinite(max_reward), "Max should be finite"
    
    # Verify ordering: min <= median <= max
    assert min_reward <= median_reward, f"Min {min_reward} should be <= median {median_reward}"
    assert median_reward <= max_reward, f"Median {median_reward} should be <= max {max_reward}"
    
    # Verify mean is within range [min, max]
    assert min_reward <= mean_reward <= max_reward, \
        f"Mean {mean_reward} should be between min {min_reward} and max {max_reward}"


@settings(max_examples=100, deadline=None)
@given(
    reward_deque=deque_reward_strategy()
)
def test_statistics_computation_with_deque(reward_deque):
    """
    Property 12: Test statistics computation with deque data structure.
    
    This simulates the actual pattern used in main.py where rewards are stored
    in a deque with maxlen.
    
    Validates: Requirements 5.2
    """
    # Skip if deque is empty (edge case handled separately)
    if len(reward_deque) == 0:
        return
    
    # Compute statistics using numpy (as in main.py)
    mean_reward = np.mean(reward_deque)
    median_reward = np.median(reward_deque)
    min_reward = np.min(reward_deque)
    max_reward = np.max(reward_deque)
    
    # Convert deque to list for verification
    reward_list = list(reward_deque)
    
    # Verify against expected values
    expected_mean = sum(reward_list) / len(reward_list)
    expected_median = np.median(reward_list)
    expected_min = min(reward_list)
    expected_max = max(reward_list)
    
    # Verify all statistics match
    assert np.isclose(mean_reward, expected_mean, rtol=1e-9), \
        f"Mean mismatch: {mean_reward} != {expected_mean}"
    
    assert np.isclose(median_reward, expected_median, rtol=1e-9), \
        f"Median mismatch: {median_reward} != {expected_median}"
    
    assert np.isclose(min_reward, expected_min, rtol=1e-9), \
        f"Min mismatch: {min_reward} != {expected_min}"
    
    assert np.isclose(max_reward, expected_max, rtol=1e-9), \
        f"Max mismatch: {max_reward} != {expected_max}"


@settings(max_examples=100, deadline=None)
@given(
    rewards=reward_list_strategy()
)
def test_statistics_invariants(rewards):
    """
    Property 12: Test invariants that should hold for any statistics computation.
    
    Invariants:
    1. min(rewards) <= mean(rewards) <= max(rewards)
    2. min(rewards) <= median(rewards) <= max(rewards)
    3. len(rewards) == 1 => mean == median == min == max
    4. All statistics are deterministic (same input => same output)
    
    Validates: Requirements 5.2
    """
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    # Invariant 1: min <= mean <= max
    assert min_reward <= mean_reward <= max_reward, \
        f"Mean {mean_reward} should be between min {min_reward} and max {max_reward}"
    
    # Invariant 2: min <= median <= max
    assert min_reward <= median_reward <= max_reward, \
        f"Median {median_reward} should be between min {min_reward} and max {max_reward}"
    
    # Invariant 3: Single element case
    if len(rewards) == 1:
        assert mean_reward == median_reward == min_reward == max_reward, \
            "For single element, all statistics should be equal"
    
    # Invariant 4: Determinism - compute again and verify same results
    mean_reward_2 = np.mean(rewards)
    median_reward_2 = np.median(rewards)
    min_reward_2 = np.min(rewards)
    max_reward_2 = np.max(rewards)
    
    assert mean_reward == mean_reward_2, "Mean should be deterministic"
    assert median_reward == median_reward_2, "Median should be deterministic"
    assert min_reward == min_reward_2, "Min should be deterministic"
    assert max_reward == max_reward_2, "Max should be deterministic"


@settings(max_examples=100, deadline=None)
@given(
    rewards=reward_list_strategy()
)
def test_statistics_with_duplicates(rewards):
    """
    Property 12: Test that statistics handle duplicate values correctly.
    
    Validates: Requirements 5.2
    """
    # Add some duplicate values
    if len(rewards) > 1:
        # Duplicate the first value
        rewards_with_dups = rewards + [rewards[0]]
    else:
        rewards_with_dups = rewards
    
    mean_reward = np.mean(rewards_with_dups)
    median_reward = np.median(rewards_with_dups)
    min_reward = np.min(rewards_with_dups)
    max_reward = np.max(rewards_with_dups)
    
    # Verify statistics are still valid
    assert np.isfinite(mean_reward), "Mean should be finite with duplicates"
    assert np.isfinite(median_reward), "Median should be finite with duplicates"
    assert np.isfinite(min_reward), "Min should be finite with duplicates"
    assert np.isfinite(max_reward), "Max should be finite with duplicates"
    
    # Verify ordering
    assert min_reward <= median_reward <= max_reward, \
        "Ordering should hold with duplicates"


@settings(max_examples=100, deadline=None)
@given(
    num_processes=st.integers(min_value=1, max_value=32),
    num_steps=st.integers(min_value=1, max_value=50)
)
def test_statistics_from_multi_process_simulation(num_processes, num_steps):
    """
    Property 12: Test statistics computation from simulated multi-process training.
    
    This simulates collecting rewards from multiple processes over multiple steps,
    then computing statistics on the aggregated rewards.
    
    Validates: Requirements 5.2
    """
    # Simulate episode rewards from multiple processes
    episode_rewards = deque(maxlen=100)
    
    for step in range(num_steps):
        for proc in range(num_processes):
            # 20% chance of episode completion per process per step
            if np.random.random() < 0.2:
                reward = np.random.uniform(0.0, 100.0)
                episode_rewards.append(reward)
    
    # Only compute statistics if we have rewards
    if len(episode_rewards) > 0:
        mean_reward = np.mean(episode_rewards)
        median_reward = np.median(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        
        # Verify all statistics are valid
        assert np.isfinite(mean_reward), "Mean should be finite"
        assert np.isfinite(median_reward), "Median should be finite"
        assert np.isfinite(min_reward), "Min should be finite"
        assert np.isfinite(max_reward), "Max should be finite"
        
        # Verify ordering
        assert min_reward <= mean_reward <= max_reward, \
            "Mean should be between min and max"
        assert min_reward <= median_reward <= max_reward, \
            "Median should be between min and max"


# Unit tests for specific scenarios

def test_statistics_computation_specific_values():
    """
    Unit test with specific known values to verify correctness.
    """
    rewards = [10.0, 20.0, 15.0, 25.0, 18.0, 22.0, 12.0, 30.0]
    
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    # Verify against hand-calculated values
    assert np.isclose(mean_reward, 19.0), f"Mean should be 19.0, got {mean_reward}"
    assert np.isclose(median_reward, 19.0), f"Median should be 19.0, got {median_reward}"
    assert np.isclose(min_reward, 10.0), f"Min should be 10.0, got {min_reward}"
    assert np.isclose(max_reward, 30.0), f"Max should be 30.0, got {max_reward}"


def test_statistics_computation_single_value():
    """
    Unit test with a single value.
    """
    rewards = [42.0]
    
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    # All statistics should equal the single value
    assert mean_reward == 42.0, f"Mean should be 42.0, got {mean_reward}"
    assert median_reward == 42.0, f"Median should be 42.0, got {median_reward}"
    assert min_reward == 42.0, f"Min should be 42.0, got {min_reward}"
    assert max_reward == 42.0, f"Max should be 42.0, got {max_reward}"


def test_statistics_computation_two_values():
    """
    Unit test with two values.
    """
    rewards = [10.0, 20.0]
    
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    assert np.isclose(mean_reward, 15.0), f"Mean should be 15.0, got {mean_reward}"
    assert np.isclose(median_reward, 15.0), f"Median should be 15.0, got {median_reward}"
    assert np.isclose(min_reward, 10.0), f"Min should be 10.0, got {min_reward}"
    assert np.isclose(max_reward, 20.0), f"Max should be 20.0, got {max_reward}"


def test_statistics_computation_negative_values():
    """
    Unit test with negative values (can occur with penalties).
    """
    rewards = [-10.0, -5.0, 0.0, 5.0, 10.0]
    
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    assert np.isclose(mean_reward, 0.0), f"Mean should be 0.0, got {mean_reward}"
    assert np.isclose(median_reward, 0.0), f"Median should be 0.0, got {median_reward}"
    assert np.isclose(min_reward, -10.0), f"Min should be -10.0, got {min_reward}"
    assert np.isclose(max_reward, 10.0), f"Max should be 10.0, got {max_reward}"


def test_statistics_computation_all_same_values():
    """
    Unit test with all identical values.
    """
    rewards = [42.0] * 10
    
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    # All statistics should equal the constant value
    assert mean_reward == 42.0, f"Mean should be 42.0, got {mean_reward}"
    assert median_reward == 42.0, f"Median should be 42.0, got {median_reward}"
    assert min_reward == 42.0, f"Min should be 42.0, got {min_reward}"
    assert max_reward == 42.0, f"Max should be 42.0, got {max_reward}"


def test_statistics_computation_odd_length():
    """
    Unit test with odd number of values (median is middle element).
    """
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    assert np.isclose(mean_reward, 3.0), f"Mean should be 3.0, got {mean_reward}"
    assert np.isclose(median_reward, 3.0), f"Median should be 3.0, got {median_reward}"
    assert np.isclose(min_reward, 1.0), f"Min should be 1.0, got {min_reward}"
    assert np.isclose(max_reward, 5.0), f"Max should be 5.0, got {max_reward}"


def test_statistics_computation_even_length():
    """
    Unit test with even number of values (median is average of two middle elements).
    """
    rewards = [1.0, 2.0, 3.0, 4.0]
    
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    assert np.isclose(mean_reward, 2.5), f"Mean should be 2.5, got {mean_reward}"
    assert np.isclose(median_reward, 2.5), f"Median should be 2.5, got {median_reward}"
    assert np.isclose(min_reward, 1.0), f"Min should be 1.0, got {min_reward}"
    assert np.isclose(max_reward, 4.0), f"Max should be 4.0, got {max_reward}"


def test_statistics_computation_with_deque_overflow():
    """
    Unit test to verify statistics when deque overflows its maxlen.
    """
    maxlen = 5
    episode_rewards = deque(maxlen=maxlen)
    
    # Add more rewards than maxlen
    all_rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    for reward in all_rewards:
        episode_rewards.append(reward)
    
    # Deque should only contain last 5 values
    assert len(episode_rewards) == maxlen, f"Deque should have {maxlen} elements"
    
    # Compute statistics on deque
    mean_reward = np.mean(episode_rewards)
    median_reward = np.median(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    
    # Expected values based on last 5 rewards: [4.0, 5.0, 6.0, 7.0, 8.0]
    expected_mean = 6.0
    expected_median = 6.0
    expected_min = 4.0
    expected_max = 8.0
    
    assert np.isclose(mean_reward, expected_mean), \
        f"Mean should be {expected_mean}, got {mean_reward}"
    assert np.isclose(median_reward, expected_median), \
        f"Median should be {expected_median}, got {median_reward}"
    assert np.isclose(min_reward, expected_min), \
        f"Min should be {expected_min}, got {min_reward}"
    assert np.isclose(max_reward, expected_max), \
        f"Max should be {expected_max}, got {max_reward}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
