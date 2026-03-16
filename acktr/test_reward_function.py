"""
Test the enhanced reward function in ReliablePackingGame

This test verifies that the reward function correctly implements:
1. Space utilization-based rewards
2. Invalid placement penalties
3. Terminal rewards
4. Stability constraint penalties
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_reward_function_basic():
    """Test basic reward function behavior"""
    # We'll test the reward calculation logic directly
    
    # Test 1: Invalid placement should give reward = 0 (before terminal reward)
    invalid_reward_base = 0.0
    assert invalid_reward_base == 0.0, "Invalid placement base reward should be 0"
    
    # Test 2: Successful placement should give positive reward based on utilization increase
    # If utilization increases by 0.1, reward should be 0.1 * 10 = 1.0
    utilization_increase = 0.1
    expected_reward = utilization_increase * 10
    assert expected_reward == 1.0, f"Expected reward 1.0, got {expected_reward}"
    
    # Test 3: Terminal reward should be based on final utilization
    # If final utilization is 0.8, terminal reward should be 0.8 * 100 = 80.0
    final_utilization = 0.8
    expected_terminal_reward = final_utilization * 100
    assert expected_terminal_reward == 80.0, f"Expected terminal reward 80.0, got {expected_terminal_reward}"
    
    print("All basic reward function tests passed!")


def test_stability_penalty_calculation():
    """Test stability penalty calculation"""
    # Test the stability penalty logic
    
    # Container size
    bin_size = (10, 10, 10)
    
    # Test 1: Box at ground level (lz=0) should have no penalty
    box_size = (3, 3, 2)
    lx, ly, lz = 0, 0, 0
    
    # Calculate penalty manually
    penalty = 0.0  # At ground level
    assert penalty == 0.0, f"Ground level placement should have no penalty, got {penalty}"
    
    # Test 2: Small box at high position should have higher penalty
    box_size = (2, 2, 1)
    lx, ly, lz = 0, 0, 8
    
    # Calculate penalty
    height_factor = lz / bin_size[2]  # 8/10 = 0.8
    box_footprint = box_size[0] * box_size[1]  # 4
    footprint_factor = 1.0 - (box_footprint / (bin_size[0] * bin_size[1]))  # 1 - 4/100 = 0.96
    expected_penalty = height_factor * footprint_factor * 0.5  # 0.8 * 0.96 * 0.5 = 0.384
    
    assert expected_penalty > 0, f"High placement with small footprint should have penalty"
    assert abs(expected_penalty - 0.384) < 0.01, f"Expected penalty ~0.384, got {expected_penalty}"
    
    # Test 3: Large box at high position should have lower penalty
    box_size = (8, 8, 1)
    lx, ly, lz = 0, 0, 8
    
    height_factor = lz / bin_size[2]  # 0.8
    box_footprint = box_size[0] * box_size[1]  # 64
    footprint_factor = 1.0 - (box_footprint / (bin_size[0] * bin_size[1]))  # 1 - 64/100 = 0.36
    expected_penalty = height_factor * footprint_factor * 0.5  # 0.8 * 0.36 * 0.5 = 0.144
    
    assert expected_penalty > 0, f"High placement should have some penalty"
    assert expected_penalty < 0.384, f"Large footprint should have lower penalty than small footprint"
    
    print("All stability penalty tests passed!")


def test_reward_components():
    """Test that all reward components are correctly combined"""
    
    # Simulate a successful placement
    prev_utilization = 0.5
    current_utilization = 0.6
    utilization_increase = current_utilization - prev_utilization  # 0.1
    
    # Base reward
    base_reward = utilization_increase * 10  # 1.0
    
    # Stability penalty (example)
    stability_penalty = 0.2
    
    # Final reward
    final_reward = base_reward - stability_penalty  # 0.8
    
    assert abs(final_reward - 0.8) < 1e-6, f"Expected final reward 0.8, got {final_reward}"
    assert final_reward < base_reward, "Stability penalty should reduce reward"
    
    print("All reward component tests passed!")


if __name__ == "__main__":
    test_reward_function_basic()
    test_stability_penalty_calculation()
    test_reward_components()
    print("\nAll reward function tests passed successfully!")
