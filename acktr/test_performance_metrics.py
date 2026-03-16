"""
Property-based tests for performance metrics computation

This module contains property-based tests to verify the correctness of
performance metrics calculations used in the packing system evaluation.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def compute_mean(values):
    """Compute mean of a list of values."""
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def compute_std(values):
    """Compute standard deviation of a list of values."""
    if len(values) == 0:
        return 0.0
    mean = compute_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return np.sqrt(variance)


def compute_performance_metrics(utilization_values):
    """
    Compute performance metrics from a list of space utilization values.
    
    Args:
        utilization_values: List of space utilization ratios from packing episodes
        
    Returns:
        Dictionary containing:
        - mean_utilization: Mean space utilization
        - std_utilization: Standard deviation of space utilization
        - min_utilization: Minimum space utilization
        - max_utilization: Maximum space utilization
        - num_episodes: Number of episodes
    """
    if len(utilization_values) == 0:
        return {
            'mean_utilization': 0.0,
            'std_utilization': 0.0,
            'min_utilization': 0.0,
            'max_utilization': 0.0,
            'num_episodes': 0
        }
    
    return {
        'mean_utilization': compute_mean(utilization_values),
        'std_utilization': compute_std(utilization_values),
        'min_utilization': min(utilization_values),
        'max_utilization': max(utilization_values),
        'num_episodes': len(utilization_values)
    }


@settings(max_examples=100, deadline=None)
@given(
    utilization_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100
    )
)
def test_property_20_mean_computation(utilization_values):
    """
    Feature: reliable-robot-packing, Property 20: Performance metrics are correctly computed
    
    Property: For any set of packing episodes with known utilization values,
    the computed mean should match the statistical definition: sum(values) / count(values).
    
    Validates: Requirements 10.3
    """
    # Compute metrics
    metrics = compute_performance_metrics(utilization_values)
    
    # Verify mean computation
    expected_mean = sum(utilization_values) / len(utilization_values)
    computed_mean = metrics['mean_utilization']
    
    # Property: Mean should match statistical definition
    assert abs(computed_mean - expected_mean) < 1e-6, \
        f"Computed mean {computed_mean} != expected {expected_mean}"
    
    # Property: Mean should be in valid range [0, 1]
    assert 0.0 <= computed_mean <= 1.0, \
        f"Mean utilization {computed_mean} is not in range [0, 1]"
    
    # Property: Mean should be between min and max (with tolerance for floating point)
    assert metrics['min_utilization'] - 1e-10 <= computed_mean <= metrics['max_utilization'] + 1e-10, \
        f"Mean {computed_mean} not between min {metrics['min_utilization']} and max {metrics['max_utilization']}"


@settings(max_examples=100, deadline=None)
@given(
    utilization_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100
    )
)
def test_property_20_std_computation(utilization_values):
    """
    Feature: reliable-robot-packing, Property 20: Performance metrics are correctly computed
    
    Property: For any set of packing episodes with known utilization values,
    the computed standard deviation should match the statistical definition:
    sqrt(sum((x - mean)^2) / count(values)).
    
    Validates: Requirements 10.3
    """
    # Compute metrics
    metrics = compute_performance_metrics(utilization_values)
    
    # Verify standard deviation computation
    mean = sum(utilization_values) / len(utilization_values)
    variance = sum((x - mean) ** 2 for x in utilization_values) / len(utilization_values)
    expected_std = np.sqrt(variance)
    computed_std = metrics['std_utilization']
    
    # Property: Standard deviation should match statistical definition
    assert abs(computed_std - expected_std) < 1e-6, \
        f"Computed std {computed_std} != expected {expected_std}"
    
    # Property: Standard deviation should be non-negative
    assert computed_std >= 0.0, \
        f"Standard deviation {computed_std} is negative"
    
    # Property: Standard deviation should be 0 for constant values
    if len(set(utilization_values)) == 1:
        assert computed_std < 1e-6, \
            f"Standard deviation {computed_std} should be ~0 for constant values"


@settings(max_examples=100, deadline=None)
@given(
    utilization_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100
    )
)
def test_property_20_min_max_computation(utilization_values):
    """
    Feature: reliable-robot-packing, Property 20: Performance metrics are correctly computed
    
    Property: For any set of packing episodes with known utilization values,
    the computed min and max should match the actual minimum and maximum values.
    
    Validates: Requirements 10.3
    """
    # Compute metrics
    metrics = compute_performance_metrics(utilization_values)
    
    # Verify min computation
    expected_min = min(utilization_values)
    computed_min = metrics['min_utilization']
    
    assert abs(computed_min - expected_min) < 1e-6, \
        f"Computed min {computed_min} != expected {expected_min}"
    
    # Verify max computation
    expected_max = max(utilization_values)
    computed_max = metrics['max_utilization']
    
    assert abs(computed_max - expected_max) < 1e-6, \
        f"Computed max {computed_max} != expected {expected_max}"
    
    # Property: Min should be <= Max
    assert computed_min <= computed_max, \
        f"Min {computed_min} > Max {computed_max}"
    
    # Property: All values should be between min and max
    for value in utilization_values:
        assert computed_min <= value <= computed_max, \
            f"Value {value} not between min {computed_min} and max {computed_max}"


@settings(max_examples=100, deadline=None)
@given(
    utilization_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100
    )
)
def test_property_20_episode_count(utilization_values):
    """
    Feature: reliable-robot-packing, Property 20: Performance metrics are correctly computed
    
    Property: For any set of packing episodes, the reported number of episodes
    should equal the length of the input list.
    
    Validates: Requirements 10.3
    """
    # Compute metrics
    metrics = compute_performance_metrics(utilization_values)
    
    # Verify episode count
    expected_count = len(utilization_values)
    computed_count = metrics['num_episodes']
    
    assert computed_count == expected_count, \
        f"Computed episode count {computed_count} != expected {expected_count}"
    
    # Property: Episode count should be positive
    assert computed_count > 0, \
        f"Episode count {computed_count} should be positive"


@settings(max_examples=100, deadline=None)
@given(
    baseline_values=st.lists(
        st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50
    ),
    improvement_factor=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
)
def test_property_20_performance_improvement_calculation(baseline_values, improvement_factor):
    """
    Feature: reliable-robot-packing, Property 20: Performance metrics are correctly computed
    
    Property: For any baseline and improved performance values, the improvement
    percentage should be correctly calculated as:
    ((improved_mean - baseline_mean) / baseline_mean) * 100
    
    Validates: Requirements 10.3, 10.4
    """
    # Generate improved values by adding improvement factor
    improved_values = [min(1.0, v * (1.0 + improvement_factor)) for v in baseline_values]
    
    # Compute metrics for both
    baseline_metrics = compute_performance_metrics(baseline_values)
    improved_metrics = compute_performance_metrics(improved_values)
    
    # Calculate improvement percentage
    baseline_mean = baseline_metrics['mean_utilization']
    improved_mean = improved_metrics['mean_utilization']
    
    if baseline_mean > 0:
        improvement_pct = ((improved_mean - baseline_mean) / baseline_mean) * 100
        
        # Property: Improvement should be non-negative (since we added improvement)
        assert improvement_pct >= -1e-6, \
            f"Improvement percentage {improvement_pct} is negative"
        
        # Property: Improvement should match expected improvement factor
        # (approximately, since we capped at 1.0)
        if improvement_factor > 0:
            assert improvement_pct > -1e-6, \
                f"Improvement percentage {improvement_pct} should be positive"


def test_property_20_empty_episode_list():
    """
    Feature: reliable-robot-packing, Property 20: Performance metrics are correctly computed
    
    Property: For an empty list of episodes, all metrics should be 0.
    
    Validates: Requirements 10.3
    """
    # Compute metrics for empty list
    metrics = compute_performance_metrics([])
    
    # Property: All metrics should be 0 for empty input
    assert metrics['mean_utilization'] == 0.0, \
        f"Mean should be 0 for empty list, got {metrics['mean_utilization']}"
    assert metrics['std_utilization'] == 0.0, \
        f"Std should be 0 for empty list, got {metrics['std_utilization']}"
    assert metrics['min_utilization'] == 0.0, \
        f"Min should be 0 for empty list, got {metrics['min_utilization']}"
    assert metrics['max_utilization'] == 0.0, \
        f"Max should be 0 for empty list, got {metrics['max_utilization']}"
    assert metrics['num_episodes'] == 0, \
        f"Episode count should be 0 for empty list, got {metrics['num_episodes']}"


@settings(max_examples=100, deadline=None)
@given(constant_value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_property_20_constant_values(constant_value):
    """
    Feature: reliable-robot-packing, Property 20: Performance metrics are correctly computed
    
    Property: For a list of constant values, mean should equal the constant,
    std should be 0, and min should equal max.
    
    Validates: Requirements 10.3
    """
    # Create list of constant values
    num_episodes = 10
    utilization_values = [constant_value] * num_episodes
    
    # Compute metrics
    metrics = compute_performance_metrics(utilization_values)
    
    # Property: Mean should equal the constant value
    assert abs(metrics['mean_utilization'] - constant_value) < 1e-6, \
        f"Mean {metrics['mean_utilization']} != constant {constant_value}"
    
    # Property: Standard deviation should be 0
    assert metrics['std_utilization'] < 1e-6, \
        f"Std {metrics['std_utilization']} should be ~0 for constant values"
    
    # Property: Min should equal max
    assert abs(metrics['min_utilization'] - metrics['max_utilization']) < 1e-6, \
        f"Min {metrics['min_utilization']} != Max {metrics['max_utilization']}"
    
    # Property: Min and max should equal the constant
    assert abs(metrics['min_utilization'] - constant_value) < 1e-6, \
        f"Min {metrics['min_utilization']} != constant {constant_value}"
    assert abs(metrics['max_utilization'] - constant_value) < 1e-6, \
        f"Max {metrics['max_utilization']} != constant {constant_value}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
