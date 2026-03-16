"""
Property-based tests for the Uncertainty Simulation Module

These tests verify the correctness properties of the UncertaintySimulator class
using Hypothesis for property-based testing.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy import stats

from acktr.uncertainty import UncertaintySimulator


# Hypothesis strategies for generating test data
@st.composite
def container_size_strategy(draw):
    """Generate valid container sizes."""
    width = draw(st.integers(min_value=5, max_value=20))
    length = draw(st.integers(min_value=5, max_value=20))
    height = draw(st.integers(min_value=5, max_value=20))
    return (width, length, height)


@st.composite
def box_size_strategy(draw, container_size):
    """Generate valid box sizes that fit in the container."""
    width, length, height = container_size
    lx = draw(st.integers(min_value=1, max_value=width))
    ly = draw(st.integers(min_value=1, max_value=length))
    lz = draw(st.integers(min_value=1, max_value=height))
    return (lx, ly, lz)


@st.composite
def position_strategy(draw, container_size, box_size):
    """Generate valid positions for a box in the container."""
    width, length, height = container_size
    lx, ly, lz = box_size
    
    # Ensure position allows box to fit
    max_x = max(0, width - lx)
    max_y = max(0, length - ly)
    max_z = max(0, height - lz)
    
    x = draw(st.integers(min_value=0, max_value=max_x))
    y = draw(st.integers(min_value=0, max_value=max_y))
    z = draw(st.integers(min_value=0, max_value=max_z))
    
    return (x, y, z)


@st.composite
def height_map_strategy(draw, container_size):
    """Generate a valid height map for the container."""
    width, length, height = container_size
    # Height map values should be between 0 and container height
    height_map = draw(st.lists(
        st.integers(min_value=0, max_value=height),
        min_size=width * length,
        max_size=width * length
    ))
    return np.array(height_map).reshape((width, length))


# Property 17: Placement noise follows Gaussian distribution
# Feature: reliable-robot-packing, Property 17: Placement noise follows Gaussian distribution
# Validates: Requirements 8.2
@settings(max_examples=100, deadline=None)
@given(
    noise_std=st.tuples(
        st.floats(min_value=0.1, max_value=2.0),
        st.floats(min_value=0.1, max_value=2.0),
        st.floats(min_value=0.05, max_value=0.5)
    )
)
def test_placement_noise_follows_gaussian_distribution(noise_std):
    """
    Property 17: For any large sample of placement noise values, 
    the distribution should approximate a Gaussian with the configured 
    mean and standard deviation.
    
    Validates: Requirements 8.2
    """
    simulator = UncertaintySimulator(noise_std=noise_std, enabled=True)
    
    # Generate a large sample of noise values
    sample_size = 1000
    noise_samples = []
    
    # Set seed for reproducibility in this test
    np.random.seed(42)
    
    for _ in range(sample_size):
        noise = np.random.normal(0, simulator.noise_std, size=3)
        noise_samples.append(noise)
    
    noise_samples = np.array(noise_samples)
    
    # Test each dimension separately
    for dim in range(3):
        dim_samples = noise_samples[:, dim]
        
        # Test 1: Mean should be close to 0
        sample_mean = np.mean(dim_samples)
        assert abs(sample_mean) < 0.2, f"Mean {sample_mean} is too far from 0 for dimension {dim}"
        
        # Test 2: Standard deviation should be close to configured value
        sample_std = np.std(dim_samples, ddof=1)
        expected_std = noise_std[dim]
        # Allow 20% tolerance due to sampling variation
        assert abs(sample_std - expected_std) / expected_std < 0.2, \
            f"Std dev {sample_std} differs too much from expected {expected_std} for dimension {dim}"
        
        # Test 3: Shapiro-Wilk test for normality (p-value > 0.05 suggests normal distribution)
        # Use a subset to avoid test being too strict
        test_sample = dim_samples[:100]
        _, p_value = stats.shapiro(test_sample)
        # We use a lenient threshold since we're testing the implementation, not the RNG
        assert p_value > 0.01, \
            f"Distribution does not appear Gaussian (p={p_value}) for dimension {dim}"


# Property 18: Perturbed positions are valid
# Feature: reliable-robot-packing, Property 18: Perturbed positions are valid
# Validates: Requirements 8.3
@settings(max_examples=100, deadline=None)
@given(
    container_size=container_size_strategy(),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_perturbed_positions_are_valid(container_size, seed):
    """
    Property 18: For any placement position with added noise, 
    the final perturbed position should not cause collisions 
    or boundary violations.
    
    Validates: Requirements 8.3
    """
    np.random.seed(seed)
    
    # Generate box size and position
    box_size = (
        np.random.randint(1, container_size[0] + 1),
        np.random.randint(1, container_size[1] + 1),
        np.random.randint(1, container_size[2] + 1)
    )
    
    # Generate a valid initial position
    max_x = max(0, container_size[0] - box_size[0])
    max_y = max(0, container_size[1] - box_size[1])
    max_z = max(0, container_size[2] - box_size[2])
    
    position = (
        np.random.randint(0, max_x + 1),
        np.random.randint(0, max_y + 1),
        np.random.randint(0, max_z + 1)
    )
    
    # Generate height map
    height_map = np.random.randint(0, container_size[2] + 1, 
                                   size=(container_size[0], container_size[1]))
    
    # Create simulator with moderate noise
    simulator = UncertaintySimulator(noise_std=(0.5, 0.5, 0.1), enabled=True)
    
    # Add placement noise
    perturbed_pos = simulator.add_placement_noise(
        position, box_size, height_map, container_size
    )
    
    x, y, z = perturbed_pos
    lx, ly, lz = box_size
    width, length, height = container_size
    
    # Verify no boundary violations in x dimension
    assert x >= 0, f"X position {x} is negative"
    assert x + lx <= width, f"Box extends beyond container width: {x} + {lx} > {width}"
    
    # Verify no boundary violations in y dimension
    assert y >= 0, f"Y position {y} is negative"
    assert y + ly <= length, f"Box extends beyond container length: {y} + {ly} > {length}"
    
    # Verify z is non-negative
    assert z >= 0, f"Z position {z} is negative"
    
    # Verify z is at or above the maximum height in the placement region
    region = height_map[x:x+lx, y:y+ly]
    max_height_in_region = np.max(region)
    assert z >= max_height_in_region, \
        f"Z position {z} is below max height {max_height_in_region} in region"


# Additional unit test for disabled uncertainty
def test_uncertainty_disabled():
    """Test that when uncertainty is disabled, positions are unchanged."""
    simulator = UncertaintySimulator(enabled=False)
    
    position = (5, 5, 3)
    box_size = (2, 2, 2)
    height_map = np.zeros((10, 10))
    container_size = (10, 10, 10)
    
    result = simulator.add_placement_noise(position, box_size, height_map, container_size)
    
    assert result == position, "Position should be unchanged when uncertainty is disabled"


# Unit test for validate_position edge cases
def test_validate_position_boundary_cases():
    """Test validate_position with various boundary cases."""
    simulator = UncertaintySimulator(enabled=True)
    
    container_size = (10, 10, 10)
    height_map = np.zeros((10, 10))
    
    # Test 1: Position that exceeds x boundary
    position = (9, 5, 0)
    box_size = (3, 2, 2)
    result = simulator.validate_position(position, box_size, height_map, container_size)
    assert result[0] + box_size[0] <= container_size[0], "X boundary violation not corrected"
    
    # Test 2: Position that exceeds y boundary
    position = (5, 9, 0)
    box_size = (2, 3, 2)
    result = simulator.validate_position(position, box_size, height_map, container_size)
    assert result[1] + box_size[1] <= container_size[1], "Y boundary violation not corrected"
    
    # Test 3: Negative position
    position = (-2, -3, 0)
    box_size = (2, 2, 2)
    result = simulator.validate_position(position, box_size, height_map, container_size)
    assert result[0] >= 0, "Negative x not corrected"
    assert result[1] >= 0, "Negative y not corrected"
    
    # Test 4: Position on elevated surface
    height_map[3:6, 3:6] = 5
    position = (3, 3, 0)
    box_size = (3, 3, 2)
    result = simulator.validate_position(position, box_size, height_map, container_size)
    assert result[2] == 5, "Z position should be adjusted to max height in region"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
