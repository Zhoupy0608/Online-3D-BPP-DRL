"""
Property-Based Tests for Parallel Entry Motion Module

This module contains property-based tests using Hypothesis to verify
the correctness of the parallel entry motion primitive implementation.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from acktr.motion_primitive import ParallelEntryMotion, MotionOption


# Test strategies for generating random inputs
@st.composite
def container_size_strategy(draw):
    """Generate valid container sizes."""
    width = draw(st.integers(min_value=5, max_value=20))
    length = draw(st.integers(min_value=5, max_value=20))
    height = draw(st.integers(min_value=5, max_value=20))
    return (width, length, height)


@st.composite
def buffer_range_strategy(draw):
    """Generate valid buffer ranges."""
    delta_x = draw(st.integers(min_value=0, max_value=3))
    delta_y = draw(st.integers(min_value=0, max_value=3))
    return (delta_x, delta_y)


@st.composite
def box_size_strategy(draw, container_size):
    """Generate valid box sizes that fit in container."""
    width, length, height = container_size
    lx = draw(st.integers(min_value=1, max_value=min(width, 5)))
    ly = draw(st.integers(min_value=1, max_value=min(length, 5)))
    lz = draw(st.integers(min_value=1, max_value=min(height, 5)))
    return (lx, ly, lz)


@st.composite
def target_position_strategy(draw, container_size, box_size):
    """Generate valid target positions."""
    width, length, height = container_size
    lx, ly, lz = box_size
    # Ensure target position allows box to fit
    x = draw(st.integers(min_value=0, max_value=max(0, width - lx)))
    y = draw(st.integers(min_value=0, max_value=max(0, length - ly)))
    z = draw(st.integers(min_value=0, max_value=max(0, height - lz)))
    return (x, y, z)


@st.composite
def height_map_strategy(draw, container_size):
    """Generate valid height maps."""
    width, length, height = container_size
    # Generate height map with values between 0 and height
    height_map = draw(st.lists(
        st.integers(min_value=0, max_value=height),
        min_size=width * length,
        max_size=width * length
    ))
    return np.array(height_map).reshape((width, length))


# Property 4: Motion options are within buffer range
# Feature: reliable-robot-packing, Property 4: Motion options are within buffer range
# Validates: Requirements 2.2
@settings(max_examples=100)
@given(
    container_size=container_size_strategy(),
    buffer_range=buffer_range_strategy(),
)
def test_motion_options_within_buffer_range(container_size, buffer_range):
    """
    Property: For any target position and buffer range (Δx, Δy), all generated
    motion options should have positions within [target_x - Δx, target_x + Δx]
    and [target_y - Δy, target_y + Δy].
    """
    # Generate box size and target position
    box_size = (2, 2, 2)  # Simple box for testing
    width, length, height = container_size
    
    # Generate valid target position
    target_x = width // 2
    target_y = length // 2
    target_z = 0
    target_pos = (target_x, target_y, target_z)
    
    # Create height map (all zeros for simplicity)
    height_map = np.zeros((width, length), dtype=np.float32)
    
    # Create motion primitive
    motion = ParallelEntryMotion(buffer_range=buffer_range, container_size=container_size)
    
    # Generate motion options
    options = motion.generate_motion_options(target_pos, box_size, height_map)
    
    # Verify all options are within buffer range
    delta_x, delta_y = buffer_range
    for option in options:
        x, y, z = option.position
        assert target_x - delta_x <= x <= target_x + delta_x, \
            f"Position x={x} outside buffer range [{target_x - delta_x}, {target_x + delta_x}]"
        assert target_y - delta_y <= y <= target_y + delta_y, \
            f"Position y={y} outside buffer range [{target_y - delta_y}, {target_y + delta_y}]"


# Property 5: Motion option weights are deterministic
# Feature: reliable-robot-packing, Property 5: Motion option weights are deterministic
# Validates: Requirements 2.3
@settings(max_examples=100)
@given(
    container_size=container_size_strategy(),
    buffer_range=buffer_range_strategy(),
)
def test_motion_option_weights_deterministic(container_size, buffer_range):
    """
    Property: For any motion option and height map, calculating the weight
    twice should produce identical results.
    """
    # Generate test data
    box_size = (2, 2, 2)
    width, length, height = container_size
    target_pos = (width // 2, length // 2, 0)
    
    # Create random height map
    np.random.seed(42)  # For reproducibility in this test
    height_map = np.random.randint(0, height // 2, size=(width, length)).astype(np.float32)
    
    # Create motion primitive
    motion = ParallelEntryMotion(buffer_range=buffer_range, container_size=container_size)
    
    # Generate motion options twice
    options1 = motion.generate_motion_options(target_pos, box_size, height_map)
    options2 = motion.generate_motion_options(target_pos, box_size, height_map)
    
    # Verify same number of options
    assert len(options1) == len(options2), "Different number of options generated"
    
    # Sort options by position for comparison
    options1_sorted = sorted(options1, key=lambda opt: opt.position)
    options2_sorted = sorted(options2, key=lambda opt: opt.position)
    
    # Verify weights are identical
    for opt1, opt2 in zip(options1_sorted, options2_sorted):
        assert opt1.position == opt2.position, "Positions don't match"
        assert abs(opt1.weight - opt2.weight) < 1e-6, \
            f"Weights differ: {opt1.weight} vs {opt2.weight}"


# Property 6: Selected motion option has maximum weight
# Feature: reliable-robot-packing, Property 6: Selected motion option has maximum weight
# Validates: Requirements 2.4
@settings(max_examples=100)
@given(
    container_size=container_size_strategy(),
    buffer_range=buffer_range_strategy(),
)
def test_selected_option_has_maximum_weight(container_size, buffer_range):
    """
    Property: For any non-empty list of collision-free motion options,
    the selected option should have weight greater than or equal to all other options.
    """
    # Generate test data
    box_size = (2, 2, 2)
    width, length, height = container_size
    target_pos = (width // 2, length // 2, 0)
    
    # Create height map
    height_map = np.zeros((width, length), dtype=np.float32)
    
    # Create motion primitive
    motion = ParallelEntryMotion(buffer_range=buffer_range, container_size=container_size)
    
    # Generate motion options
    options = motion.generate_motion_options(target_pos, box_size, height_map)
    
    # Filter collision-free options
    collision_free_options = [opt for opt in options if opt.collision_free]
    
    # Skip if no collision-free options
    if not collision_free_options:
        return
    
    # Select best option
    best_option = motion.select_best_option(options)
    
    # Verify best option has maximum weight
    for option in collision_free_options:
        assert best_option.weight >= option.weight, \
            f"Selected option weight {best_option.weight} < option weight {option.weight}"


# Property 7: Collision detection prevents invalid placements
# Feature: reliable-robot-packing, Property 7: Collision detection prevents invalid placements
# Validates: Requirements 2.5
@settings(max_examples=100)
@given(
    container_size=container_size_strategy(),
)
def test_collision_detection_prevents_invalid_placements(container_size):
    """
    Property: For any motion option that would cause a box to exceed container
    boundaries or overlap with existing boxes, collision checking should return False.
    """
    width, length, height = container_size
    motion = ParallelEntryMotion(container_size=container_size)
    
    # Test 1: Position outside boundaries (negative coordinates)
    box_size = (2, 2, 2)
    height_map = np.zeros((width, length), dtype=np.float32)
    
    invalid_position = (-1, 0, 0)
    assert not motion.check_collision(invalid_position, box_size, height_map), \
        "Collision detection should reject negative x coordinate"
    
    invalid_position = (0, -1, 0)
    assert not motion.check_collision(invalid_position, box_size, height_map), \
        "Collision detection should reject negative y coordinate"
    
    # Test 2: Position exceeds container boundaries
    invalid_position = (width - 1, 0, 0)  # Box would extend beyond width
    assert not motion.check_collision(invalid_position, box_size, height_map), \
        "Collision detection should reject position exceeding width"
    
    invalid_position = (0, length - 1, 0)  # Box would extend beyond length
    assert not motion.check_collision(invalid_position, box_size, height_map), \
        "Collision detection should reject position exceeding length"
    
    # Test 3: Box exceeds container height
    invalid_position = (0, 0, height - 1)  # Box would extend beyond height
    assert not motion.check_collision(invalid_position, box_size, height_map), \
        "Collision detection should reject position exceeding height"
    
    # Test 4: Valid position should pass
    valid_position = (0, 0, 0)
    assert motion.check_collision(valid_position, box_size, height_map), \
        "Collision detection should accept valid position"
    
    # Test 5: Box not resting on height map surface
    height_map[0:2, 0:2] = 3.0  # Set height to 3
    invalid_position = (0, 0, 0)  # Box at z=0 but should be at z=3
    assert not motion.check_collision(invalid_position, box_size, height_map), \
        "Collision detection should reject box not resting on surface"
    
    valid_position = (0, 0, 3)  # Box at correct height
    assert motion.check_collision(valid_position, box_size, height_map), \
        "Collision detection should accept box resting on surface"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
