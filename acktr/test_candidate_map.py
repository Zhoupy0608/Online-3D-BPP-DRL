"""
Property-based tests for candidate map generation with buffer space.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a mock torch module for testing
    class MockTensor:
        def __init__(self, data):
            self.data = np.array(data)
        
        def cpu(self):
            return self
        
        def numpy(self):
            return self.data
        
        def float(self):
            return self
    
    class torch:
        @staticmethod
        def from_numpy(arr):
            return MockTensor(arr)

from acktr.utils import get_possible_position, get_rotation_mask, check_buffer_space


# Strategy for generating valid container sizes
@st.composite
def container_size_strategy(draw):
    """Generate valid container sizes."""
    width = draw(st.integers(min_value=5, max_value=20))
    length = draw(st.integers(min_value=5, max_value=20))
    height = draw(st.integers(min_value=5, max_value=20))
    return (width, length, height)


# Strategy for generating valid box sizes given a container
@st.composite
def box_size_strategy(draw, container_size):
    """Generate valid box sizes that fit in the container."""
    width, length, height = container_size
    x = draw(st.integers(min_value=1, max_value=width))
    y = draw(st.integers(min_value=1, max_value=length))
    z = draw(st.integers(min_value=1, max_value=height))
    return (x, y, z)


# Strategy for generating height maps
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


# Strategy for generating observations
@st.composite
def observation_strategy(draw, container_size):
    """Generate valid observations."""
    width, length, height = container_size
    height_map = draw(height_map_strategy(container_size))
    box_size = draw(box_size_strategy(container_size))
    
    # Create observation in the format expected by the functions
    # Shape: (4, width * length)
    # Channel 0: height_map (flattened)
    # Channel 1: box x dimension (repeated)
    # Channel 2: box y dimension (repeated)
    # Channel 3: box z dimension (repeated)
    obs = np.zeros((4, width * length))
    obs[0, :] = height_map.flatten()
    obs[1, :] = box_size[0]
    obs[2, :] = box_size[1]
    obs[3, :] = box_size[2]
    
    return torch.from_numpy(obs).float(), box_size


# Strategy for buffer range
@st.composite
def buffer_range_strategy(draw):
    """Generate valid buffer ranges."""
    buffer_x = draw(st.integers(min_value=0, max_value=3))
    buffer_y = draw(st.integers(min_value=0, max_value=3))
    return (buffer_x, buffer_y)


@settings(max_examples=100)
@given(
    container_size=container_size_strategy(),
    buffer_range=buffer_range_strategy(),
    data=st.data()
)
def test_candidate_map_validity(container_size, buffer_range, data):
    """
    Feature: reliable-robot-packing, Property 9: Candidate map marks only valid positions
    
    For any item and container state, all positions marked as valid (1) in the 
    candidate map should allow placement without exceeding container boundaries.
    
    Validates: Requirements 3.3, 6.1
    """
    width, length, height = container_size
    
    # Generate observation
    observation, box_size = data.draw(observation_strategy(container_size))
    box_x, box_y, box_z = box_size
    
    # Get candidate map
    candidate_map = get_possible_position(observation, container_size, buffer_range)
    candidate_array = np.array(candidate_map).reshape((width, length))
    
    # Check all positions marked as valid
    for i in range(width):
        for j in range(length):
            if candidate_array[i, j] == 1:
                # Position is marked as valid, verify it doesn't exceed boundaries
                # Check x boundary
                assert i + box_x <= width, \
                    f"Position ({i}, {j}) marked valid but box extends beyond x boundary: {i + box_x} > {width}"
                
                # Check y boundary
                assert j + box_y <= length, \
                    f"Position ({i}, {j}) marked valid but box extends beyond y boundary: {j + box_y} > {length}"
                
                # Check z boundary (height)
                height_map = observation.cpu().numpy()[0, :].reshape((width, length))
                max_height_at_pos = np.max(height_map[i:i + box_x, j:j + box_y])
                assert max_height_at_pos + box_z <= height, \
                    f"Position ({i}, {j}) marked valid but box exceeds height: {max_height_at_pos + box_z} > {height}"


@settings(max_examples=100)
@given(
    container_size=container_size_strategy(),
    data=st.data()
)
def test_rotation_generates_two_candidate_maps(container_size, data):
    """
    Feature: reliable-robot-packing, Property 10: Rotation generates two candidate maps
    
    For any item when rotation is enabled, the system should generate exactly 
    two candidate maps (original orientation and 90-degree rotation).
    
    Validates: Requirements 3.4
    """
    width, length, height = container_size
    
    # Generate observation
    observation, box_size = data.draw(observation_strategy(container_size))
    
    # Get rotation mask
    rotation_mask = get_rotation_mask(observation, container_size)
    
    # The rotation mask should have length = 2 * width * length
    # First half is original orientation, second half is rotated
    expected_length = 2 * width * length
    assert len(rotation_mask) == expected_length, \
        f"Rotation mask should have length {expected_length}, got {len(rotation_mask)}"
    
    # Split into two masks
    mask1 = rotation_mask[:width * length]
    mask2 = rotation_mask[width * length:]
    
    # Both masks should have the correct length
    assert len(mask1) == width * length, \
        f"First mask should have length {width * length}, got {len(mask1)}"
    assert len(mask2) == width * length, \
        f"Second mask should have length {width * length}, got {len(mask2)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@settings(max_examples=100)
@given(
    container_size=container_size_strategy(),
    buffer_range=buffer_range_strategy(),
    data=st.data()
)
def test_buffer_space_validation(container_size, buffer_range, data):
    """
    Feature: reliable-robot-packing, Property 14: Buffer space validation
    
    For any position when buffer space is enabled, if the position is marked valid, 
    there should exist minimum clearance distance in x and y directions.
    
    Validates: Requirements 6.2, 6.3
    """
    buffer_x, buffer_y = buffer_range
    
    # Skip test if no buffer is required
    if buffer_x == 0 and buffer_y == 0:
        return
    
    width, length, height = container_size
    
    # Generate observation
    observation, box_size = data.draw(observation_strategy(container_size))
    box_x, box_y, box_z = box_size
    
    # Get candidate map with buffer space enabled
    candidate_map = get_possible_position(observation, container_size, buffer_range)
    candidate_array = np.array(candidate_map).reshape((width, length))
    
    # Get height map
    height_map = observation.cpu().numpy()[0, :].reshape((width, length))
    
    # Check all positions marked as valid have sufficient buffer space
    for i in range(width):
        for j in range(length):
            if candidate_array[i, j] == 1:
                # Position is marked as valid, verify buffer space exists
                # Get the placement height
                if i + box_x <= width and j + box_y <= length:
                    placement_height = np.max(height_map[i:i + box_x, j:j + box_y])
                    
                    # Check buffer space in negative x direction
                    if i >= buffer_x:
                        buffer_region = height_map[i - buffer_x:i, j:j + box_y]
                        # Buffer region should not have items higher than placement
                        assert np.all(buffer_region <= placement_height), \
                            f"Position ({i}, {j}) marked valid but buffer space violated in -x direction"
                    
                    # Check buffer space in positive x direction
                    if i + box_x + buffer_x <= width:
                        buffer_region = height_map[i + box_x:i + box_x + buffer_x, j:j + box_y]
                        assert np.all(buffer_region <= placement_height), \
                            f"Position ({i}, {j}) marked valid but buffer space violated in +x direction"
                    
                    # Check buffer space in negative y direction
                    if j >= buffer_y:
                        buffer_region = height_map[i:i + box_x, j - buffer_y:j]
                        assert np.all(buffer_region <= placement_height), \
                            f"Position ({i}, {j}) marked valid but buffer space violated in -y direction"
                    
                    # Check buffer space in positive y direction
                    if j + box_y + buffer_y <= length:
                        buffer_region = height_map[i:i + box_x, j + box_y:j + box_y + buffer_y]
                        assert np.all(buffer_region <= placement_height), \
                            f"Position ({i}, {j}) marked valid but buffer space violated in +y direction"
