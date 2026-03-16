"""
Property-based tests for environment parameter serialization in multi-process training

These tests verify that environment configuration parameters can be properly
serialized via pickle for multiprocessing, which is required for the spawn
context on Windows and ensures cross-platform compatibility.
"""

import pytest
import pickle
import numpy as np
from hypothesis import given, settings, strategies as st
import json
import tempfile
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any


# Hypothesis strategies for generating test data
@st.composite
def box_size_strategy(draw):
    """Generate valid box sizes as tuples of (x, y, z)."""
    x = draw(st.integers(min_value=1, max_value=10))
    y = draw(st.integers(min_value=1, max_value=10))
    z = draw(st.integers(min_value=1, max_value=10))
    return (x, y, z)


@st.composite
def box_set_strategy(draw):
    """Generate a list of box sizes."""
    num_boxes = draw(st.integers(min_value=1, max_value=10))
    return [draw(box_size_strategy()) for _ in range(num_boxes)]


@st.composite
def container_size_strategy(draw):
    """Generate valid container sizes."""
    x = draw(st.integers(min_value=5, max_value=20))
    y = draw(st.integers(min_value=5, max_value=20))
    z = draw(st.integers(min_value=5, max_value=20))
    return (x, y, z)


@st.composite
def uncertainty_std_strategy(draw):
    """Generate uncertainty standard deviation tuples."""
    x_std = draw(st.floats(min_value=0.0, max_value=2.0))
    y_std = draw(st.floats(min_value=0.0, max_value=2.0))
    z_std = draw(st.floats(min_value=0.0, max_value=2.0))
    return (x_std, y_std, z_std)


@st.composite
def buffer_range_strategy(draw):
    """Generate buffer range tuples."""
    min_buffer = draw(st.integers(min_value=0, max_value=5))
    max_buffer = draw(st.integers(min_value=min_buffer, max_value=10))
    return (min_buffer, max_buffer)


@st.composite
def camera_config_strategy(draw):
    """Generate camera configuration (None or dict)."""
    use_camera = draw(st.booleans())
    if not use_camera:
        return None
    
    # Generate a simple camera config dict
    return {
        'position': [
            draw(st.floats(min_value=-10, max_value=10)),
            draw(st.floats(min_value=-10, max_value=10)),
            draw(st.floats(min_value=1, max_value=20))
        ],
        'target': [
            draw(st.floats(min_value=-5, max_value=5)),
            draw(st.floats(min_value=-5, max_value=5)),
            draw(st.floats(min_value=0, max_value=10))
        ],
        'fov': draw(st.floats(min_value=30, max_value=120))
    }


@dataclass
class ArgsConfig:
    """Serializable configuration object for environment parameters."""
    enable_rotation: bool
    box_size_set: List[Tuple[int, int, int]]
    container_size: Tuple[int, int, int]
    data_type: str
    uncertainty_enabled: bool
    visual_feedback_enabled: bool
    parallel_motion_enabled: bool
    uncertainty_std: Tuple[float, float, float]
    buffer_range: Tuple[int, int]
    camera_config: Optional[Dict[str, Any]]


@st.composite
def args_object_strategy(draw):
    """Generate a serializable args object with all environment parameters."""
    return ArgsConfig(
        enable_rotation=draw(st.booleans()),
        box_size_set=draw(box_set_strategy()),
        container_size=draw(container_size_strategy()),
        data_type=draw(st.sampled_from(['cut1', 'cut2', 'rs'])),
        uncertainty_enabled=draw(st.booleans()),
        visual_feedback_enabled=draw(st.booleans()),
        parallel_motion_enabled=draw(st.booleans()),
        uncertainty_std=draw(uncertainty_std_strategy()),
        buffer_range=draw(buffer_range_strategy()),
        camera_config=draw(camera_config_strategy())
    )


# Property 9: Environment parameter serialization
# Feature: multi-process-training, Property 9: Environment parameter serialization
# Validates: Requirements 3.3
@settings(max_examples=100, deadline=None)
@given(args=args_object_strategy())
def test_environment_parameters_are_serializable(args):
    """
    Property 9: For any environment configuration, the parameters 
    (box_set, container_size, reliability settings) should be 
    serializable via pickle for multiprocessing.
    
    Validates: Requirements 3.3
    """
    # Extract all environment parameters that need to be serializable
    env_params = {
        'enable_rotation': args.enable_rotation,
        'box_set': args.box_size_set,
        'container_size': args.container_size,
        'data_type': args.data_type,
        'uncertainty_enabled': args.uncertainty_enabled,
        'visual_feedback_enabled': args.visual_feedback_enabled,
        'parallel_motion_enabled': args.parallel_motion_enabled,
        'uncertainty_std': args.uncertainty_std,
        'buffer_range': args.buffer_range,
        'camera_config': args.camera_config
    }
    
    # Property 1: All parameters should be serializable via pickle
    try:
        serialized = pickle.dumps(env_params)
    except Exception as e:
        pytest.fail(f"Failed to serialize environment parameters: {e}\nParams: {env_params}")
    
    # Property 2: Deserialized parameters should equal original parameters
    try:
        deserialized = pickle.loads(serialized)
    except Exception as e:
        pytest.fail(f"Failed to deserialize environment parameters: {e}")
    
    # Verify all parameters match after round-trip
    assert deserialized['enable_rotation'] == env_params['enable_rotation'], \
        "enable_rotation changed after serialization"
    
    assert deserialized['box_set'] == env_params['box_set'], \
        f"box_set changed after serialization: {deserialized['box_set']} != {env_params['box_set']}"
    
    assert deserialized['container_size'] == env_params['container_size'], \
        "container_size changed after serialization"
    
    assert deserialized['data_type'] == env_params['data_type'], \
        "data_type changed after serialization"
    
    assert deserialized['uncertainty_enabled'] == env_params['uncertainty_enabled'], \
        "uncertainty_enabled changed after serialization"
    
    assert deserialized['visual_feedback_enabled'] == env_params['visual_feedback_enabled'], \
        "visual_feedback_enabled changed after serialization"
    
    assert deserialized['parallel_motion_enabled'] == env_params['parallel_motion_enabled'], \
        "parallel_motion_enabled changed after serialization"
    
    assert deserialized['uncertainty_std'] == env_params['uncertainty_std'], \
        "uncertainty_std changed after serialization"
    
    assert deserialized['buffer_range'] == env_params['buffer_range'], \
        "buffer_range changed after serialization"
    
    assert deserialized['camera_config'] == env_params['camera_config'], \
        f"camera_config changed after serialization: {deserialized['camera_config']} != {env_params['camera_config']}"
    
    # Property 3: The args object itself should be serializable
    try:
        serialized_args = pickle.dumps(args)
        deserialized_args = pickle.loads(serialized_args)
    except Exception as e:
        pytest.fail(f"Failed to serialize args object: {e}")
    
    # Verify key attributes are preserved
    assert deserialized_args.enable_rotation == args.enable_rotation
    assert deserialized_args.box_size_set == args.box_size_set
    assert deserialized_args.container_size == args.container_size


# Unit test: Verify basic parameter serialization
def test_basic_parameter_serialization():
    """Unit test to verify basic environment parameters can be serialized."""
    args = ArgsConfig(
        enable_rotation=False,
        box_size_set=[(2, 2, 2), (3, 3, 3)],
        container_size=(10, 10, 10),
        data_type='cut2',
        uncertainty_enabled=False,
        visual_feedback_enabled=False,
        parallel_motion_enabled=False,
        uncertainty_std=(0.5, 0.5, 0.1),
        buffer_range=(1, 1),
        camera_config=None
    )
    
    # Serialize and deserialize
    serialized = pickle.dumps(args)
    deserialized = pickle.loads(serialized)
    
    # Verify all attributes match
    assert deserialized.enable_rotation == args.enable_rotation
    assert deserialized.box_size_set == args.box_size_set
    assert deserialized.container_size == args.container_size
    assert deserialized.data_type == args.data_type
    assert deserialized.uncertainty_enabled == args.uncertainty_enabled
    assert deserialized.visual_feedback_enabled == args.visual_feedback_enabled
    assert deserialized.parallel_motion_enabled == args.parallel_motion_enabled
    assert deserialized.uncertainty_std == args.uncertainty_std
    assert deserialized.buffer_range == args.buffer_range
    assert deserialized.camera_config == args.camera_config


# Unit test: Verify camera config serialization
def test_camera_config_serialization():
    """Unit test to verify camera configuration can be serialized."""
    camera_config = {
        'position': [5.0, 5.0, 10.0],
        'target': [0.0, 0.0, 5.0],
        'fov': 60.0
    }
    
    args = ArgsConfig(
        enable_rotation=True,
        box_size_set=[(2, 2, 2)],
        container_size=(10, 10, 10),
        data_type='cut2',
        uncertainty_enabled=True,
        visual_feedback_enabled=True,
        parallel_motion_enabled=True,
        uncertainty_std=(0.5, 0.5, 0.1),
        buffer_range=(1, 2),
        camera_config=camera_config
    )
    
    # Serialize and deserialize
    serialized = pickle.dumps(args)
    deserialized = pickle.loads(serialized)
    
    # Verify camera config is preserved
    assert deserialized.camera_config == camera_config
    assert deserialized.camera_config['position'] == [5.0, 5.0, 10.0]
    assert deserialized.camera_config['target'] == [0.0, 0.0, 5.0]
    assert deserialized.camera_config['fov'] == 60.0


# Unit test: Verify complex box set serialization
def test_complex_box_set_serialization():
    """Unit test to verify complex box sets can be serialized."""
    box_set = [
        (1, 1, 1),
        (2, 3, 4),
        (5, 5, 5),
        (10, 8, 6),
        (3, 7, 2)
    ]
    
    args = ArgsConfig(
        enable_rotation=True,
        box_size_set=box_set,
        container_size=(20, 20, 20),
        data_type='rs',
        uncertainty_enabled=False,
        visual_feedback_enabled=False,
        parallel_motion_enabled=False,
        uncertainty_std=(0.0, 0.0, 0.0),
        buffer_range=(0, 0),
        camera_config=None
    )
    
    # Serialize and deserialize
    serialized = pickle.dumps(args)
    deserialized = pickle.loads(serialized)
    
    # Verify box set is preserved
    assert deserialized.box_size_set == box_set
    assert len(deserialized.box_size_set) == 5
    assert deserialized.box_size_set[0] == (1, 1, 1)
    assert deserialized.box_size_set[-1] == (3, 7, 2)


# Unit test: Verify serialization with all reliability features enabled
def test_all_reliability_features_serialization():
    """Unit test to verify serialization with all reliability features enabled."""
    args = ArgsConfig(
        enable_rotation=True,
        box_size_set=[(2, 2, 2), (3, 3, 3), (4, 4, 4)],
        container_size=(15, 15, 15),
        data_type='cut1',
        uncertainty_enabled=True,
        visual_feedback_enabled=True,
        parallel_motion_enabled=True,
        uncertainty_std=(1.0, 1.0, 0.5),
        buffer_range=(2, 5),
        camera_config={
            'position': [10.0, 10.0, 15.0],
            'target': [5.0, 5.0, 7.5],
            'fov': 90.0,
            'resolution': [640, 480]
        }
    )
    
    # Serialize and deserialize
    serialized = pickle.dumps(args)
    deserialized = pickle.loads(serialized)
    
    # Verify all reliability features are preserved
    assert deserialized.uncertainty_enabled == True
    assert deserialized.visual_feedback_enabled == True
    assert deserialized.parallel_motion_enabled == True
    assert deserialized.uncertainty_std == (1.0, 1.0, 0.5)
    assert deserialized.buffer_range == (2, 5)
    assert deserialized.camera_config['resolution'] == [640, 480]


# Unit test: Verify serialization preserves data types
def test_serialization_preserves_data_types():
    """Unit test to verify serialization preserves correct data types."""
    args = ArgsConfig(
        enable_rotation=True,  # bool
        box_size_set=[(2, 2, 2)],  # list of tuples
        container_size=(10, 10, 10),  # tuple
        data_type='cut2',  # string
        uncertainty_enabled=False,  # bool
        visual_feedback_enabled=False,  # bool
        parallel_motion_enabled=False,  # bool
        uncertainty_std=(0.5, 0.5, 0.1),  # tuple of floats
        buffer_range=(1, 1),  # tuple of ints
        camera_config=None  # None
    )
    
    # Serialize and deserialize
    serialized = pickle.dumps(args)
    deserialized = pickle.loads(serialized)
    
    # Verify data types are preserved
    assert isinstance(deserialized.enable_rotation, bool)
    assert isinstance(deserialized.box_size_set, list)
    assert isinstance(deserialized.box_size_set[0], tuple)
    assert isinstance(deserialized.container_size, tuple)
    assert isinstance(deserialized.data_type, str)
    assert isinstance(deserialized.uncertainty_enabled, bool)
    assert isinstance(deserialized.uncertainty_std, tuple)
    assert isinstance(deserialized.buffer_range, tuple)
    assert deserialized.camera_config is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
