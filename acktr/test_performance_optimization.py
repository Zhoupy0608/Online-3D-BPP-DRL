"""Tests for performance optimization features."""

import pytest
import numpy as np
import time
from acktr.performance_optimizer import PerformanceProfiler, CandidateMapCache, OptimizedMotionGenerator, get_profiler
from acktr.visual_feedback import VisualFeedbackModule
from acktr.motion_primitive import ParallelEntryMotion
from acktr.utils import get_possible_position, clear_candidate_map_cache


def test_profiler_records_metrics():
    """Test that profiler correctly records operation metrics."""
    profiler = PerformanceProfiler()
    with profiler.profile("test_operation"):
        time.sleep(0.01)
    summary = profiler.get_summary()
    assert "test_operation" in summary
    assert summary["test_operation"]["count"] == 1


def test_cache_miss_then_hit():
    """Test cache miss followed by cache hit."""
    cache = CandidateMapCache(max_size=10)
    height_map = np.zeros((10, 10))
    result = cache.get(height_map, (2, 2, 2), (10, 10, 10), (1, 1), False)
    assert result is None
    candidate_map = np.ones((10, 10))
    cache.put(height_map, (2, 2, 2), (10, 10, 10), (1, 1), False, candidate_map)
    result = cache.get(height_map, (2, 2, 2), (10, 10, 10), (1, 1), False)
    assert result is not None


def test_optimized_generation():
    """Test optimized motion generation."""
    generator = OptimizedMotionGenerator(buffer_range=(3, 3), container_size=(10, 10, 10))
    height_map = np.zeros((10, 10))
    options = generator.generate_motion_options_optimized((5, 5, 0), (2, 2, 2), height_map)
    assert len(options) > 0


def test_downsampling():
    """Test point cloud downsampling."""
    module = VisualFeedbackModule(container_size=(10, 10, 10), simulation_mode=True)
    points = np.random.rand(10000, 3) * 10
    downsampled = module._downsample_point_cloud(points, voxel_size=0.5)
    assert len(downsampled) < len(points)


def test_profiling_integration():
    """Test profiling integration."""
    profiler = get_profiler()
    profiler.reset()
    module = VisualFeedbackModule(container_size=(10, 10, 10), simulation_mode=True)
    points = np.random.rand(100, 3) * 10
    boxes = module.process_point_cloud(points)
    summary = profiler.get_summary()
    assert "point_cloud_processing_total" in summary


def test_cache_integration():
    """Test cache integration with utils."""
    clear_candidate_map_cache()
    height_map = np.zeros((10, 10))
    box_info = np.array([height_map.flatten(), np.full(100, 2), np.full(100, 2), np.full(100, 2)]).flatten()
    result1 = get_possible_position(box_info, container_size=(10, 10, 10), buffer_range=(1, 1), use_cache=True)
    result2 = get_possible_position(box_info, container_size=(10, 10, 10), buffer_range=(1, 1), use_cache=True)
    assert result1 == result2


def test_performance_comparison():
    """Test performance with and without optimization."""
    standard = ParallelEntryMotion(buffer_range=(5, 5), container_size=(20, 20, 20), use_optimized=False)
    optimized = ParallelEntryMotion(buffer_range=(5, 5), container_size=(20, 20, 20), use_optimized=True)
    height_map = np.zeros((20, 20))
    std_opts = standard.generate_motion_options((10, 10, 0), (2, 2, 2), height_map)
    opt_opts = optimized.generate_motion_options((10, 10, 0), (2, 2, 2), height_map)
    assert len(std_opts) > 0
    assert len(opt_opts) > 0
