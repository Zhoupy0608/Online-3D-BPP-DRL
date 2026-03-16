"""
Performance Optimization Module for Reliable Robot Packing

This module provides performance optimizations including:
1. Profiling utilities for point cloud processing
2. Caching for frequently computed values (candidate maps, collision checks)
3. Optimized motion option generation for large buffer ranges
"""

import time
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from functools import lru_cache
import hashlib


class PerformanceProfiler:
    """
    Profiler for tracking performance metrics of key operations.
    
    Tracks execution time and call counts for different operations
    to identify performance bottlenecks.
    """
    
    def __init__(self):
        """Initialize the profiler with empty metrics."""
        self.metrics = {}
        self.enabled = True
        
    def profile(self, operation_name: str):
        """
        Context manager for profiling an operation.
        
        Usage:
            with profiler.profile("point_cloud_processing"):
                # code to profile
                pass
        
        Args:
            operation_name: Name of the operation being profiled
        """
        return ProfileContext(self, operation_name)
        
    def record(self, operation_name: str, duration: float):
        """
        Record a profiling measurement.
        
        Args:
            operation_name: Name of the operation
            duration: Execution time in seconds
        """
        if not self.enabled:
            return
            
        if operation_name not in self.metrics:
            self.metrics[operation_name] = {
                'count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        metrics = self.metrics[operation_name]
        metrics['count'] += 1
        metrics['total_time'] += duration
        metrics['min_time'] = min(metrics['min_time'], duration)
        metrics['max_time'] = max(metrics['max_time'], duration)
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of profiling metrics.
        
        Returns:
            Dictionary mapping operation names to their metrics
        """
        summary = {}
        for op_name, metrics in self.metrics.items():
            count = metrics['count']
            if count > 0:
                summary[op_name] = {
                    'count': count,
                    'total_time': metrics['total_time'],
                    'avg_time': metrics['total_time'] / count,
                    'min_time': metrics['min_time'],
                    'max_time': metrics['max_time']
                }
        return summary
        
    def print_summary(self):
        """Print a formatted summary of profiling metrics."""
        print("\n=== Performance Profile Summary ===")
        summary = self.get_summary()
        
        if not summary:
            print("No profiling data collected.")
            return
            
        # Sort by total time descending
        sorted_ops = sorted(summary.items(), 
                          key=lambda x: x[1]['total_time'], 
                          reverse=True)
        
        print(f"{'Operation':<40} {'Count':>8} {'Total(s)':>10} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
        print("-" * 98)
        
        for op_name, metrics in sorted_ops:
            print(f"{op_name:<40} {metrics['count']:>8} "
                  f"{metrics['total_time']:>10.3f} "
                  f"{metrics['avg_time']*1000:>10.2f} "
                  f"{metrics['min_time']*1000:>10.2f} "
                  f"{metrics['max_time']*1000:>10.2f}")
                  
    def reset(self):
        """Reset all profiling metrics."""
        self.metrics = {}
        
    def enable(self):
        """Enable profiling."""
        self.enabled = True
        
    def disable(self):
        """Disable profiling."""
        self.enabled = False


class ProfileContext:
    """Context manager for profiling operations."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.profiler.record(self.operation_name, duration)
        return False


class CandidateMapCache:
    """
    Cache for candidate maps to avoid recomputation.
    
    Candidate maps depend on:
    - Height map state
    - Box dimensions
    - Container size
    - Buffer range
    - Rotation enabled/disabled
    
    This cache stores computed candidate maps and retrieves them when
    the same state is encountered again.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to cache
        """
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def _compute_key(self, 
                     height_map: np.ndarray,
                     box_size: Tuple[int, int, int],
                     container_size: Tuple[int, int, int],
                     buffer_range: Tuple[int, int],
                     rotation: bool) -> str:
        """
        Compute cache key from state parameters.
        
        Args:
            height_map: Current height map
            box_size: Box dimensions (x, y, z)
            container_size: Container dimensions
            buffer_range: Buffer space requirements
            rotation: Whether rotation is enabled
            
        Returns:
            Hash string as cache key
        """
        # Create a hashable representation of the state
        # Use height map hash + parameters
        height_hash = hashlib.md5(height_map.tobytes()).hexdigest()
        params = (height_hash, box_size, container_size, buffer_range, rotation)
        key = str(params)
        return key
        
    def get(self,
            height_map: np.ndarray,
            box_size: Tuple[int, int, int],
            container_size: Tuple[int, int, int],
            buffer_range: Tuple[int, int],
            rotation: bool) -> Optional[np.ndarray]:
        """
        Retrieve cached candidate map if available.
        
        Args:
            height_map: Current height map
            box_size: Box dimensions
            container_size: Container dimensions
            buffer_range: Buffer space requirements
            rotation: Whether rotation is enabled
            
        Returns:
            Cached candidate map or None if not found
        """
        key = self._compute_key(height_map, box_size, container_size, 
                               buffer_range, rotation)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key].copy()
        else:
            self.misses += 1
            return None
            
    def put(self,
            height_map: np.ndarray,
            box_size: Tuple[int, int, int],
            container_size: Tuple[int, int, int],
            buffer_range: Tuple[int, int],
            rotation: bool,
            candidate_map: np.ndarray):
        """
        Store candidate map in cache.
        
        Args:
            height_map: Current height map
            box_size: Box dimensions
            container_size: Container dimensions
            buffer_range: Buffer space requirements
            rotation: Whether rotation is enabled
            candidate_map: Computed candidate map to cache
        """
        # Implement simple LRU by removing oldest entry when full
        if len(self.cache) >= self.max_size:
            # Remove first (oldest) entry
            first_key = next(iter(self.cache))
            del self.cache[first_key]
            
        key = self._compute_key(height_map, box_size, container_size,
                               buffer_range, rotation)
        self.cache[key] = candidate_map.copy()
        
    def clear(self):
        """Clear the cache."""
        self.cache = {}
        self.hits = 0
        self.misses = 0
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class OptimizedMotionGenerator:
    """
    Optimized motion option generator for large buffer ranges.
    
    Uses spatial indexing and early termination to reduce computation
    when buffer ranges are large.
    """
    
    def __init__(self, 
                 buffer_range: Tuple[int, int] = (1, 1),
                 container_size: Tuple[int, int, int] = (10, 10, 10)):
        """
        Initialize the optimized generator.
        
        Args:
            buffer_range: (delta_x, delta_y) buffer space range
            container_size: Container dimensions
        """
        self.buffer_range = buffer_range
        self.container_size = container_size
        
    def generate_motion_options_optimized(self,
                                         target_pos: Tuple[int, int, int],
                                         box_size: Tuple[int, int, int],
                                         height_map: np.ndarray,
                                         max_options: int = 50) -> List[Dict[str, Any]]:
        """
        Generate motion options with optimization for large buffer ranges.
        
        Optimizations:
        1. Early termination after finding max_options valid options
        2. Spiral search pattern starting from target (most likely positions first)
        3. Vectorized height map queries
        
        Args:
            target_pos: (x, y, z) target position from DRL agent
            box_size: (lx, ly, lz) dimensions of box to place
            height_map: Current height map
            max_options: Maximum number of options to generate
            
        Returns:
            List of motion option dictionaries
        """
        options = []
        target_x, target_y, target_z = target_pos
        lx, ly, lz = box_size
        delta_x, delta_y = self.buffer_range
        width, length, height = self.container_size
        
        # Generate positions in spiral order (closest to target first)
        positions = self._spiral_positions(target_x, target_y, delta_x, delta_y)
        
        for candidate_x, candidate_y in positions:
            # Early termination if we have enough options
            if len(options) >= max_options:
                break
                
            # Boundary check
            if candidate_x < 0 or candidate_y < 0:
                continue
            if candidate_x + lx > width or candidate_y + ly > length:
                continue
                
            # Vectorized height map query
            region = height_map[candidate_x:candidate_x + lx,
                               candidate_y:candidate_y + ly]
            max_height = np.max(region)
            candidate_z = max_height
            
            # Height check
            if candidate_z + lz > height:
                continue
                
            # Calculate weight (vectorized)
            height_sum = np.sum(region)
            buffer_x = min(candidate_x, width - (candidate_x + lx))
            buffer_y = min(candidate_y, length - (candidate_y + ly))
            weight = height_sum + (buffer_x + buffer_y) * 10.0
            
            # Collision check
            collision_free = self._check_collision_fast(
                candidate_x, candidate_y, candidate_z,
                lx, ly, lz, max_height
            )
            
            if collision_free:
                option = {
                    'position': (candidate_x, candidate_y, candidate_z),
                    'weight': weight,
                    'buffer_space': (buffer_x, buffer_y),
                    'collision_free': True
                }
                options.append(option)
                
        return options
        
    def _spiral_positions(self, 
                         center_x: int, 
                         center_y: int,
                         delta_x: int,
                         delta_y: int) -> List[Tuple[int, int]]:
        """
        Generate positions in spiral order starting from center.
        
        This ensures we check positions closest to the target first,
        which are most likely to be good options.
        
        Args:
            center_x: Center x coordinate
            center_y: Center y coordinate
            delta_x: Maximum x offset
            delta_y: Maximum y offset
            
        Returns:
            List of (x, y) positions in spiral order
        """
        positions = []
        
        # Start with center
        positions.append((center_x, center_y))
        
        # Generate spiral outward
        for radius in range(1, max(delta_x, delta_y) + 1):
            # Top edge
            for dx in range(-min(radius, delta_x), min(radius, delta_x) + 1):
                dy = -min(radius, delta_y)
                if abs(dx) <= delta_x and abs(dy) <= delta_y:
                    positions.append((center_x + dx, center_y + dy))
                    
            # Bottom edge
            for dx in range(-min(radius, delta_x), min(radius, delta_x) + 1):
                dy = min(radius, delta_y)
                if abs(dx) <= delta_x and abs(dy) <= delta_y:
                    positions.append((center_x + dx, center_y + dy))
                    
            # Left edge (excluding corners)
            for dy in range(-min(radius, delta_y) + 1, min(radius, delta_y)):
                dx = -min(radius, delta_x)
                if abs(dx) <= delta_x and abs(dy) <= delta_y:
                    positions.append((center_x + dx, center_y + dy))
                    
            # Right edge (excluding corners)
            for dy in range(-min(radius, delta_y) + 1, min(radius, delta_y)):
                dx = min(radius, delta_x)
                if abs(dx) <= delta_x and abs(dy) <= delta_y:
                    positions.append((center_x + dx, center_y + dy))
                    
        return positions
        
    def _check_collision_fast(self,
                              x: int, y: int, z: int,
                              lx: int, ly: int, lz: int,
                              expected_height: float) -> bool:
        """
        Fast collision check using pre-computed height.
        
        Args:
            x, y, z: Position coordinates
            lx, ly, lz: Box dimensions
            expected_height: Pre-computed maximum height at position
            
        Returns:
            True if collision-free, False otherwise
        """
        width, length, height = self.container_size
        
        # Boundary checks
        if x < 0 or y < 0 or z < 0:
            return False
        if x + lx > width or y + ly > length:
            return False
        if z + lz > height:
            return False
            
        # Height consistency check (with tolerance)
        if abs(z - expected_height) > 0.01:
            return False
            
        return True


# Global profiler instance
_global_profiler = PerformanceProfiler()

def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler
