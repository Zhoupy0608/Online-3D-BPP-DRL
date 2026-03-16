"""
Parallel Entry Motion Module for Reliable Robot Packing

This module implements the parallel entry motion primitive with buffer space
to mitigate placement uncertainty. It generates candidate motion options around
a target position and selects the best option based on height map and buffer space.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from acktr.performance_optimizer import get_profiler, OptimizedMotionGenerator


@dataclass
class MotionOption:
    """
    Represents a candidate motion option for parallel entry placement.
    
    Attributes:
        position: (x, y, z) placement position in container coordinates
        weight: Weight score based on height map and buffer availability
        buffer_space: Available buffer in (x, y) directions
        collision_free: Whether this option is collision-free
    """
    position: Tuple[int, int, int]
    weight: float
    buffer_space: Tuple[int, int]
    collision_free: bool


class ParallelEntryMotion:
    """
    Implements parallel entry motion primitive with buffer space strategy.
    
    This class generates candidate placement positions around a target position
    predicted by the DRL agent, evaluates them based on height map and buffer
    availability, and selects the best collision-free option.
    """
    
    def __init__(self, 
                 buffer_range: Tuple[int, int] = (1, 1),
                 container_size: Tuple[int, int, int] = (10, 10, 10),
                 use_optimized: bool = True):
        """
        Initialize the parallel entry motion module.
        
        Args:
            buffer_range: (delta_x, delta_y) buffer space range for motion options
            container_size: Container dimensions (width, length, height)
            use_optimized: Whether to use optimized motion generation for large buffers
        """
        self.buffer_range = buffer_range
        self.container_size = container_size
        self.use_optimized = use_optimized
        
        # Create optimized generator for large buffer ranges
        if use_optimized and (buffer_range[0] > 2 or buffer_range[1] > 2):
            self.optimized_generator = OptimizedMotionGenerator(
                buffer_range=buffer_range,
                container_size=container_size
            )
        else:
            self.optimized_generator = None
        
    def generate_motion_options(self,
                                target_pos: Tuple[int, int, int],
                                box_size: Tuple[int, int, int],
                                height_map: np.ndarray) -> List[MotionOption]:
        """
        Generate candidate motion options around target position.
        
        Creates a grid of candidate positions within the buffer range around
        the target position. Each option is evaluated based on:
        1. Sum of height map values in the placement region
        2. Available buffer space around the placement
        
        Args:
            target_pos: (x, y, z) target position from DRL agent
            box_size: (lx, ly, lz) dimensions of box to place
            height_map: Current height map (width x length)
            
        Returns:
            List of MotionOption objects with calculated weights
        """
        profiler = get_profiler()
        
        with profiler.profile("motion_option_generation"):
            # Use optimized generator for large buffer ranges
            if self.optimized_generator is not None:
                option_dicts = self.optimized_generator.generate_motion_options_optimized(
                    target_pos, box_size, height_map
                )
                # Convert to MotionOption objects
                options = [
                    MotionOption(
                        position=opt['position'],
                        weight=opt['weight'],
                        buffer_space=opt['buffer_space'],
                        collision_free=opt['collision_free']
                    )
                    for opt in option_dicts
                ]
                return options
            
            # Standard generation for small buffer ranges
            options = []
            target_x, target_y, target_z = target_pos
            lx, ly, lz = box_size
            delta_x, delta_y = self.buffer_range
            width, length, height = self.container_size
            
            # Generate candidate positions within buffer range
            for dx in range(-delta_x, delta_x + 1):
                for dy in range(-delta_y, delta_y + 1):
                    candidate_x = target_x + dx
                    candidate_y = target_y + dy
                    
                    # Check if candidate position is within container boundaries
                    if candidate_x < 0 or candidate_y < 0:
                        continue
                    if candidate_x + lx > width or candidate_y + ly > length:
                        continue
                        
                    # Get the height at this position
                    region = height_map[candidate_x:candidate_x + lx, 
                                       candidate_y:candidate_y + ly]
                    max_height = np.max(region)
                    candidate_z = max_height
                    
                    # Check if placement would exceed container height
                    if candidate_z + lz > height:
                        continue
                        
                    # Calculate weight based on height map sum
                    height_sum = np.sum(region)
                    
                    # Calculate available buffer space
                    buffer_x = min(candidate_x, width - (candidate_x + lx))
                    buffer_y = min(candidate_y, length - (candidate_y + ly))
                    
                    # Weight calculation: prioritize lower height sum and more buffer
                    # Lower height sum means more stable placement
                    # More buffer space means more tolerance for uncertainty
                    weight = height_sum + (buffer_x + buffer_y) * 10.0
                    
                    # Check collision (will be done separately)
                    collision_free = self.check_collision(
                        (candidate_x, candidate_y, candidate_z),
                        box_size,
                        height_map
                    )
                    
                    option = MotionOption(
                        position=(candidate_x, candidate_y, candidate_z),
                        weight=weight,
                        buffer_space=(buffer_x, buffer_y),
                        collision_free=collision_free
                    )
                    options.append(option)
                    
            return options
        
    def select_best_option(self, options: List[MotionOption]) -> MotionOption:
        """
        Select motion option with maximum weight from collision-free options.
        
        Args:
            options: List of candidate motion options
            
        Returns:
            Selected motion option with maximum weight
            
        Raises:
            ValueError: If no collision-free options are available
        """
        # Filter to collision-free options only
        collision_free_options = [opt for opt in options if opt.collision_free]
        
        if not collision_free_options:
            raise ValueError("No collision-free motion options available")
            
        # Select option with maximum weight
        best_option = max(collision_free_options, key=lambda opt: opt.weight)
        
        return best_option
        
    def check_collision(self,
                       position: Tuple[int, int, int],
                       box_size: Tuple[int, int, int],
                       height_map: np.ndarray) -> bool:
        """
        Check if placement at position would cause collision.
        
        A collision occurs if:
        1. Position is outside container boundaries
        2. Box would exceed container height
        3. Box would not rest properly on the height map surface
        
        Args:
            position: (x, y, z) placement position
            box_size: (lx, ly, lz) box dimensions
            height_map: Current height map (width x length)
            
        Returns:
            True if collision-free, False if collision detected
        """
        x, y, z = position
        lx, ly, lz = box_size
        width, length, height = self.container_size
        
        # Check boundary violations
        if x < 0 or y < 0 or z < 0:
            return False
        if x + lx > width or y + ly > length:
            return False
        if z + lz > height:
            return False
            
        # Check if box rests properly on height map
        region = height_map[x:x + lx, y:y + ly]
        max_height = np.max(region)
        
        # Box should be placed at the maximum height in the region
        # Allow small tolerance for floating point errors
        if abs(z - max_height) > 0.01:
            return False
            
        return True
