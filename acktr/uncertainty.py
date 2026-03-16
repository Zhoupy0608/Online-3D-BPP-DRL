"""
Uncertainty Simulation Module for Reliable Robot Packing

This module provides functionality to simulate placement uncertainty during training
by adding Gaussian noise to placement positions and validating/adjusting positions
to avoid collisions and boundary violations.
"""

import numpy as np
from typing import Tuple


class UncertaintySimulator:
    """
    Simulates placement uncertainty by adding random noise to placement positions.
    
    This class is used during training to make the learned policy robust to
    real-world placement errors that occur in physical robot systems.
    """
    
    def __init__(self, noise_std: Tuple[float, float, float] = (0.5, 0.5, 0.1), 
                 enabled: bool = True):
        """
        Initialize the uncertainty simulator.
        
        Args:
            noise_std: Standard deviation of Gaussian noise for (x, y, z) dimensions
            enabled: Whether to apply uncertainty simulation
        """
        self.noise_std = np.array(noise_std)
        self.enabled = enabled
        
    def add_placement_noise(self, 
                           position: Tuple[int, int, int],
                           box_size: Tuple[int, int, int],
                           height_map: np.ndarray,
                           container_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Add random Gaussian noise to placement position.
        
        Args:
            position: Intended placement position (x, y, z)
            box_size: Box dimensions (lx, ly, lz)
            height_map: Current height map (width x length)
            container_size: Container dimensions (width, length, height)
            
        Returns:
            Perturbed position (adjusted if collision occurs)
        """
        if not self.enabled:
            return position
            
        # Generate Gaussian noise
        noise = np.random.normal(0, self.noise_std, size=3)
        
        # Apply noise to position
        perturbed_pos = (
            position[0] + noise[0],
            position[1] + noise[1],
            position[2] + noise[2]
        )
        
        # Round to integer coordinates
        perturbed_pos = (
            int(round(perturbed_pos[0])),
            int(round(perturbed_pos[1])),
            int(round(perturbed_pos[2]))
        )
        
        # Validate and adjust if needed
        valid_pos = self.validate_position(perturbed_pos, box_size, height_map, container_size)
        
        return valid_pos
        
    def validate_position(self, 
                         position: Tuple[int, int, int],
                         box_size: Tuple[int, int, int],
                         height_map: np.ndarray,
                         container_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Validate and adjust position if needed to avoid collisions and boundary violations.
        
        Args:
            position: Position to validate (x, y, z)
            box_size: Box dimensions (lx, ly, lz)
            height_map: Current height map (width x length)
            container_size: Container dimensions (width, length, height)
            
        Returns:
            Adjusted valid position
        """
        x, y, z = position
        lx, ly, lz = box_size
        width, length, height = container_size
        
        # Adjust x to stay within container boundaries
        if x < 0:
            x = 0
        if x + lx > width:
            x = width - lx
            
        # Adjust y to stay within container boundaries
        if y < 0:
            y = 0
        if y + ly > length:
            y = length - ly
            
        # Check if position is valid (no negative dimensions after adjustment)
        if x < 0 or y < 0:
            # Box is too large for container, place at origin
            x, y = 0, 0
            
        # Get the maximum height in the placement region
        if x >= 0 and y >= 0 and x + lx <= width and y + ly <= length:
            region = height_map[x:x+lx, y:y+ly]
            max_height = np.max(region)
        else:
            # Fallback to origin if still invalid
            x, y = 0, 0
            region = height_map[x:x+lx, y:y+ly]
            max_height = np.max(region)
            
        # Adjust z to be at the correct height (on top of existing boxes)
        z = max_height
        
        # Check if placement would exceed container height
        if z + lz > height:
            # Cannot place box without exceeding height, return position anyway
            # (the environment will handle this as an invalid placement)
            pass
            
        return (x, y, z)
