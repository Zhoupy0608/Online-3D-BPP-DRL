"""
Enhanced Packing Environment with Reliability Features

This module extends the base PackingGame environment with reliability improvements:
1. Uncertainty simulation for robust training
2. Parallel entry motion primitive for placement
3. Visual feedback for height map correction
"""

from .bin3D import PackingGame
from .space import Box
import numpy as np
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path to import acktr modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from acktr.uncertainty import UncertaintySimulator
from acktr.motion_primitive import ParallelEntryMotion
from acktr.visual_feedback import VisualFeedbackModule


class ReliablePackingGame(PackingGame):
    """
    Extended PackingGame environment with reliability features.
    
    This environment adds three key reliability improvements:
    1. Uncertainty simulation: Adds placement noise during training
    2. Parallel entry motion: Uses buffer space strategy for placement
    3. Visual feedback: Updates height map based on actual positions
    
    The features can be enabled/disabled through constructor flags to support
    both training (with uncertainty) and deployment (with visual feedback).
    """
    
    def __init__(self,
                 uncertainty_enabled: bool = False,
                 visual_feedback_enabled: bool = False,
                 parallel_motion_enabled: bool = False,
                 noise_std: Tuple[float, float, float] = (0.5, 0.5, 0.1),
                 buffer_range: Tuple[int, int] = (1, 1),
                 camera_config = None,
                 **kwargs):
        """
        Initialize the reliable packing environment.
        
        Args:
            uncertainty_enabled: Add placement noise during training
            visual_feedback_enabled: Use visual feedback for height map updates
            parallel_motion_enabled: Use parallel entry motion primitive
            noise_std: Standard deviation for placement noise (x, y, z)
            buffer_range: Buffer space range for motion options (delta_x, delta_y)
            camera_config: Camera configuration for visual feedback
            **kwargs: Additional arguments passed to PackingGame
        """
        super().__init__(**kwargs)
        
        # Initialize reliability modules based on flags
        self.uncertainty_enabled = uncertainty_enabled
        self.visual_feedback_enabled = visual_feedback_enabled
        self.parallel_motion_enabled = parallel_motion_enabled
        
        # Create uncertainty simulator if enabled
        self.uncertainty_sim = None
        if uncertainty_enabled:
            self.uncertainty_sim = UncertaintySimulator(
                noise_std=noise_std,
                enabled=True
            )
            
        # Create parallel entry motion module if enabled
        self.parallel_motion = None
        if parallel_motion_enabled:
            self.parallel_motion = ParallelEntryMotion(
                buffer_range=buffer_range,
                container_size=self.bin_size
            )
            
        # Create visual feedback module if enabled
        self.visual_feedback = None
        if visual_feedback_enabled:
            self.visual_feedback = VisualFeedbackModule(
                camera_config=camera_config,
                container_size=self.bin_size,
                grid_size=self.bin_size[:2],
                simulation_mode=(camera_config is None)
            )
            
    def step(self, action):
        """
        Execute one step in the environment with reliability features.
        
        This method extends the base step() to incorporate:
        1. Parallel entry motion for action refinement
        2. Uncertainty simulation for training robustness
        3. Visual feedback for height map correction
        
        Args:
            action: Action index or array from the agent
            
        Returns:
            observation: Updated state observation
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information dictionary
        """
        # Parse action
        if isinstance(action, np.ndarray) or isinstance(action, list):
            idx = action[0]
        else:
            idx = action
            
        # Check whether to rotate the box
        flag = False
        if idx >= self.area:
            assert self.can_rotate
            idx = idx - self.area
            flag = True
            
        # Convert action index to position
        lx, ly = self.space.idx_to_position(idx)
        
        # Get box size (with rotation if needed)
        box_size = self.next_box
        if flag:
            # Rotate box: swap x and y dimensions
            box_size = (box_size[1], box_size[0], box_size[2])
            
        # Get current height at target position
        plain = self.space.plain
        if lx + box_size[0] <= self.bin_size[0] and ly + box_size[1] <= self.bin_size[1]:
            region = plain[lx:lx + box_size[0], ly:ly + box_size[1]]
            target_z = int(np.max(region))
        else:
            target_z = 0
            
        target_pos = (lx, ly, target_z)
        
        # Apply parallel entry motion if enabled
        final_pos = target_pos
        motion_option_used = False
        visual_feedback_update = False  # Initialize visual feedback flag
        if self.parallel_motion_enabled and self.parallel_motion is not None:
            try:
                # Generate motion options around target position
                options = self.parallel_motion.generate_motion_options(
                    target_pos=target_pos,
                    box_size=box_size,
                    height_map=plain
                )
                
                # Select best option if any are available
                if options:
                    best_option = self.parallel_motion.select_best_option(options)
                    final_pos = best_option.position
                    motion_option_used = True
            except ValueError:
                # No valid motion options, use target position
                pass
                
        # Apply uncertainty simulation if enabled (training mode)
        noise_applied = False
        if self.uncertainty_enabled and self.uncertainty_sim is not None:
            original_pos = final_pos
            final_pos = self.uncertainty_sim.add_placement_noise(
                position=final_pos,
                box_size=box_size,
                height_map=plain,
                container_size=self.bin_size
            )
            # Check if noise actually changed the position
            if final_pos != original_pos:
                noise_applied = True
            
        # Convert final position back to action index
        final_lx, final_ly, final_lz = final_pos
        final_idx = self.space.position_to_index((final_lx, final_ly))
        
        # Store previous utilization for reward calculation
        prev_utilization = self.space.get_ratio()
        
        # Execute placement using base class logic
        succeeded = self.space.drop_box(box_size, final_idx, flag)
        
        if not succeeded:
            # Invalid placement penalty (Requirement 4.3)
            reward = 0.0
            
            # Add terminal reward when episode ends (Requirement 4.4)
            terminal_reward = self._calculate_terminal_reward()
            reward += terminal_reward
            
            terminated = True
            truncated = False
            info = {
                'counter': len(self.space.boxes),
                'ratio': self.space.get_ratio(),
                'mask': np.ones(shape=self.act_len),
                'terminal_reward': terminal_reward
            }
            
            # Add reliability feature tracking info (Requirements 8.1-8.5)
            if noise_applied:
                info['noise_applied'] = True
            if visual_feedback_update:
                info['visual_feedback_update'] = True
            if motion_option_used:
                info['motion_option_used'] = True
            
            return self.cur_observation, reward, terminated, truncated, info
            
        # Calculate reward based on space utilization change (Requirements 4.1, 4.2)
        current_utilization = self.space.get_ratio()
        utilization_increase = current_utilization - prev_utilization
        
        # Base reward: utilization increase scaled by 10
        reward = utilization_increase * 10
        
        # Add stability constraint penalties (Requirement 4.5)
        # Check if the placement violates stability constraints
        stability_penalty = self._calculate_stability_penalty(box_size, final_lx, final_ly, final_lz)
        reward -= stability_penalty
        
        # Apply visual feedback if enabled (deployment mode)
        visual_feedback_update = False
        if self.visual_feedback_enabled and self.visual_feedback is not None:
            # Capture point cloud from camera
            point_cloud = self.visual_feedback.capture_point_cloud()
            
            # Process point cloud to detect boxes
            detected_boxes = self.visual_feedback.process_point_cloud(point_cloud)
            
            # Update height map based on detected positions
            if detected_boxes:
                self.space.plain = self.visual_feedback.update_height_map(
                    height_map=self.space.plain,
                    boxes=detected_boxes
                )
                visual_feedback_update = True
                
        # Update box creator
        self.box_creator.drop_box()  # Remove current box from list
        self.box_creator.generate_box_size()  # Add new box to list
        
        # Check if episode should end
        terminated = False
        truncated = False
        
        # Check if we should add terminal reward (Requirement 4.4)
        # Terminal reward is added when the episode ends naturally
        # (This would be triggered by the box creator running out of boxes or other conditions)
        
        # Prepare return values
        info = {
            'counter': len(self.space.boxes),
            'ratio': self.space.get_ratio()
        }
        
        # Add reliability feature tracking info (Requirements 8.1-8.5)
        if noise_applied:
            info['noise_applied'] = True
        if visual_feedback_update:
            info['visual_feedback_update'] = True
        if motion_option_used:
            info['motion_option_used'] = True
        
        return self.cur_observation, reward, terminated, truncated, info
        
    def _calculate_stability_penalty(self, box_size: Tuple[int, int, int], 
                                     lx: int, ly: int, lz: int) -> float:
        """
        Calculate stability penalty for a placement.
        
        Stability constraints penalize:
        1. Boxes placed high above the ground with insufficient support
        2. Boxes that create unstable configurations
        
        Args:
            box_size: (x, y, z) dimensions of the placed box
            lx, ly, lz: Position of the placed box
            
        Returns:
            Penalty value (0 if stable, positive if unstable)
        """
        penalty = 0.0
        
        # Penalty for placing boxes at high positions with small footprint
        # This encourages placing larger boxes at the bottom
        if lz > 0:
            box_volume = box_size[0] * box_size[1] * box_size[2]
            box_footprint = box_size[0] * box_size[1]
            
            # Higher penalty for small footprint at high positions
            height_factor = lz / self.bin_size[2]  # Normalized height
            footprint_factor = 1.0 - (box_footprint / (self.bin_size[0] * self.bin_size[1]))
            
            # Penalty increases with height and decreases with footprint
            penalty += height_factor * footprint_factor * 0.5
            
        return penalty
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional random seed
            options: Optional configuration options
            
        Returns:
            Initial observation, info dictionary
        """
        # Call base class reset
        observation, info = super().reset(seed=seed, options=options)
        
        # Reset any internal state in reliability modules if needed
        # (Currently no state to reset in the modules)
        
        return observation, info
        
    def _calculate_terminal_reward(self) -> float:
        """
        Calculate terminal reward based on final space utilization.
        
        This reward is added when the episode ends to encourage
        maximizing overall space utilization (Requirement 4.4).
        
        Returns:
            Terminal reward value
        """
        final_utilization = self.space.get_ratio()
        
        # Terminal reward is proportional to final utilization
        # Scale by 100 to make it significant
        terminal_reward = final_utilization * 100
        
        return terminal_reward

