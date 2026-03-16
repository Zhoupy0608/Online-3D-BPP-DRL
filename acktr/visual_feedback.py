"""
Visual Feedback Module for Reliable Robot Packing

This module provides functionality to process 3D camera point cloud data,
extract box positions, and update the height map based on actual measured
positions to correct for placement errors.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from acktr.performance_optimizer import get_profiler


@dataclass
class Box:
    """
    Represents a detected box in the container.
    
    Attributes:
        x: Length dimension of the box
        y: Width dimension of the box
        z: Height dimension of the box
        lx: X position in container (left edge)
        ly: Y position in container (front edge)
        lz: Z position in container (bottom edge)
        rotated: Whether box is rotated (optional)
    """
    x: int
    y: int
    z: int
    lx: int
    ly: int
    lz: int
    rotated: bool = False


@dataclass
class CameraConfig:
    """
    Configuration for 3D camera system.
    
    Attributes:
        intrinsics: 3x3 camera intrinsic matrix
        extrinsics: 4x4 camera extrinsic matrix (world to camera transform)
        resolution: (width, height) in pixels
        depth_range: (min_depth, max_depth) in meters
    """
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    resolution: Tuple[int, int]
    depth_range: Tuple[float, float]


class VisualFeedbackModule:
    """
    Processes 3D camera data to detect actual box positions and update height map.
    
    This module bridges the gap between predicted and actual box positions by
    using visual feedback from 3D cameras. It can operate in two modes:
    1. Simulation mode: Uses mock/synthetic point clouds for testing
    2. Deployment mode: Uses real camera data from physical setup
    
    Supports both single-camera and multi-camera setups for improved coverage
    and occlusion handling.
    """
    
    def __init__(self, 
                 camera_config: Optional[CameraConfig] = None,
                 camera_configs: Optional[List[CameraConfig]] = None,
                 container_size: Tuple[int, int, int] = (10, 10, 10),
                 grid_size: Tuple[int, int] = (10, 10),
                 simulation_mode: bool = True):
        """
        Initialize the visual feedback module.
        
        Args:
            camera_config: Configuration for single 3D camera (None for simulation)
            camera_configs: List of configurations for multiple cameras (for multi-camera setup)
            container_size: Tuple (width, length, height) of container
            grid_size: Discretization of height map (width_cells, length_cells)
            simulation_mode: If True, use mock capture; if False, use real camera
        """
        # Support both single camera and multi-camera configurations
        if camera_configs is not None:
            self.camera_configs = camera_configs
            self.multi_camera = True
        elif camera_config is not None:
            self.camera_configs = [camera_config]
            self.multi_camera = False
        else:
            self.camera_configs = []
            self.multi_camera = False
            
        # Legacy support for single camera_config attribute
        self.camera_config = camera_config
        
        self.container_size = container_size
        self.grid_size = grid_size
        self.simulation_mode = simulation_mode
        
    def capture_point_cloud(self) -> np.ndarray:
        """
        Capture point cloud from 3D camera(s).
        
        In simulation mode, returns an empty point cloud (mock).
        In deployment mode, would interface with real camera SDK.
        For multi-camera setups, captures from all cameras and merges them.
        
        Returns:
            Point cloud as Nx3 numpy array (x, y, z coordinates) in container frame
        """
        if self.simulation_mode:
            # Mock capture for simulation - return empty point cloud
            # In real implementation, this would be replaced with actual data
            return np.empty((0, 3), dtype=np.float32)
        else:
            # Real camera capture would go here
            # This would use Open3D or camera SDK to capture actual data
            raise NotImplementedError("Real camera capture not implemented yet")
    
    def capture_point_clouds_multi_camera(self) -> List[np.ndarray]:
        """
        Capture point clouds from multiple cameras simultaneously.
        
        In simulation mode, returns empty point clouds (mock).
        In deployment mode, would interface with real camera SDK for each camera.
        
        Returns:
            List of point clouds, one per camera, each as Nx3 numpy array
        """
        if self.simulation_mode:
            # Mock capture for simulation - return empty point clouds
            return [np.empty((0, 3), dtype=np.float32) for _ in self.camera_configs]
        else:
            # Real multi-camera capture would go here
            # This would use Open3D or camera SDK to capture from each camera
            raise NotImplementedError("Real multi-camera capture not implemented yet")
    
    def transform_point_cloud_to_common_frame(self, 
                                              point_cloud: np.ndarray,
                                              camera_config: CameraConfig) -> np.ndarray:
        """
        Transform point cloud from camera coordinate frame to common container frame.
        
        Uses the camera's extrinsic parameters to transform points from the camera's
        local coordinate system to the world/container coordinate system.
        
        Args:
            point_cloud: Nx3 array of points in camera coordinates
            camera_config: Camera configuration with extrinsic parameters
            
        Returns:
            Nx3 array of points in container/world coordinates
        """
        if len(point_cloud) == 0:
            return point_cloud
            
        # Convert to homogeneous coordinates
        n_points = point_cloud.shape[0]
        # Check for valid input
        if not np.all(np.isfinite(point_cloud)):
            return point_cloud.astype(np.float32)
        
        homogeneous = np.ones((n_points, 4), dtype=np.float64)  # Use float64 for better numerical stability
        homogeneous[:, :3] = point_cloud.astype(np.float64)
        
        # Apply inverse of extrinsic matrix (camera to world)
        try:
            extrinsics = camera_config.extrinsics.astype(np.float64)
            # Check if matrix is valid
            if not np.all(np.isfinite(extrinsics)):
                return point_cloud.astype(np.float32)
            
            # Check matrix shape
            if extrinsics.shape != (4, 4):
                return point_cloud.astype(np.float32)
            
            # Check determinant to avoid singular matrices
            try:
                with np.errstate(all='ignore'):  # Suppress warnings
                    det = np.linalg.det(extrinsics)
                if not np.isfinite(det) or abs(det) < 1e-6:  # More conservative threshold
                    return point_cloud.astype(np.float32)
            except:
                return point_cloud.astype(np.float32)
            
            # Use more robust inversion with error handling
            try:
                with np.errstate(all='ignore'):  # Suppress warnings
                    extrinsics_inv = np.linalg.inv(extrinsics)
            except:
                # Fallback: try pseudo-inverse
                try:
                    with np.errstate(all='ignore'):
                        extrinsics_inv = np.linalg.pinv(extrinsics)
                except:
                    return point_cloud.astype(np.float32)
            
            # Check for NaN or Inf in the inverse
            if not np.all(np.isfinite(extrinsics_inv)):
                return point_cloud.astype(np.float32)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, Exception):
            # If matrix is singular or invalid, return original points
            return point_cloud.astype(np.float32)
        
        try:
            transformed = (extrinsics_inv @ homogeneous.T).T
            # Check for NaN or Inf in result
            if not np.all(np.isfinite(transformed)):
                return point_cloud.astype(np.float32)
        except (ValueError, FloatingPointError, RuntimeWarning):
            return point_cloud.astype(np.float32)
        
        # Convert back to 3D coordinates
        return transformed[:, :3].astype(np.float32)
    
    def merge_point_clouds(self, point_clouds: List[np.ndarray]) -> np.ndarray:
        """
        Merge multiple point clouds into a single unified point cloud.
        
        All input point clouds should already be in the same coordinate frame
        (typically the container/world frame). This method simply concatenates
        all points into a single array.
        
        Args:
            point_clouds: List of Nx3 point cloud arrays in common coordinate frame
            
        Returns:
            Single merged point cloud containing all points from all inputs
        """
        # Filter out empty point clouds
        non_empty_clouds = [pc for pc in point_clouds if len(pc) > 0]
        
        if len(non_empty_clouds) == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        # Concatenate all point clouds
        merged = np.vstack(non_empty_clouds)
        
        return merged.astype(np.float32)
    
    def capture_and_merge_multi_camera(self) -> np.ndarray:
        """
        Capture point clouds from all cameras and merge them into unified cloud.
        
        This is the main entry point for multi-camera operation. It:
        1. Captures point clouds from all cameras
        2. Transforms each to the common container coordinate frame
        3. Merges all transformed clouds into a single point cloud
        
        Returns:
            Merged point cloud in container coordinates as Nx3 array
        """
        # Capture from all cameras
        point_clouds = self.capture_point_clouds_multi_camera()
        
        # Transform each point cloud to common frame
        transformed_clouds = []
        for pc, camera_config in zip(point_clouds, self.camera_configs):
            if len(pc) > 0:
                transformed = self.transform_point_cloud_to_common_frame(pc, camera_config)
                transformed_clouds.append(transformed)
            else:
                transformed_clouds.append(pc)
        
        # Merge all transformed point clouds
        merged = self.merge_point_clouds(transformed_clouds)
        
        return merged
            
    def process_point_cloud(self, point_cloud: np.ndarray) -> List[Box]:
        """
        Extract box positions from point cloud using region growing and orthogonal fitting.
        
        This method implements a simplified box extraction algorithm:
        1. Project point cloud to container coordinate system
        2. Segment point cloud into regions (region growing)
        3. Fit orthogonal bounding boxes to each region
        4. Return detected boxes with positions
        
        Args:
            point_cloud: Nx3 numpy array of 3D points
            
        Returns:
            List of detected Box objects with positions and dimensions
        """
        profiler = get_profiler()
        
        with profiler.profile("point_cloud_processing_total"):
            if len(point_cloud) == 0:
                return []
                
            # Step 1: Project to container coordinate system
            with profiler.profile("point_cloud_projection"):
                projected_points = self._project_to_container_coords(point_cloud)
            
            # Step 2: Segment into regions using simple clustering
            with profiler.profile("point_cloud_segmentation"):
                regions = self._segment_regions(projected_points)
            
            # Step 3: Fit bounding boxes to each region
            boxes = []
            with profiler.profile("bounding_box_fitting"):
                for region_points in regions:
                    box = self._fit_bounding_box(region_points)
                    if box is not None:
                        boxes.append(box)
                    
            return boxes
        
    def _project_to_container_coords(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Project point cloud from camera coordinates to container coordinates.
        
        Args:
            point_cloud: Nx3 array of points in camera coordinates
            
        Returns:
            Nx3 array of points in container coordinates
        """
        if self.camera_config is None:
            # No transformation needed in simulation mode
            return point_cloud.copy()
        
        # Use the new transformation method
        return self.transform_point_cloud_to_common_frame(point_cloud, self.camera_config)
        
    def _downsample_point_cloud(self, points: np.ndarray, 
                                voxel_size: float = 0.1) -> np.ndarray:
        """
        Downsample point cloud using voxel grid filtering for performance.
        
        Args:
            points: Nx3 array of points
            voxel_size: Size of voxel grid cells
            
        Returns:
            Downsampled point cloud
        """
        if len(points) == 0:
            return points
            
        # Simple voxel grid downsampling
        # Quantize points to voxel grid and keep one point per voxel
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # Use dictionary to keep unique voxels
        unique_voxels = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in unique_voxels:
                unique_voxels[key] = points[i]
                
        downsampled = np.array(list(unique_voxels.values()))
        return downsampled
    
    def _segment_regions(self, points: np.ndarray, 
                        distance_threshold: float = 0.5,
                        max_points_for_full_processing: int = 10000) -> List[np.ndarray]:
        """
        Segment point cloud into regions using simple spatial clustering.
        
        Args:
            points: Nx3 array of points
            distance_threshold: Maximum distance for points to be in same region
            max_points_for_full_processing: Downsample if more points than this
            
        Returns:
            List of point arrays, one per region
        """
        if len(points) == 0:
            return []
            
        # Downsample if point cloud is too large for performance
        original_points = points
        if len(points) > max_points_for_full_processing:
            points = self._downsample_point_cloud(points, voxel_size=0.1)
            
        # Simple region growing based on spatial proximity
        # This is a simplified version - real implementation would use
        # more sophisticated algorithms like RANSAC or region growing
        
        regions = []
        unassigned = set(range(len(points)))
        
        while unassigned:
            # Start new region with first unassigned point
            seed_idx = next(iter(unassigned))
            region_indices = {seed_idx}
            unassigned.remove(seed_idx)
            
            # Grow region by adding nearby points
            changed = True
            while changed:
                changed = False
                to_add = set()
                
                for idx in list(unassigned):
                    point = points[idx]
                    # Check distance to any point in current region
                    for region_idx in region_indices:
                        region_point = points[region_idx]
                        dist = np.linalg.norm(point - region_point)
                        if dist < distance_threshold:
                            to_add.add(idx)
                            changed = True
                            break
                            
                region_indices.update(to_add)
                unassigned -= to_add
                
            # Add region if it has enough points
            if len(region_indices) >= 8:  # Minimum points for a box
                region_points = points[list(region_indices)]
                regions.append(region_points)
                
        return regions
        
    def _fit_bounding_box(self, points: np.ndarray) -> Optional[Box]:
        """
        Fit an axis-aligned bounding box to a set of points.
        
        Args:
            points: Nx3 array of points in a region
            
        Returns:
            Box object with fitted dimensions and position, or None if invalid
        """
        if len(points) < 8:
            return None
            
        # Compute axis-aligned bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Extract box parameters
        lx = int(round(min_coords[0]))
        ly = int(round(min_coords[1]))
        lz = int(round(min_coords[2]))
        
        x = int(round(max_coords[0] - min_coords[0]))
        y = int(round(max_coords[1] - min_coords[1]))
        z = int(round(max_coords[2] - min_coords[2]))
        
        # Validate box dimensions
        if x <= 0 or y <= 0 or z <= 0:
            return None
            
        # Check if box is within container bounds
        width, length, height = self.container_size
        if lx < 0 or ly < 0 or lz < 0:
            return None
        if lx + x > width or ly + y > length or lz + z > height:
            return None
            
        return Box(x=x, y=y, z=z, lx=lx, ly=ly, lz=lz)
        
    def update_height_map(self, 
                         height_map: np.ndarray,
                         boxes: List[Box]) -> np.ndarray:
        """
        Update height map based on detected box positions.
        
        For each detected box, updates the height map grid cells covered by
        the box's footprint to reflect the actual measured height.
        
        Args:
            height_map: Current height map (width x length)
            boxes: List of detected boxes with positions
            
        Returns:
            Updated height map with same shape as input
        """
        # Create a copy to avoid modifying the original
        updated_map = height_map.copy()
        
        width, length = height_map.shape
        container_width, container_length, _ = self.container_size
        
        # Calculate scaling factors from container to grid coordinates
        scale_x = width / container_width
        scale_y = length / container_length
        
        for box in boxes:
            # Convert box coordinates to grid coordinates
            grid_x_min = int(box.lx * scale_x)
            grid_x_max = int((box.lx + box.x) * scale_x)
            grid_y_min = int(box.ly * scale_y)
            grid_y_max = int((box.ly + box.y) * scale_y)
            
            # Clamp to grid boundaries
            grid_x_min = max(0, min(grid_x_min, width))
            grid_x_max = max(0, min(grid_x_max, width))
            grid_y_min = max(0, min(grid_y_min, length))
            grid_y_max = max(0, min(grid_y_max, length))
            
            # Skip if invalid region
            if grid_x_min >= grid_x_max or grid_y_min >= grid_y_max:
                continue
                
            # Update height map for this box's footprint
            # Use maximum to ensure monotonicity (heights never decrease)
            box_top_height = box.lz + box.z
            current_heights = updated_map[grid_x_min:grid_x_max, grid_y_min:grid_y_max]
            updated_map[grid_x_min:grid_x_max, grid_y_min:grid_y_max] = np.maximum(
                current_heights,
                box_top_height
            )
            
        return updated_map

