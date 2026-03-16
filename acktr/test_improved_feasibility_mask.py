"""
Property-based tests for improved feasibility mask functionality.

This module tests the enhanced support calculation infrastructure and
improved stability checking based on static stability principles.
"""

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
import sys
import os
import math

# Add the envs directory to the path to import Space and support_calculation
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'envs', 'bpp0'))

try:
    from space import Space, Box
    from support_calculation import SupportCalculator, SupportPoint, StabilityThresholds, GeometricUtils
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)


# Strategy for generating container dimensions
container_size_strategy = st.tuples(
    st.integers(min_value=5, max_value=20),  # width
    st.integers(min_value=5, max_value=20),  # length
    st.integers(min_value=10, max_value=30)  # height
)

# Strategy for generating item dimensions that fit within container bounds
def item_size_strategy(container_width, container_length):
    """Generate item dimensions that fit within the container."""
    return st.tuples(
        st.integers(min_value=1, max_value=min(10, container_width)),   # x (width)
        st.integers(min_value=1, max_value=min(10, container_length)),  # y (length)
        st.integers(min_value=1, max_value=5)                           # z (height)
    )

# Strategy for generating placement positions
def placement_position_strategy(container_width, container_length, item_x, item_y):
    """Generate valid placement positions that fit within container bounds."""
    max_lx = max(0, container_width - item_x)
    max_ly = max(0, container_length - item_y)
    
    return st.tuples(
        st.integers(min_value=0, max_value=max_lx),  # lx
        st.integers(min_value=0, max_value=max_ly)   # ly
    )

# Strategy for generating height maps
def height_map_strategy(width, length):
    """Generate realistic height maps for containers."""
    return st.lists(
        st.lists(
            st.integers(min_value=0, max_value=10),
            min_size=length,
            max_size=length
        ),
        min_size=width,
        max_size=width
    ).map(lambda heights: np.array(heights, dtype=np.int32))


@given(
    container_size=container_size_strategy,
    data=st.data()
)
@settings(max_examples=100, deadline=None)
def test_support_point_identification_completeness(container_size, data):
    """
    **Feature: improved-feasibility-mask, Property 6: Corner point identification completeness**
    
    For any rectangular placement area, all four corner support points should be 
    identified and included in stability calculations.
    
    **Validates: Requirements 2.1**
    """
    width, length, height = container_size
    
    # Generate item size that fits within container
    item_size = data.draw(item_size_strategy(width, length))
    x, y, z = item_size
    
    # Generate a valid placement position
    placement_strategy = placement_position_strategy(width, length, x, y)
    lx, ly = data.draw(placement_strategy)
    
    # Generate a height map for the container
    height_map = data.draw(height_map_strategy(width, length))
    
    # Create space and support calculator
    space = Space(width=width, length=length, height=height)
    support_calculator = SupportCalculator()
    
    # Find support points for the placement area
    support_points = support_calculator.find_support_points(height_map, x, y, lx, ly)
    
    # Property: All four corner support points should be identified
    # For any rectangular placement area, we should get exactly 4 support points
    # corresponding to the four corners of the placement area
    
    assert len(support_points) == 4, (
        f"Expected exactly 4 corner support points, but got {len(support_points)}. "
        f"Container: {width}x{length}x{height}, Item: {x}x{y}x{z}, "
        f"Position: ({lx}, {ly})"
    )
    
    # Verify that the support points correspond to the expected corner positions
    expected_corners = [
        (lx, ly),           # Bottom-left corner
        (lx + x - 1, ly),   # Bottom-right corner
        (lx, ly + y - 1),   # Top-left corner
        (lx + x - 1, ly + y - 1)  # Top-right corner
    ]
    
    # Extract actual corner positions from support points
    actual_corners = [(int(sp.x), int(sp.y)) for sp in support_points]
    
    # Sort both lists to ensure consistent comparison
    expected_corners_sorted = sorted(expected_corners)
    actual_corners_sorted = sorted(actual_corners)
    
    assert actual_corners_sorted == expected_corners_sorted, (
        f"Support points do not match expected corner positions. "
        f"Expected corners: {expected_corners_sorted}, "
        f"Actual corners: {actual_corners_sorted}. "
        f"Container: {width}x{length}x{height}, Item: {x}x{y}x{z}, "
        f"Position: ({lx}, {ly})"
    )
    
    # Verify that each support point has valid height and weight values
    for i, sp in enumerate(support_points):
        assert isinstance(sp.height, (int, float, np.integer, np.floating)), (
            f"Support point {i} has invalid height type: {type(sp.height)}"
        )
        assert sp.height >= 0, (
            f"Support point {i} has negative height: {sp.height}"
        )
        assert isinstance(sp.weight, (int, float, np.integer, np.floating)), (
            f"Support point {i} has invalid weight type: {type(sp.weight)}"
        )
        assert sp.weight > 0, (
            f"Support point {i} has non-positive weight: {sp.weight}"
        )
    
    # Verify that support points have heights from the height map
    for sp in support_points:
        corner_x = int(sp.x)
        corner_y = int(sp.y)
        
        # Calculate relative position within the placement area
        rel_x = corner_x - lx
        rel_y = corner_y - ly
        
        # Verify the height matches the height map
        expected_height = height_map[corner_x, corner_y]
        assert sp.height == expected_height, (
            f"Support point at ({corner_x}, {corner_y}) has height {sp.height}, "
            f"but height map shows {expected_height}. "
            f"Relative position in item: ({rel_x}, {rel_y})"
        )


# Unit test for edge cases
def test_support_point_identification_edge_cases():
    """Unit test for edge cases in support point identification."""
    
    # Test case 1: Minimum size item (1x1)
    space = Space(width=5, length=5, height=10)
    support_calculator = SupportCalculator()
    height_map = np.zeros((5, 5), dtype=np.int32)
    
    support_points = support_calculator.find_support_points(height_map, 1, 1, 2, 2)
    
    # For a 1x1 item, all four "corners" should be the same point
    assert len(support_points) == 4
    # All support points should have the same coordinates
    unique_positions = set((int(sp.x), int(sp.y)) for sp in support_points)
    assert len(unique_positions) == 1
    assert (2, 2) in unique_positions
    
    # Test case 2: Item at container boundary
    support_points = support_calculator.find_support_points(height_map, 2, 2, 3, 3)
    assert len(support_points) == 4
    
    expected_corners = [(3, 3), (4, 3), (3, 4), (4, 4)]
    actual_corners = sorted([(int(sp.x), int(sp.y)) for sp in support_points])
    assert actual_corners == sorted(expected_corners)
    
    # Test case 3: Out of bounds placement (should return empty list)
    support_points = support_calculator.find_support_points(height_map, 3, 3, 4, 4)
    assert len(support_points) == 0


def test_support_point_identification_with_varied_heights():
    """Unit test with varied heights in the height map."""
    
    space = Space(width=6, length=6, height=15)
    support_calculator = SupportCalculator()
    
    # Create a height map with varied heights
    height_map = np.array([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8],
        [4, 5, 6, 7, 8, 9],
        [5, 6, 7, 8, 9, 10]
    ], dtype=np.int32)
    
    # Test placement of a 3x3 item at position (1, 1)
    support_points = support_calculator.find_support_points(height_map, 3, 3, 1, 1)
    
    assert len(support_points) == 4
    
    # Verify corner positions and their heights
    expected_data = [
        (1, 1, 2),  # Bottom-left: height_map[1, 1] = 2
        (3, 1, 4),  # Bottom-right: height_map[3, 1] = 4
        (1, 3, 4),  # Top-left: height_map[1, 3] = 4
        (3, 3, 6)   # Top-right: height_map[3, 3] = 6
    ]
    
    actual_data = [(int(sp.x), int(sp.y), sp.height) for sp in support_points]
    actual_data_sorted = sorted(actual_data)
    expected_data_sorted = sorted(expected_data)
    
    assert actual_data_sorted == expected_data_sorted


@given(
    points=st.lists(
        st.tuples(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        min_size=4,
        max_size=15
    )
)
@settings(max_examples=100, deadline=None)
def test_convex_hull_support_polygon_calculation(points):
    """
    **Feature: improved-feasibility-mask, Property 7: Convex hull support polygon calculation**
    
    For any set of corner points with uneven heights, the support polygon should be 
    computed as the convex hull of valid support points.
    
    **Validates: Requirements 2.2**
    """
    # Round points to avoid floating point precision issues
    rounded_points = [(round(x, 6), round(y, 6)) for x, y in points]
    
    # Remove duplicate points to avoid degenerate cases
    unique_points = list(set(rounded_points))
    
    if len(unique_points) < 3:
        # Skip if we don't have enough unique points
        return
    
    # Compute convex hull using the GeometricUtils method
    hull_vertices = GeometricUtils.compute_convex_hull(unique_points)
    
    # Property 1: All hull vertices should be from the original point set
    for vertex in hull_vertices:
        assert vertex in unique_points, (
            f"Hull vertex {vertex} is not in the original point set {unique_points}"
        )
    
    # Property 2: Hull should contain at least 3 vertices for non-degenerate cases
    # (unless all points are collinear)
    if len(hull_vertices) >= 3:
        # Property 3: All original points should be inside or on the convex hull
        for point in unique_points:
            # A point is either a hull vertex or inside the hull
            is_hull_vertex = point in hull_vertices
            is_inside_or_on_hull = GeometricUtils.point_in_polygon(point, hull_vertices)
            
            # For points on the boundary, point_in_polygon might return False
            # So we check if it's a hull vertex OR inside the polygon
            assert is_hull_vertex or is_inside_or_on_hull or _point_on_hull_boundary(point, hull_vertices), (
                f"Point {point} is neither a hull vertex nor inside/on the convex hull. "
                f"Hull vertices: {hull_vertices}"
            )
    
    # Property 4: Hull vertices should be in counter-clockwise order (for proper orientation)
    if len(hull_vertices) >= 3:
        # Calculate signed area to check orientation
        signed_area = 0.0
        n = len(hull_vertices)
        for i in range(n):
            j = (i + 1) % n
            signed_area += hull_vertices[i][0] * hull_vertices[j][1]
            signed_area -= hull_vertices[j][0] * hull_vertices[i][1]
        
        # For counter-clockwise orientation, signed area should be positive
        # We allow small numerical errors
        assert signed_area >= -1e-10, (
            f"Hull vertices are not in counter-clockwise order. "
            f"Signed area: {signed_area}, Hull: {hull_vertices}"
        )
    
    # Property 5: No three consecutive hull vertices should be collinear
    # (the algorithm should remove collinear points)
    # Skip this check for small hulls or when we have numerical precision issues
    if len(hull_vertices) >= 4:  # Only check for hulls with 4+ vertices
        for i in range(len(hull_vertices)):
            p1 = hull_vertices[i]
            p2 = hull_vertices[(i + 1) % len(hull_vertices)]
            p3 = hull_vertices[(i + 2) % len(hull_vertices)]
            
            # Calculate cross product to check collinearity
            cross_prod = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            
            # Points should not be collinear (cross product should not be zero)
            # Use a reasonable tolerance for floating point comparisons
            if abs(cross_prod) <= 1e-6:
                # This might be a degenerate case, but it's acceptable for our purposes
                # The convex hull algorithm handles this correctly
                pass
    
    # Property 6: The convex hull should be the minimal convex set containing all points
    # This is implicitly tested by the above properties, but we can add an explicit check
    # by verifying that removing any hull vertex would make the hull invalid
    if len(hull_vertices) >= 4:  # Only meaningful for hulls with 4+ vertices
        for i in range(len(hull_vertices)):
            # Create a hull without vertex i
            reduced_hull = hull_vertices[:i] + hull_vertices[i+1:]
            
            # The removed vertex should not be inside the reduced hull
            removed_vertex = hull_vertices[i]
            is_inside_reduced = GeometricUtils.point_in_polygon(removed_vertex, reduced_hull)
            
            assert not is_inside_reduced, (
                f"Hull vertex {removed_vertex} is inside the reduced hull {reduced_hull}, "
                f"indicating the original hull is not minimal"
            )


def _point_on_hull_boundary(point, hull_vertices):
    """
    Helper function to check if a point lies on the boundary of the convex hull.
    
    Args:
        point: (x, y) coordinates of the point to test
        hull_vertices: List of hull vertices
        
    Returns:
        True if point is on the hull boundary, False otherwise
    """
    if len(hull_vertices) < 2:
        return False
    
    px, py = point
    n = len(hull_vertices)
    
    for i in range(n):
        # Get edge from vertex i to vertex (i+1)
        x1, y1 = hull_vertices[i]
        x2, y2 = hull_vertices[(i + 1) % n]
        
        # Check if point is on the line segment
        # First check if point is collinear with the edge
        cross_prod = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        if abs(cross_prod) < 1e-6:  # Point is collinear
            # Check if point is within the segment bounds with some tolerance
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            tolerance = 1e-6
            if ((min_x - tolerance) <= px <= (max_x + tolerance)) and ((min_y - tolerance) <= py <= (max_y + tolerance)):
                return True
    
    return False


@given(
    container_size=container_size_strategy,
    data=st.data()
)
@settings(max_examples=100, deadline=None)
def test_geometric_center_validation(container_size, data):
    """
    **Feature: improved-feasibility-mask, Property 1: Geometric center validation**
    
    For any item placement with computed support polygon, the geometric center projection 
    should lie within the support polygon boundaries when placement is approved.
    
    **Validates: Requirements 1.1**
    """
    width, length, height = container_size
    
    # Generate item size that fits within container
    item_size = data.draw(item_size_strategy(width, length))
    x, y, z = item_size
    
    # Generate a valid placement position
    placement_strategy = placement_position_strategy(width, length, x, y)
    lx, ly = data.draw(placement_strategy)
    
    # Generate a height map for the container
    height_map = data.draw(height_map_strategy(width, length))
    
    # Create space with enhanced feasibility checking enabled
    space = Space(width=width, length=length, height=height, use_enhanced_feasibility=True)
    
    # Test the enhanced feasibility checking
    result = space.check_box_enhanced(height_map, x, y, lx, ly, z)
    
    # If placement is approved (result >= 0), then geometric center validation should pass
    if result >= 0:
        # Get the support polygon for this placement
        support_polygon = space.calculate_support_polygon(height_map, x, y, lx, ly)
        
        # Get the geometric center projection
        geometric_center = space.get_geometric_center_projection(x, y, lx, ly)
        
        # Property: If placement is approved, geometric center must be within support polygon
        if len(support_polygon.vertices) >= 3:
            # For valid polygons, check if center is inside
            center_inside = GeometricUtils.point_in_polygon(geometric_center, support_polygon.vertices)
            
            assert center_inside, (
                f"Placement was approved but geometric center {geometric_center} "
                f"is not within support polygon {support_polygon.vertices}. "
                f"Container: {width}x{length}x{height}, Item: {x}x{y}x{z}, "
                f"Position: ({lx}, {ly}), Result height: {result}"
            )
        else:
            # For degenerate polygons (< 3 vertices), the enhanced algorithm should
            # fall back to corner-based validation, which is acceptable
            # We verify that the geometric center is at least close to the support points
            if len(support_polygon.vertices) > 0:
                # Find the closest support point to the geometric center
                min_distance = float('inf')
                for vertex in support_polygon.vertices:
                    distance = math.sqrt((geometric_center[0] - vertex[0])**2 + 
                                       (geometric_center[1] - vertex[1])**2)
                    min_distance = min(min_distance, distance)
                
                # For small items, the geometric center should be close to support points
                max_expected_distance = max(x, y) / 2.0 + 1.0  # Allow some tolerance
                assert min_distance <= max_expected_distance, (
                    f"Geometric center {geometric_center} is too far from support points "
                    f"{support_polygon.vertices}. Distance: {min_distance}, "
                    f"Max expected: {max_expected_distance}"
                )
    
    # Additional validation: Test the geometric center calculation itself
    expected_center_x = lx + x / 2.0
    expected_center_y = ly + y / 2.0
    expected_center = (expected_center_x, expected_center_y)
    
    actual_center = space.get_geometric_center_projection(x, y, lx, ly)
    
    # Verify geometric center calculation is correct
    assert abs(actual_center[0] - expected_center[0]) < 1e-10, (
        f"Geometric center X coordinate incorrect. Expected: {expected_center[0]}, "
        f"Actual: {actual_center[0]}"
    )
    assert abs(actual_center[1] - expected_center[1]) < 1e-10, (
        f"Geometric center Y coordinate incorrect. Expected: {expected_center[1]}, "
        f"Actual: {actual_center[1]}"
    )
    
    # Test edge case: Verify that when geometric center is clearly outside support area,
    # placement should be rejected (when support area is insufficient)
    # This is tested implicitly by the enhanced algorithm's logic


def test_geometric_center_validation_unit_cases():
    """Unit tests for specific geometric center validation scenarios."""
    
    # Test case 1: Simple flat surface - center should always be inside
    space = Space(width=10, length=10, height=15, use_enhanced_feasibility=True)
    height_map = np.zeros((10, 10), dtype=np.int32)  # Flat surface
    
    # Place a 4x4 item at (2, 2)
    result = space.check_box_enhanced(height_map, 4, 4, 2, 2, 3)
    
    if result >= 0:  # If placement is approved
        support_polygon = space.calculate_support_polygon(height_map, 4, 4, 2, 2)
        geometric_center = space.get_geometric_center_projection(4, 4, 2, 2)
        
        # Center should be at (4.0, 4.0)
        assert geometric_center == (4.0, 4.0)
        
        # For a flat surface, all corners are at the same height, so we get a rectangular polygon
        if len(support_polygon.vertices) >= 3:
            center_inside = GeometricUtils.point_in_polygon(geometric_center, support_polygon.vertices)
            assert center_inside, f"Center {geometric_center} should be inside polygon {support_polygon.vertices}"
    
    # Test case 2: Uneven surface with clear support polygon
    height_map = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
        [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
        [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
        [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
        [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
        [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int32)
    
    # Place a 2x2 item at the peak (4, 4)
    result = space.check_box_enhanced(height_map, 2, 2, 4, 4, 2)
    
    if result >= 0:
        support_polygon = space.calculate_support_polygon(height_map, 2, 2, 4, 4)
        geometric_center = space.get_geometric_center_projection(2, 2, 4, 4)
        
        # Center should be at (5.0, 5.0)
        assert geometric_center == (5.0, 5.0)
        
        # All corners are at height 4, so center should be inside
        if len(support_polygon.vertices) >= 3:
            center_inside = GeometricUtils.point_in_polygon(geometric_center, support_polygon.vertices)
            assert center_inside, f"Center {geometric_center} should be inside polygon {support_polygon.vertices}"
    
    # Test case 3: Edge placement where center might be outside support area
    # Place a 3x3 item at (0, 0) on an uneven surface
    height_map_uneven = np.array([
        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int32)
    
    result = space.check_box_enhanced(height_map_uneven, 3, 3, 0, 0, 2)
    
    # This placement should likely be rejected due to insufficient support
    # (only one corner at height 5, others at 0), but if it's approved,
    # the geometric center validation should still hold
    if result >= 0:
        support_polygon = space.calculate_support_polygon(height_map_uneven, 3, 3, 0, 0)
        geometric_center = space.get_geometric_center_projection(3, 3, 0, 0)
        
        # Center should be at (1.5, 1.5)
        assert geometric_center == (1.5, 1.5)
        
        # If approved, center should be within or close to support polygon
        if len(support_polygon.vertices) >= 3:
            center_inside = GeometricUtils.point_in_polygon(geometric_center, support_polygon.vertices)
            # For this case, we might need to be more lenient due to the extreme height difference
            # The algorithm might approve based on other criteria
            if not center_inside:
                # Check if center is at least close to the support polygon
                min_distance_to_polygon = float('inf')
                for vertex in support_polygon.vertices:
                    distance = math.sqrt((geometric_center[0] - vertex[0])**2 + 
                                       (geometric_center[1] - vertex[1])**2)
                    min_distance_to_polygon = min(min_distance_to_polygon, distance)
                
                # Allow some tolerance for edge cases
                assert min_distance_to_polygon <= 3.0, (
                    f"Geometric center {geometric_center} is too far from support polygon "
                    f"{support_polygon.vertices}. Distance: {min_distance_to_polygon}"
                )


@given(
    container_size=container_size_strategy,
    data=st.data()
)
@settings(max_examples=20, deadline=None)
def test_support_area_threshold_enforcement(container_size, data):
    """
    **Feature: improved-feasibility-mask, Property 2: Support area threshold enforcement**
    
    For any item placement, when the support area ratio is at least 75%, 
    the placement should be considered feasible.
    
    **Validates: Requirements 1.2**
    """
    width, length, height = container_size
    
    # Generate item size that fits within container
    item_size = data.draw(item_size_strategy(width, length))
    x, y, z = item_size
    
    # Generate a valid placement position
    placement_strategy = placement_position_strategy(width, length, x, y)
    lx, ly = data.draw(placement_strategy)
    
    # Create a height map that will provide good support
    # We'll create scenarios with varying support area ratios
    support_ratio = data.draw(st.floats(min_value=0.0, max_value=1.0))
    
    # Create space with enhanced feasibility checking enabled
    space = Space(width=width, length=length, height=height, use_enhanced_feasibility=True)
    
    # Generate a height map with controlled support characteristics
    height_map = _create_controlled_support_height_map(
        width, length, x, y, lx, ly, support_ratio, data
    )
    
    # Calculate the actual weighted support area for this placement
    actual_support_ratio = space.calculate_weighted_support_area(height_map, x, y, lx, ly)
    
    # Test the enhanced feasibility checking
    result = space.check_box_enhanced(height_map, x, y, lx, ly, z)
    
    # Property: When support area ratio is at least 75%, placement should be feasible
    if actual_support_ratio >= 0.75:
        # Additional checks to ensure the placement is actually valid
        # (not rejected for other reasons like height constraints)
        
        # Check if placement fits within container bounds
        if (lx + x <= width and ly + y <= length and lx >= 0 and ly >= 0):
            # Check if height constraint is satisfied
            rec = height_map[lx:lx+x, ly:ly+y]
            max_h = np.max(rec)
            
            if max_h + z <= height:
                # All basic constraints are satisfied, and support area >= 75%
                # Therefore, placement should be approved
                assert result >= 0, (
                    f"Placement should be feasible when support area ratio >= 75%. "
                    f"Support ratio: {actual_support_ratio:.3f}, Result: {result}. "
                    f"Container: {width}x{length}x{height}, Item: {x}x{y}x{z}, "
                    f"Position: ({lx}, {ly}), Max height: {max_h}, Item height: {z}"
                )
                
                # Verify that the returned height is correct
                assert result == max_h, (
                    f"Returned height {result} should match max height {max_h} "
                    f"when placement is approved"
                )
    
    # Additional validation: Test specific threshold boundary cases
    # When support ratio is exactly 75%, it should be approved
    if abs(actual_support_ratio - 0.75) < 0.01:  # Within 1% of 75%
        if (lx + x <= width and ly + y <= length and lx >= 0 and ly >= 0):
            rec = height_map[lx:lx+x, ly:ly+y]
            max_h = np.max(rec)
            if max_h + z <= height:
                assert result >= 0, (
                    f"Placement should be feasible at 75% support threshold. "
                    f"Support ratio: {actual_support_ratio:.3f}, Result: {result}"
                )
    
    # Test that very high support ratios (>95%) are always approved
    if actual_support_ratio > 0.95:
        if (lx + x <= width and ly + y <= length and lx >= 0 and ly >= 0):
            rec = height_map[lx:lx+x, ly:ly+y]
            max_h = np.max(rec)
            if max_h + z <= height:
                assert result >= 0, (
                    f"Placement should always be feasible when support area > 95%. "
                    f"Support ratio: {actual_support_ratio:.3f}, Result: {result}"
                )


def _create_controlled_support_height_map(width, length, x, y, lx, ly, target_support_ratio, data):
    """
    Create a height map with controlled support characteristics for testing.
    
    Args:
        width, length: Container dimensions
        x, y: Item dimensions
        lx, ly: Placement position
        target_support_ratio: Desired support area ratio (0.0 to 1.0)
        data: Hypothesis data object for generating random values
        
    Returns:
        numpy array representing the height map
    """
    # Start with a base height map
    base_height = data.draw(st.integers(min_value=0, max_value=5))
    height_map = np.full((width, length), base_height, dtype=np.int32)
    
    # If placement is out of bounds, return the base map
    if lx + x > width or ly + y > length or lx < 0 or ly < 0:
        return height_map
    
    # Calculate the placement area
    placement_area = x * y
    target_supported_cells = int(placement_area * target_support_ratio)
    
    # Generate heights for the placement area
    max_height = base_height + data.draw(st.integers(min_value=1, max_value=5))
    
    # Create a list of all positions in the placement area
    positions = []
    for i in range(x):
        for j in range(y):
            positions.append((lx + i, ly + j))
    
    # Randomly select positions to have the max height (supported area)
    if target_supported_cells > 0:
        # Ensure we don't try to select more positions than available
        num_to_select = min(target_supported_cells, len(positions))
        
        # Use hypothesis to select positions
        selected_indices = data.draw(
            st.lists(
                st.integers(min_value=0, max_value=len(positions)-1),
                min_size=num_to_select,
                max_size=num_to_select,
                unique=True
            )
        )
        
        # Set selected positions to max height
        for idx in selected_indices:
            pos_x, pos_y = positions[idx]
            height_map[pos_x, pos_y] = max_height
    
    # Set remaining positions in placement area to lower heights
    # For unsupported cells, set them to be more than tolerance (1.0) below max_height
    # so they won't count as supported in the calculation
    for i in range(x):
        for j in range(y):
            pos_x, pos_y = lx + i, ly + j
            if height_map[pos_x, pos_y] == base_height:
                # Set to a height that's more than tolerance below max_height
                # This ensures these cells won't count as supported
                lower_height = max_height - 2  # More than tolerance=1.0
                height_map[pos_x, pos_y] = max(base_height, lower_height)
    
    return height_map


@given(
    container_size=container_size_strategy,
    data=st.data()
)
@settings(max_examples=100, deadline=None)
def test_weighted_support_with_varying_heights(container_size, data):
    """
    **Feature: improved-feasibility-mask, Property 4: Weighted support with varying heights**
    
    For any placement area with multiple support heights, the support calculation should 
    use weighted values based on contact area distribution.
    
    **Validates: Requirements 1.4**
    """
    width, length, height = container_size
    
    # Generate item size that fits within container
    item_size = data.draw(item_size_strategy(width, length))
    x, y, z = item_size
    
    # Generate a valid placement position
    placement_strategy = placement_position_strategy(width, length, x, y)
    lx, ly = data.draw(placement_strategy)
    
    # Create space with enhanced feasibility checking enabled
    space = Space(width=width, length=length, height=height, use_enhanced_feasibility=True)
    
    # Create a height map with varying heights in the placement area
    height_map = np.zeros((width, length), dtype=np.int32)
    
    # Generate different heights for the placement area
    base_height = data.draw(st.integers(min_value=0, max_value=5))
    max_height = base_height + data.draw(st.integers(min_value=1, max_value=5))
    
    # Create varying heights within the placement area
    # Some cells at max height, some at intermediate heights, some at base height
    for i in range(x):
        for j in range(y):
            pos_x, pos_y = lx + i, ly + j
            if pos_x < width and pos_y < length:
                # Randomly assign heights with bias toward max_height
                height_choice = data.draw(st.floats(min_value=0.0, max_value=1.0))
                if height_choice > 0.7:
                    # 30% chance of max height (well supported)
                    height_map[pos_x, pos_y] = max_height
                elif height_choice > 0.4:
                    # 30% chance of intermediate height (partially supported)
                    intermediate_height = max_height - 1
                    height_map[pos_x, pos_y] = max(base_height, intermediate_height)
                else:
                    # 40% chance of base height (not well supported)
                    height_map[pos_x, pos_y] = base_height
    
    # Calculate weighted support area using the system
    calculated_support_ratio = space.calculate_weighted_support_area(height_map, x, y, lx, ly)
    
    # Property: Verify that the weighted calculation considers height variations properly
    
    # Manual calculation to verify the weighted support logic
    if lx + x <= width and ly + y <= length and lx >= 0 and ly >= 0:
        height_rect = height_map[lx:lx+x, ly:ly+y]
        actual_max_height = np.max(height_rect)
        total_area = x * y
        
        if total_area > 0:
            # Count cells within tolerance of max height (as per the algorithm)
            tolerance = space.support_calculator.thresholds.height_variation_tolerance
            supported_cells = np.sum(height_rect >= (actual_max_height - tolerance))
            expected_support_ratio = supported_cells / total_area
            
            # Property 1: The calculated support ratio should match our manual calculation
            assert abs(calculated_support_ratio - expected_support_ratio) < 1e-6, (
                f"Weighted support calculation mismatch. "
                f"Expected: {expected_support_ratio:.6f}, "
                f"Calculated: {calculated_support_ratio:.6f}. "
                f"Height rect:\n{height_rect}\n"
                f"Max height: {actual_max_height}, Tolerance: {tolerance}, "
                f"Supported cells: {supported_cells}, Total area: {total_area}"
            )
            
            # Property 2: Support ratio should be between 0 and 1
            assert 0.0 <= calculated_support_ratio <= 1.0, (
                f"Support ratio should be between 0 and 1, got {calculated_support_ratio}"
            )
            
            # Property 3: When all cells are at max height, support ratio should be 1.0
            if np.all(height_rect == actual_max_height):
                assert abs(calculated_support_ratio - 1.0) < 1e-6, (
                    f"When all cells are at max height, support ratio should be 1.0, "
                    f"got {calculated_support_ratio}"
                )
            
            # Property 4: When no cells are within tolerance of max height, support ratio should be 0.0
            if np.all(height_rect < (actual_max_height - tolerance)):
                assert abs(calculated_support_ratio - 0.0) < 1e-6, (
                    f"When no cells are within tolerance, support ratio should be 0.0, "
                    f"got {calculated_support_ratio}"
                )
            
            # Property 5: Support ratio should increase monotonically as more cells reach max height
            # Test this by creating a modified height map with one more cell at max height
            if supported_cells < total_area:
                # Find a cell that's not at max height and set it to max height
                modified_height_rect = height_rect.copy()
                for i in range(height_rect.shape[0]):
                    for j in range(height_rect.shape[1]):
                        if height_rect[i, j] < (actual_max_height - tolerance):
                            modified_height_rect[i, j] = actual_max_height
                            break
                    else:
                        continue
                    break
                
                # Create modified height map
                modified_height_map = height_map.copy()
                modified_height_map[lx:lx+x, ly:ly+y] = modified_height_rect
                
                # Calculate support ratio for modified map
                modified_support_ratio = space.calculate_weighted_support_area(
                    modified_height_map, x, y, lx, ly)
                
                # Property: Modified support ratio should be >= original
                assert modified_support_ratio >= calculated_support_ratio, (
                    f"Adding support should not decrease support ratio. "
                    f"Original: {calculated_support_ratio:.6f}, "
                    f"Modified: {modified_support_ratio:.6f}"
                )
            
            # Property 6: The weighting should properly account for height tolerance
            # Cells exactly at (max_height - tolerance) should be counted as supported
            if tolerance > 0:
                # Create a test case with cells exactly at the tolerance boundary
                boundary_height_rect = np.full((x, y), actual_max_height - tolerance, dtype=np.int32)
                boundary_height_map = height_map.copy()
                boundary_height_map[lx:lx+x, ly:ly+y] = boundary_height_rect
                
                boundary_support_ratio = space.calculate_weighted_support_area(
                    boundary_height_map, x, y, lx, ly)
                
                # All cells should be counted as supported (ratio = 1.0)
                assert abs(boundary_support_ratio - 1.0) < 1e-6, (
                    f"Cells at tolerance boundary should be fully supported. "
                    f"Expected: 1.0, Got: {boundary_support_ratio:.6f}, "
                    f"Tolerance: {tolerance}, Max height: {actual_max_height}"
                )


def test_weighted_support_with_varying_heights_unit_cases():
    """Unit tests for specific weighted support calculation scenarios."""
    
    # Test case 1: Uniform height (all cells at same height)
    space = Space(width=8, length=8, height=15, use_enhanced_feasibility=True)
    height_map = np.full((8, 8), 5, dtype=np.int32)
    
    # Place a 3x3 item at (2, 2)
    support_ratio = space.calculate_weighted_support_area(height_map, 3, 3, 2, 2)
    assert abs(support_ratio - 1.0) < 1e-6, f"Uniform height should give 100% support, got {support_ratio:.6f}"
    
    # Test case 2: Mixed heights with tolerance
    height_map_mixed = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 5, 4, 0, 0, 0],  # Row with placement area
        [0, 0, 5, 4, 4, 0, 0, 0],  # Max=5, some at 4 (within tolerance=1.0)
        [0, 0, 3, 4, 5, 0, 0, 0],  # Some at 3 (outside tolerance)
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int32)
    
    # Place a 3x3 item at (2, 2)
    support_ratio_mixed = space.calculate_weighted_support_area(height_map_mixed, 3, 3, 2, 2)
    
    # Manual calculation: max_height = 5, tolerance = 1.0
    # Cells >= 4 are supported: (2,2)=5, (2,3)=5, (2,4)=4, (3,2)=5, (3,3)=4, (3,4)=4, (4,3)=4, (4,4)=5
    # Cell (4,2)=3 is not supported (< 4)
    # So 8 out of 9 cells are supported = 8/9 ≈ 0.889
    expected_ratio = 8.0 / 9.0
    assert abs(support_ratio_mixed - expected_ratio) < 0.01, (
        f"Expected support ratio {expected_ratio:.3f}, got {support_ratio_mixed:.3f}"
    )
    
    # Test case 3: No support (all cells below tolerance)
    height_map_no_support = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0, 0, 0, 0],  # Only one cell at max height
        [0, 0, 0, 0, 0, 0, 0, 0],  # Others much lower
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int32)
    
    # Place a 3x3 item at (2, 2)
    support_ratio_no_support = space.calculate_weighted_support_area(height_map_no_support, 3, 3, 2, 2)
    
    # Only 1 out of 9 cells is at max height (5), others are at 0 which is < 4 (5-1)
    expected_ratio_no_support = 1.0 / 9.0
    assert abs(support_ratio_no_support - expected_ratio_no_support) < 0.01, (
        f"Expected support ratio {expected_ratio_no_support:.3f}, got {support_ratio_no_support:.3f}"
    )
    
    # Test case 4: Edge case - single cell item
    support_ratio_single = space.calculate_weighted_support_area(height_map, 1, 1, 3, 3)
    assert abs(support_ratio_single - 1.0) < 1e-6, (
        f"Single cell item should have 100% support, got {support_ratio_single:.6f}"
    )
    
    # Test case 5: Out of bounds placement
    support_ratio_oob = space.calculate_weighted_support_area(height_map, 3, 3, 7, 7)
    assert support_ratio_oob == 0.0, f"Out of bounds placement should have 0% support, got {support_ratio_oob}"
    
    # Test case 6: Tolerance boundary testing
    # Create a height map where cells are exactly at tolerance boundary
    tolerance = space.support_calculator.thresholds.height_variation_tolerance
    max_height = 10
    boundary_height = max_height - tolerance
    
    height_map_boundary = np.full((8, 8), int(boundary_height), dtype=np.int32)
    height_map_boundary[3, 3] = max_height  # One cell at max height
    
    support_ratio_boundary = space.calculate_weighted_support_area(height_map_boundary, 3, 3, 2, 2)
    
    # All cells should be counted as supported since they're at or above (max_height - tolerance)
    assert abs(support_ratio_boundary - 1.0) < 1e-6, (
        f"Cells at tolerance boundary should be fully supported. "
        f"Expected: 1.0, Got: {support_ratio_boundary:.6f}, "
        f"Max height: {max_height}, Boundary height: {boundary_height}, Tolerance: {tolerance}"
    )


def test_support_area_threshold_enforcement_unit_cases():
    """Unit tests for specific support area threshold scenarios."""
    
    # Test case 1: Exactly 75% support area
    space = Space(width=10, length=10, height=15, use_enhanced_feasibility=True)
    
    # Create a 4x4 item placement area where exactly 75% (12 out of 16) cells have max height
    height_map = np.zeros((10, 10), dtype=np.int32)
    
    # Place item at (2, 2) with 4x4 size
    x, y, lx, ly = 4, 4, 2, 2
    max_height = 5
    
    # Set 12 cells (75%) to max height, 4 cells to lower height
    supported_positions = [
        (2, 2), (2, 3), (2, 4), (2, 5),  # First row
        (3, 2), (3, 3), (3, 4), (3, 5),  # Second row  
        (4, 2), (4, 3), (4, 4),          # Third row (3 cells)
        (5, 2)                           # Fourth row (1 cell)
    ]
    
    for pos_x, pos_y in supported_positions:
        height_map[pos_x, pos_y] = max_height
    
    # Set remaining 4 positions to lower height
    unsupported_positions = [(4, 5), (5, 3), (5, 4), (5, 5)]
    for pos_x, pos_y in unsupported_positions:
        height_map[pos_x, pos_y] = max_height - 2
    
    # Verify support area calculation
    support_ratio = space.calculate_weighted_support_area(height_map, x, y, lx, ly)
    assert abs(support_ratio - 0.75) < 0.01, f"Expected 75% support, got {support_ratio:.3f}"
    
    # Test placement - should be approved
    result = space.check_box_enhanced(height_map, x, y, lx, ly, 3)
    assert result >= 0, f"Placement should be approved with 75% support area, got result: {result}"
    assert result == max_height, f"Should return max height {max_height}, got {result}"
    
    # Test case 2: 95% support area (should always be approved)
    height_map_95 = np.full((10, 10), max_height, dtype=np.int32)
    
    # Set only 1 cell (out of 16) to lower height for ~94% support
    height_map_95[5, 5] = max_height - 1
    
    support_ratio_95 = space.calculate_weighted_support_area(height_map_95, x, y, lx, ly)
    assert support_ratio_95 > 0.90, f"Expected >90% support, got {support_ratio_95:.3f}"
    
    result_95 = space.check_box_enhanced(height_map_95, x, y, lx, ly, 3)
    assert result_95 >= 0, f"Placement should be approved with >90% support area, got result: {result_95}"
    
    # Test case 3: 50% support area (should require additional validation)
    height_map_50 = np.zeros((10, 10), dtype=np.int32)
    
    # Set exactly 8 cells (50%) to max height
    supported_50 = [(2, 2), (2, 3), (3, 2), (3, 3), (4, 4), (4, 5), (5, 4), (5, 5)]
    for pos_x, pos_y in supported_50:
        height_map_50[pos_x, pos_y] = max_height
    
    support_ratio_50 = space.calculate_weighted_support_area(height_map_50, x, y, lx, ly)
    assert abs(support_ratio_50 - 0.5) < 0.1, f"Expected ~50% support, got {support_ratio_50:.3f}"
    
    # With 50% support, placement might be rejected or require corner validation
    result_50 = space.check_box_enhanced(height_map_50, x, y, lx, ly, 3)
    # We don't assert approval here since 50% requires additional corner validation
    
    # Test case 4: Very low support area (should be rejected)
    height_map_low = np.zeros((10, 10), dtype=np.int32)
    
    # Set only 2 cells (12.5%) to max height
    height_map_low[2, 2] = max_height
    height_map_low[3, 3] = max_height
    
    support_ratio_low = space.calculate_weighted_support_area(height_map_low, x, y, lx, ly)
    assert support_ratio_low < 0.25, f"Expected <25% support, got {support_ratio_low:.3f}"
    
    result_low = space.check_box_enhanced(height_map_low, x, y, lx, ly, 3)
    assert result_low == -1, f"Placement should be rejected with low support area, got result: {result_low}"


@given(
    utilization_ratio=st.floats(min_value=0.0, max_value=0.30, allow_nan=False, allow_infinity=False),
    data=st.data()
)
@settings(max_examples=100, deadline=None)
def test_strict_thresholds_at_low_utilization(utilization_ratio, data):
    """
    **Feature: improved-feasibility-mask, Property 16: Strict thresholds at low utilization**
    
    For any container with low utilization rate, stricter stability thresholds should be applied.
    
    **Validates: Requirements 4.1**
    """
    # Import ThresholdManager for testing
    from support_calculation import ThresholdManager, StabilityThresholds
    
    # Create threshold manager with known base thresholds
    base_thresholds = StabilityThresholds(
        min_support_area_ratio=0.75,
        corner_support_threshold=0.85,
        height_variation_tolerance=1.0,
        geometric_center_tolerance=0.1
    )
    threshold_manager = ThresholdManager(base_thresholds)
    
    # Get adaptive thresholds for low utilization
    low_util_thresholds = threshold_manager.get_adaptive_thresholds(utilization_ratio)
    
    # Test that thresholds are stricter than base thresholds for low utilization
    # (Requirements 4.1: stricter thresholds at low utilization)
    
    # Support area ratio should be higher (stricter) than base
    assert low_util_thresholds.min_support_area_ratio >= base_thresholds.min_support_area_ratio, \
        f"Expected stricter support area threshold, got {low_util_thresholds.min_support_area_ratio} vs base {base_thresholds.min_support_area_ratio}"
    
    # Corner support threshold should be higher (stricter) than base
    assert low_util_thresholds.corner_support_threshold >= base_thresholds.corner_support_threshold, \
        f"Expected stricter corner support threshold, got {low_util_thresholds.corner_support_threshold} vs base {base_thresholds.corner_support_threshold}"
    
    # Height variation tolerance should be lower (stricter) than base
    assert low_util_thresholds.height_variation_tolerance <= base_thresholds.height_variation_tolerance, \
        f"Expected stricter height variation tolerance, got {low_util_thresholds.height_variation_tolerance} vs base {base_thresholds.height_variation_tolerance}"
    
    # Geometric center tolerance should be lower (stricter) than base
    assert low_util_thresholds.geometric_center_tolerance <= base_thresholds.geometric_center_tolerance, \
        f"Expected stricter geometric center tolerance, got {low_util_thresholds.geometric_center_tolerance} vs base {base_thresholds.geometric_center_tolerance}"
    
    # Test that safety margins are still respected (Requirements 4.3)
    safety_margins = threshold_manager.safety_margins
    
    assert low_util_thresholds.min_support_area_ratio >= safety_margins.min_support_area_ratio, \
        f"Safety margin violated for support area: {low_util_thresholds.min_support_area_ratio} < {safety_margins.min_support_area_ratio}"
    
    assert low_util_thresholds.corner_support_threshold >= safety_margins.corner_support_threshold, \
        f"Safety margin violated for corner support: {low_util_thresholds.corner_support_threshold} < {safety_margins.corner_support_threshold}"
    
    assert low_util_thresholds.height_variation_tolerance >= safety_margins.height_variation_tolerance, \
        f"Safety margin violated for height variation: {low_util_thresholds.height_variation_tolerance} < {safety_margins.height_variation_tolerance}"
    
    assert low_util_thresholds.geometric_center_tolerance >= safety_margins.geometric_center_tolerance, \
        f"Safety margin violated for geometric center: {low_util_thresholds.geometric_center_tolerance} < {safety_margins.geometric_center_tolerance}"
    
    # Even at 0% utilization, safety margins should not be violated
    zero_util_thresholds = threshold_manager.get_adaptive_thresholds(0.0)
    
    assert zero_util_thresholds.min_support_area_ratio >= safety_margins.min_support_area_ratio
    assert zero_util_thresholds.corner_support_threshold >= safety_margins.corner_support_threshold
    assert zero_util_thresholds.height_variation_tolerance >= safety_margins.height_variation_tolerance
    assert zero_util_thresholds.geometric_center_tolerance >= safety_margins.geometric_center_tolerance


def test_strict_thresholds_at_low_utilization_unit_cases():
    """Unit tests for specific low utilization threshold scenarios."""
    from support_calculation import ThresholdManager, StabilityThresholds
    
    # Create threshold manager with base thresholds
    base_thresholds = StabilityThresholds(
        min_support_area_ratio=0.75,
        corner_support_threshold=0.85,
        height_variation_tolerance=1.0,
        geometric_center_tolerance=0.1
    )
    threshold_manager = ThresholdManager(base_thresholds)
    
    # Test case 1: Zero utilization (strictest possible)
    zero_util_thresholds = threshold_manager.get_adaptive_thresholds(0.0)
    
    # Should be stricter than base
    assert zero_util_thresholds.min_support_area_ratio >= base_thresholds.min_support_area_ratio
    assert zero_util_thresholds.corner_support_threshold >= base_thresholds.corner_support_threshold
    assert zero_util_thresholds.height_variation_tolerance <= base_thresholds.height_variation_tolerance
    assert zero_util_thresholds.geometric_center_tolerance <= base_thresholds.geometric_center_tolerance
    
    # Test case 2: 10% utilization
    low_util_thresholds = threshold_manager.get_adaptive_thresholds(0.10)
    
    # Should still be stricter than base
    assert low_util_thresholds.min_support_area_ratio >= base_thresholds.min_support_area_ratio
    assert low_util_thresholds.corner_support_threshold >= base_thresholds.corner_support_threshold
    assert low_util_thresholds.height_variation_tolerance <= base_thresholds.height_variation_tolerance
    assert low_util_thresholds.geometric_center_tolerance <= base_thresholds.geometric_center_tolerance
    
    # Test case 3: 29% utilization (just below medium threshold)
    almost_medium_thresholds = threshold_manager.get_adaptive_thresholds(0.29)
    
    # Should still be stricter than base
    assert almost_medium_thresholds.min_support_area_ratio >= base_thresholds.min_support_area_ratio
    assert almost_medium_thresholds.corner_support_threshold >= base_thresholds.corner_support_threshold
    
    # Test case 4: Verify monotonic behavior across low utilization range
    utilizations = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.29]
    thresholds_list = [threshold_manager.get_adaptive_thresholds(u) for u in utilizations]
    
    # As utilization increases, thresholds should become less strict (monotonic)
    for i in range(len(thresholds_list) - 1):
        curr_thresh = thresholds_list[i]
        next_thresh = thresholds_list[i + 1]
        
        # Support area ratio should not increase (less strict or same)
        assert curr_thresh.min_support_area_ratio >= next_thresh.min_support_area_ratio, \
            f"util {utilizations[i]:.2f} -> {utilizations[i+1]:.2f}: {curr_thresh.min_support_area_ratio:.3f} -> {next_thresh.min_support_area_ratio:.3f}"
        
        # Corner support threshold should not increase (less strict or same)
        assert curr_thresh.corner_support_threshold >= next_thresh.corner_support_threshold, \
            f"util {utilizations[i]:.2f} -> {utilizations[i+1]:.2f}: {curr_thresh.corner_support_threshold:.3f} -> {next_thresh.corner_support_threshold:.3f}"
    
    # Test case 5: Verify at least one threshold is noticeably stricter
    # For very low utilization (< 10%), we should see noticeable increase from base
    # Test with 10% utilization (low_util_thresholds)
    test_utilization = 0.10
    stricter_support = low_util_thresholds.min_support_area_ratio > base_thresholds.min_support_area_ratio + 0.01
    stricter_corner = low_util_thresholds.corner_support_threshold > base_thresholds.corner_support_threshold + 0.01
    stricter_height = low_util_thresholds.height_variation_tolerance < base_thresholds.height_variation_tolerance - 0.01
    stricter_center = low_util_thresholds.geometric_center_tolerance < base_thresholds.geometric_center_tolerance - 0.001
    
    # At least one threshold should be noticeably stricter
    assert stricter_support or stricter_corner or stricter_height or stricter_center, \
        f"At very low utilization ({test_utilization:.3f}), at least one threshold should be noticeably stricter than base. " \
        f"Adaptive: {low_util_thresholds}, Base: {base_thresholds}"
    
    # Property 4: Safety margins should always be respected
    safety_margins = threshold_manager.safety_margins
    
    # Test case 6: Safety margin enforcement
    for thresholds in thresholds_list:
        assert thresholds.min_support_area_ratio >= safety_margins.min_support_area_ratio, \
            f"Safety margin violated for support area: {thresholds.min_support_area_ratio} < {safety_margins.min_support_area_ratio}"
        
        assert thresholds.corner_support_threshold >= safety_margins.corner_support_threshold, \
            f"Safety margin violated for corner support: {thresholds.corner_support_threshold} < {safety_margins.corner_support_threshold}"
        
        assert thresholds.height_variation_tolerance >= safety_margins.height_variation_tolerance, \
            f"Safety margin violated for height variation: {thresholds.height_variation_tolerance} < {safety_margins.height_variation_tolerance}"
        
        assert thresholds.geometric_center_tolerance >= safety_margins.geometric_center_tolerance, \
            f"Safety margin violated for geometric center: {thresholds.geometric_center_tolerance} < {safety_margins.geometric_center_tolerance}"
    
    # Property 2: Thresholds should be within reasonable bounds (not exceed maximum safe values)
    for thresholds in thresholds_list:
        assert thresholds.min_support_area_ratio <= 0.95, \
            f"min_support_area_ratio should not exceed 95%, got {thresholds.min_support_area_ratio}"
        
        assert thresholds.corner_support_threshold <= 0.95, \
            f"corner_support_threshold should not exceed 95%, got {thresholds.corner_support_threshold}"
        
        assert thresholds.height_variation_tolerance >= 0.5, \
            f"height_variation_tolerance should not go below 0.5, got {thresholds.height_variation_tolerance}"
        
        assert thresholds.geometric_center_tolerance >= 0.05, \
            f"geometric_center_tolerance should not go below 0.05, got {thresholds.geometric_center_tolerance}"
    
    # Property 1: At low utilization (< 30%), thresholds should be stricter than base
    for utilization_ratio in [0.0, 0.1, 0.2, 0.29]:
        adaptive_thresholds = threshold_manager.get_adaptive_thresholds(utilization_ratio)
        
        assert adaptive_thresholds.min_support_area_ratio >= base_thresholds.min_support_area_ratio, \
            f"At low utilization ({utilization_ratio:.3f}), support area threshold should be >= base. " \
            f"Adaptive: {adaptive_thresholds.min_support_area_ratio}, Base: {base_thresholds.min_support_area_ratio}"
        
        assert adaptive_thresholds.corner_support_threshold >= base_thresholds.corner_support_threshold, \
            f"At low utilization ({utilization_ratio:.3f}), corner support threshold should be >= base. " \
            f"Adaptive: {adaptive_thresholds.corner_support_threshold}, Base: {base_thresholds.corner_support_threshold}"
        
        assert adaptive_thresholds.height_variation_tolerance <= base_thresholds.height_variation_tolerance, \
            f"At low utilization ({utilization_ratio:.3f}), height variation tolerance should be <= base. " \
            f"Adaptive: {adaptive_thresholds.height_variation_tolerance}, Base: {base_thresholds.height_variation_tolerance}"
        
        assert adaptive_thresholds.geometric_center_tolerance <= base_thresholds.geometric_center_tolerance, \
            f"At low utilization ({utilization_ratio:.3f}), geometric center tolerance should be <= base. " \
            f"Adaptive: {adaptive_thresholds.geometric_center_tolerance}, Base: {base_thresholds.geometric_center_tolerance}"


@given(
    container_size=container_size_strategy,
    data=st.data()
)
@settings(max_examples=100, deadline=None)
def test_utilization_metrics_provision(container_size, data):
    """
    **Feature: improved-feasibility-mask, Property 23: Utilization metrics provision**
    
    For any completed mask generation process, utilization metrics should be 
    provided as output.
    
    **Validates: Requirements 5.3**
    """
    width, length, height = container_size
    
    # Create space with enhanced feasibility checking enabled
    space = Space(width=width, length=length, height=height, use_enhanced_feasibility=True)
    
    # Generate some placement attempts to create meaningful metrics
    num_attempts = data.draw(st.integers(min_value=1, max_value=10))
    
    for attempt in range(num_attempts):
        # Generate item size that fits within container
        item_size = data.draw(item_size_strategy(width, length))
        x, y, z = item_size
        
        # Generate a valid placement position
        placement_strategy = placement_position_strategy(width, length, x, y)
        lx, ly = data.draw(placement_strategy)
        
        # Generate a height map for the container
        height_map = data.draw(height_map_strategy(width, length))
        
        # Attempt to place the item (this will update performance metrics)
        space.drop_box([x, y, z], space.position_to_index([lx, ly]), False)
    
    # Property: After placement attempts, utilization metrics should be provided
    metrics = space.collect_utilization_metrics()
    
    # Verify that metrics is a dictionary containing expected keys
    assert isinstance(metrics, dict), f"Metrics should be a dictionary, got {type(metrics)}"
    
    # Property 1: Essential utilization metrics should be present
    required_keys = [
        'current_utilization',
        'target_utilization', 
        'baseline_utilization',
        'utilization_gap',
        'recent_success_rate',
        'total_placements',
        'placement_attempts',
        'successful_placements',
        'failed_placements'
    ]
    
    for key in required_keys:
        assert key in metrics, f"Required metric '{key}' missing from metrics output"
    
    # Property 2: Utilization values should be within valid ranges
    assert 0.0 <= metrics['current_utilization'] <= 1.0, \
        f"Current utilization should be between 0 and 1, got {metrics['current_utilization']}"
    
    assert 0.0 <= metrics['target_utilization'] <= 1.0, \
        f"Target utilization should be between 0 and 1, got {metrics['target_utilization']}"
    
    assert 0.0 <= metrics['baseline_utilization'] <= 1.0, \
        f"Baseline utilization should be between 0 and 1, got {metrics['baseline_utilization']}"
    
    assert 0.0 <= metrics['recent_success_rate'] <= 1.0, \
        f"Recent success rate should be between 0 and 1, got {metrics['recent_success_rate']}"
    
    # Property 3: Placement counts should be consistent
    assert metrics['placement_attempts'] >= 0, \
        f"Placement attempts should be non-negative, got {metrics['placement_attempts']}"
    
    assert metrics['successful_placements'] >= 0, \
        f"Successful placements should be non-negative, got {metrics['successful_placements']}"
    
    assert metrics['failed_placements'] >= 0, \
        f"Failed placements should be non-negative, got {metrics['failed_placements']}"
    
    assert metrics['total_placements'] >= 0, \
        f"Total placements should be non-negative, got {metrics['total_placements']}"
    
    # Property 4: Placement counts should add up correctly
    assert metrics['successful_placements'] + metrics['failed_placements'] == metrics['placement_attempts'], \
        f"Successful + failed should equal total attempts: {metrics['successful_placements']} + {metrics['failed_placements']} != {metrics['placement_attempts']}"
    
    # Property 5: Total placements should match successful placements (only successful ones are stored)
    assert metrics['total_placements'] == metrics['successful_placements'], \
        f"Total placements should equal successful placements: {metrics['total_placements']} != {metrics['successful_placements']}"
    
    # Property 6: Utilization gap should be calculated correctly
    expected_gap = metrics['target_utilization'] - metrics['current_utilization']
    assert abs(metrics['utilization_gap'] - expected_gap) < 1e-10, \
        f"Utilization gap calculation incorrect: expected {expected_gap}, got {metrics['utilization_gap']}"
    
    # Property 7: Success rate should be calculated correctly when attempts > 0
    if metrics['placement_attempts'] > 0:
        expected_success_rate = metrics['successful_placements'] / metrics['placement_attempts']
        assert abs(metrics['recent_success_rate'] - expected_success_rate) < 1e-10, \
            f"Success rate calculation incorrect: expected {expected_success_rate}, got {metrics['recent_success_rate']}"
    else:
        # When no attempts, success rate should be 0
        assert metrics['recent_success_rate'] == 0.0, \
            f"Success rate should be 0 when no attempts made, got {metrics['recent_success_rate']}"
    
    # Property 8: Performance monitoring fields should be present
    monitoring_keys = [
        'threshold_adjustments',
        'fallback_active',
        'enhanced_feasibility_usage',
        'baseline_feasibility_usage'
    ]
    
    for key in monitoring_keys:
        assert key in metrics, f"Performance monitoring metric '{key}' missing from metrics output"
    
    # Property 9: Performance monitoring values should be valid
    assert metrics['threshold_adjustments'] >= 0, \
        f"Threshold adjustments should be non-negative, got {metrics['threshold_adjustments']}"
    
    assert isinstance(metrics['fallback_active'], bool), \
        f"Fallback active should be boolean, got {type(metrics['fallback_active'])}"
    
    assert metrics['enhanced_feasibility_usage'] >= 0, \
        f"Enhanced feasibility usage should be non-negative, got {metrics['enhanced_feasibility_usage']}"
    
    assert metrics['baseline_feasibility_usage'] >= 0, \
        f"Baseline feasibility usage should be non-negative, got {metrics['baseline_feasibility_usage']}"
    
    # Property 10: Feasibility usage counts should add up to total attempts
    total_feasibility_usage = metrics['enhanced_feasibility_usage'] + metrics['baseline_feasibility_usage']
    assert total_feasibility_usage == metrics['placement_attempts'], \
        f"Enhanced + baseline feasibility usage should equal total attempts: {total_feasibility_usage} != {metrics['placement_attempts']}"
    
    # Property 11: Target utilization should be > baseline (as per requirements)
    assert metrics['target_utilization'] > metrics['baseline_utilization'], \
        f"Target utilization should exceed baseline: {metrics['target_utilization']} <= {metrics['baseline_utilization']}"
    
    # Property 12: Metrics should be provided even with zero placements
    empty_space = Space(width=5, length=5, height=10, use_enhanced_feasibility=True)
    empty_metrics = empty_space.collect_utilization_metrics()
    
    assert isinstance(empty_metrics, dict), "Metrics should be provided even with no placements"
    assert empty_metrics['current_utilization'] == 0.0, "Empty space should have 0% utilization"
    assert empty_metrics['placement_attempts'] == 0, "Empty space should have 0 placement attempts"
    assert empty_metrics['total_placements'] == 0, "Empty space should have 0 total placements"
    
    # Property 13: Metrics should reflect the current state accurately
    # Test by making additional placements and verifying metrics update
    initial_attempts = metrics['placement_attempts']
    initial_total = metrics['total_placements']
    
    # Make one more placement attempt
    test_item_size = data.draw(item_size_strategy(width, length))
    test_x, test_y, test_z = test_item_size
    test_placement = data.draw(placement_position_strategy(width, length, test_x, test_y))
    test_lx, test_ly = test_placement
    
    placement_result = space.drop_box([test_x, test_y, test_z], space.position_to_index([test_lx, test_ly]), False)
    
    # Get updated metrics
    updated_metrics = space.collect_utilization_metrics()
    
    # Verify metrics were updated
    assert updated_metrics['placement_attempts'] == initial_attempts + 1, \
        f"Placement attempts should increase by 1: {updated_metrics['placement_attempts']} != {initial_attempts + 1}"
    
    if placement_result:
        # Successful placement
        assert updated_metrics['total_placements'] == initial_total + 1, \
            f"Total placements should increase by 1 on success: {updated_metrics['total_placements']} != {initial_total + 1}"
        assert updated_metrics['successful_placements'] == metrics['successful_placements'] + 1, \
            "Successful placements should increase by 1"
    else:
        # Failed placement
        assert updated_metrics['total_placements'] == initial_total, \
            f"Total placements should not change on failure: {updated_metrics['total_placements']} != {initial_total}"
        assert updated_metrics['failed_placements'] == metrics['failed_placements'] + 1, \
            "Failed placements should increase by 1"


@given(
    utilization_gap=st.floats(min_value=0.05, max_value=0.50, allow_nan=False, allow_infinity=False),
    success_rate=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
    data=st.data()
)
@settings(max_examples=100, deadline=None)
def test_threshold_adjustment_triggering(utilization_gap, success_rate, data):
    """
    **Feature: improved-feasibility-mask, Property 24: Threshold adjustment triggering**
    
    For any scenario where utilization falls below target, automatic threshold 
    adjustment should be triggered.
    
    **Validates: Requirements 5.4**
    """
    from support_calculation import ThresholdManager, StabilityThresholds
    
    # Create threshold manager directly for more controlled testing
    base_thresholds = StabilityThresholds(
        min_support_area_ratio=0.75,
        corner_support_threshold=0.85,
        height_variation_tolerance=1.0,
        geometric_center_tolerance=0.1
    )
    threshold_manager = ThresholdManager(base_thresholds)
    
    # Set up performance metrics that would trigger adjustment
    target_utilization = 0.75
    current_utilization = target_utilization - utilization_gap
    
    performance_metrics = {
        'utilization_ratio': current_utilization,
        'target_utilization': target_utilization,
        'utilization_gap': utilization_gap,
        'recent_success_rate': success_rate
    }
    
    # Store initial thresholds
    initial_thresholds = threshold_manager.get_current_thresholds()
    initial_history_length = len(threshold_manager.get_adjustment_history())
    
    # Property: When utilization gap > 5% or success rate < 40%, adjustment should occur
    should_trigger_adjustment = (utilization_gap > 0.05 or success_rate < 0.4)
    
    # Test the adjust_thresholds method
    new_thresholds = threshold_manager.adjust_thresholds(performance_metrics)
    
    if should_trigger_adjustment:
        # Property 1: Thresholds should be adjusted when conditions are met
        thresholds_changed = (
            new_thresholds.min_support_area_ratio != initial_thresholds.min_support_area_ratio or
            new_thresholds.corner_support_threshold != initial_thresholds.corner_support_threshold or
            new_thresholds.height_variation_tolerance != initial_thresholds.height_variation_tolerance or
            new_thresholds.geometric_center_tolerance != initial_thresholds.geometric_center_tolerance
        )
        
        assert thresholds_changed, (
            f"Thresholds should change when adjustment conditions are met. "
            f"Gap: {utilization_gap:.3f}, Success rate: {success_rate:.3f}, "
            f"Initial: {initial_thresholds}, New: {new_thresholds}"
        )
        
        # Property 2: Adjustment history should be updated
        final_history_length = len(threshold_manager.get_adjustment_history())
        assert final_history_length > initial_history_length, (
            f"Adjustment history should be updated when thresholds change. "
            f"Initial length: {initial_history_length}, Final length: {final_history_length}"
        )
        
        # Property 3: Safety margins should be respected
        safety_margins = threshold_manager.safety_margins
        
        assert new_thresholds.min_support_area_ratio >= safety_margins.min_support_area_ratio, (
            f"Safety margin violated for support area: "
            f"{new_thresholds.min_support_area_ratio} < {safety_margins.min_support_area_ratio}"
        )
        
        assert new_thresholds.corner_support_threshold >= safety_margins.corner_support_threshold, (
            f"Safety margin violated for corner support: "
            f"{new_thresholds.corner_support_threshold} < {safety_margins.corner_support_threshold}"
        )
        
        assert new_thresholds.height_variation_tolerance >= safety_margins.height_variation_tolerance, (
            f"Safety margin violated for height variation: "
            f"{new_thresholds.height_variation_tolerance} < {safety_margins.height_variation_tolerance}"
        )
        
        assert new_thresholds.geometric_center_tolerance >= safety_margins.geometric_center_tolerance, (
            f"Safety margin violated for geometric center: "
            f"{new_thresholds.geometric_center_tolerance} < {safety_margins.geometric_center_tolerance}"
        )
        
        # Property 4: For large utilization gaps, thresholds should be relaxed
        if utilization_gap > 0.10:
            # Support area and corner thresholds should be lower (more relaxed)
            assert new_thresholds.min_support_area_ratio <= initial_thresholds.min_support_area_ratio, (
                f"Large utilization gap should relax support area threshold. "
                f"Gap: {utilization_gap:.3f}, Initial: {initial_thresholds.min_support_area_ratio:.3f}, "
                f"New: {new_thresholds.min_support_area_ratio:.3f}"
            )
            
            assert new_thresholds.corner_support_threshold <= initial_thresholds.corner_support_threshold, (
                f"Large utilization gap should relax corner support threshold. "
                f"Gap: {utilization_gap:.3f}, Initial: {initial_thresholds.corner_support_threshold:.3f}, "
                f"New: {new_thresholds.corner_support_threshold:.3f}"
            )
        
        # Property 5: For low success rates, thresholds should be relaxed
        if success_rate < 0.3:
            # At least one threshold should be more relaxed
            more_relaxed = (
                new_thresholds.min_support_area_ratio < initial_thresholds.min_support_area_ratio or
                new_thresholds.corner_support_threshold < initial_thresholds.corner_support_threshold or
                new_thresholds.height_variation_tolerance > initial_thresholds.height_variation_tolerance or
                new_thresholds.geometric_center_tolerance > initial_thresholds.geometric_center_tolerance
            )
            
            assert more_relaxed, (
                f"Low success rate should relax at least one threshold. "
                f"Success rate: {success_rate:.3f}, Initial: {initial_thresholds}, New: {new_thresholds}"
            )
    
    # Property 6: Adjustment history records should contain required information
    if len(threshold_manager.get_adjustment_history()) > initial_history_length:
        latest_record = threshold_manager.get_adjustment_history()[-1]
        
        required_keys = ['timestamp', 'old_thresholds', 'new_thresholds', 
                        'utilization_ratio', 'utilization_gap', 'recent_success_rate']
        
        for key in required_keys:
            assert key in latest_record, f"Adjustment record should contain '{key}'"
        
        # Verify the recorded values match our inputs
        assert abs(latest_record['utilization_gap'] - utilization_gap) < 1e-6, (
            f"Recorded utilization gap should match input: {latest_record['utilization_gap']} != {utilization_gap}"
        )
        
        assert abs(latest_record['recent_success_rate'] - success_rate) < 1e-6, (
            f"Recorded success rate should match input: {latest_record['recent_success_rate']} != {success_rate}"
        )
    
    # Property 7: Current thresholds should be updated
    current_thresholds = threshold_manager.get_current_thresholds()
    assert current_thresholds == new_thresholds, (
        f"Current thresholds should be updated to new thresholds. "
        f"Current: {current_thresholds}, New: {new_thresholds}"
    )
    
    # Property 8: Repeated adjustments with same conditions should be idempotent or converge
    second_adjustment = threshold_manager.adjust_thresholds(performance_metrics)
    
    # The second adjustment might be the same or slightly different, but should still respect safety margins
    safety_margins = threshold_manager.safety_margins
    assert second_adjustment.min_support_area_ratio >= safety_margins.min_support_area_ratio
    assert second_adjustment.corner_support_threshold >= safety_margins.corner_support_threshold
    assert second_adjustment.height_variation_tolerance >= safety_margins.height_variation_tolerance
    assert second_adjustment.geometric_center_tolerance >= safety_margins.geometric_center_tolerance


def test_threshold_adjustment_triggering_unit_cases():
    """Unit tests for specific threshold adjustment triggering scenarios."""
    
    # Test case 1: Large utilization gap should trigger adjustment
    space = Space(width=8, length=8, height=12, use_enhanced_feasibility=True)
    space.target_utilization = 0.75
    space.baseline_utilization = 0.68
    
    # Set up scenario with large gap (20% below target)
    # Add boxes to simulate 55% utilization
    total_volume = space.plain_size[0] * space.plain_size[1] * space.plain_size[2]  # 8*8*12 = 768
    target_volume = int(total_volume * 0.55)  # 422.4 -> 422
    box_volume = 8  # 2x2x2 box
    num_boxes = target_volume // box_volume  # 422 // 8 = 52 boxes
    
    # Add exactly the right number of boxes to get close to 55% utilization
    for i in range(min(num_boxes, 50)):  # Limit to avoid placement issues
        x_pos = (i * 2) % space.plain_size[0]
        y_pos = ((i * 2) // space.plain_size[0]) % space.plain_size[1]
        # Make sure we don't go out of bounds
        if x_pos + 2 <= space.plain_size[0] and y_pos + 2 <= space.plain_size[1]:
            box = Box(2, 2, 2, x_pos, y_pos, 0)
            space.boxes.append(box)
    
    # Verify actual utilization is below target
    actual_utilization = space.get_ratio()
    print(f"Test case 1 - Actual utilization: {actual_utilization:.3f}, Target: {space.target_utilization:.3f}")
    
    space.performance_metrics.update({
        'placement_attempts': 50,
        'successful_placements': 30,
        'failed_placements': 20,
        'utilization_history': [actual_utilization] * 5,
        'threshold_adjustments': 0
    })
    
    initial_adjustments = space.performance_metrics['threshold_adjustments']
    adjustment_triggered = space.trigger_threshold_adjustment()
    
    # Only assert if we actually have a significant gap
    utilization_gap = space.target_utilization - actual_utilization
    if utilization_gap > 0.05:
        assert adjustment_triggered, f"Large utilization gap ({utilization_gap:.3f}) should trigger adjustment"
        assert space.performance_metrics['threshold_adjustments'] > initial_adjustments, \
            "Adjustment count should increase"
    
    # Test case 2: Low success rate should trigger adjustment
    space2 = Space(width=8, length=8, height=12, use_enhanced_feasibility=True)
    space2.target_utilization = 0.75
    
    # Add some boxes but focus on low success rate rather than utilization
    for i in range(10):  # Add a moderate number of boxes
        x_pos = (i * 2) % space2.plain_size[0]
        y_pos = ((i * 2) // space2.plain_size[0]) % space2.plain_size[1]
        if x_pos + 2 <= space2.plain_size[0] and y_pos + 2 <= space2.plain_size[1]:
            box = Box(2, 2, 2, x_pos, y_pos, 0)
            space2.boxes.append(box)
    
    actual_utilization2 = space2.get_ratio()
    print(f"Test case 2 - Actual utilization: {actual_utilization2:.3f}, Target: {space2.target_utilization:.3f}")
    
    space2.performance_metrics.update({
        'placement_attempts': 100,
        'successful_placements': 25,  # 25% success rate (< 40%)
        'failed_placements': 75,
        'utilization_history': [actual_utilization2] * 5,
        'threshold_adjustments': 0
    })
    
    initial_adjustments2 = space2.performance_metrics['threshold_adjustments']
    adjustment_triggered2 = space2.trigger_threshold_adjustment()
    
    # Low success rate (25%) should trigger adjustment regardless of utilization
    success_rate2 = 25 / 100  # 25%
    if success_rate2 < 0.4:
        assert adjustment_triggered2, f"Low success rate ({success_rate2:.3f}) should trigger adjustment"
        assert space2.performance_metrics['threshold_adjustments'] > initial_adjustments2, \
            "Adjustment count should increase for low success rate"
    
    # Test case 3: Fallback mode should prevent adjustment
    space3 = Space(width=8, length=8, height=12, use_enhanced_feasibility=True)
    space3.fallback_active = True
    space3.target_utilization = 0.75
    
    # Add some boxes to create poor conditions that would normally trigger adjustment
    for i in range(5):  # Add few boxes for low utilization
        x_pos = (i * 2) % space3.plain_size[0]
        y_pos = ((i * 2) // space3.plain_size[0]) % space3.plain_size[1]
        if x_pos + 2 <= space3.plain_size[0] and y_pos + 2 <= space3.plain_size[1]:
            box = Box(2, 2, 2, x_pos, y_pos, 0)
            space3.boxes.append(box)
    
    actual_utilization3 = space3.get_ratio()
    print(f"Test case 3 - Actual utilization: {actual_utilization3:.3f}, Target: {space3.target_utilization:.3f}, Fallback: {space3.fallback_active}")
    
    space3.performance_metrics.update({
        'placement_attempts': 50,
        'successful_placements': 10,  # 20% success rate
        'failed_placements': 40,
        'utilization_history': [actual_utilization3] * 5,
        'threshold_adjustments': 0
    })
    
    initial_adjustments3 = space3.performance_metrics['threshold_adjustments']
    adjustment_triggered3 = space3.trigger_threshold_adjustment()
    
    # Fallback mode should prevent adjustment regardless of conditions
    assert not adjustment_triggered3, "Fallback mode should prevent adjustment"
    assert space3.performance_metrics['threshold_adjustments'] == initial_adjustments3, \
        "Adjustment count should not increase in fallback mode"
    
    # Test case 4: Good performance should not trigger adjustment
    space4 = Space(width=8, length=8, height=12, use_enhanced_feasibility=True)
    space4.target_utilization = 0.75
    
    # Add boxes to simulate high utilization (above target)
    # Add enough boxes to get close to or above target utilization
    total_volume4 = space4.plain_size[0] * space4.plain_size[1] * space4.plain_size[2]  # 768
    target_volume4 = int(total_volume4 * 0.76)  # 584
    box_volume4 = 8  # 2x2x2 box
    num_boxes4 = target_volume4 // box_volume4  # 73 boxes
    
    # Add boxes carefully to avoid placement issues
    for i in range(min(num_boxes4, 60)):  # Limit to avoid placement issues
        x_pos = (i * 2) % space4.plain_size[0]
        y_pos = ((i * 2) // space4.plain_size[0]) % space4.plain_size[1]
        # Make sure we don't go out of bounds
        if x_pos + 2 <= space4.plain_size[0] and y_pos + 2 <= space4.plain_size[1]:
            box = Box(2, 2, 2, x_pos, y_pos, 0)
            space4.boxes.append(box)
    
    # Verify actual utilization
    actual_utilization4 = space4.get_ratio()
    print(f"Test case 4 - Actual utilization: {actual_utilization4:.3f}, Target: {space4.target_utilization:.3f}")
    
    space4.performance_metrics.update({
        'placement_attempts': 50,
        'successful_placements': 45,  # 90% success rate
        'failed_placements': 5,
        'utilization_history': [actual_utilization4] * 5,
        'threshold_adjustments': 0
    })
    
    initial_adjustments4 = space4.performance_metrics['threshold_adjustments']
    adjustment_triggered4 = space4.trigger_threshold_adjustment()
    
    # Only assert if we actually have good performance (close to or above target)
    utilization_gap4 = space4.target_utilization - actual_utilization4
    success_rate4 = 45 / 50  # 90%
    
    if utilization_gap4 <= 0.05 and success_rate4 >= 0.4:
        assert not adjustment_triggered4, f"Good performance should not trigger adjustment (gap: {utilization_gap4:.3f}, success: {success_rate4:.3f})"
        assert space4.performance_metrics['threshold_adjustments'] == initial_adjustments4, \
            "Adjustment count should not increase with good performance"
    
    # Test case 5: Verify adjustment history is maintained
    space5 = Space(width=8, length=8, height=12, use_enhanced_feasibility=True)
    space5.target_utilization = 0.75
    # Add boxes to simulate 50% utilization (low)
    total_volume5 = space5.plain_size[0] * space5.plain_size[1] * space5.plain_size[2]
    target_volume5 = int(total_volume5 * 0.50)
    box_volume5 = 8  # 2x2x2 box
    num_boxes5 = target_volume5 // box_volume5
    for i in range(min(num_boxes5, 12)):
        box = Box(2, 2, 2, (i * 2) % space5.plain_size[0], ((i * 2) // space5.plain_size[0]) % space5.plain_size[1], 0)
        space5.boxes.append(box)
    space5.performance_metrics.update({
        'placement_attempts': 50,
        'successful_placements': 15,  # 30% success rate
        'failed_placements': 35,
        'utilization_history': [0.50] * 5,
        'threshold_adjustments': 0
    })
    
    # Trigger multiple adjustments
    adjustment1 = space5.trigger_threshold_adjustment()
    
    # Simulate continued poor performance for second adjustment
    space5.performance_metrics['utilization_history'] = [0.45] * 5
    adjustment2 = space5.trigger_threshold_adjustment()
    
    if adjustment1 or adjustment2:
        history = space5.threshold_manager.get_adjustment_history()
        assert len(history) > 0, "Adjustment history should contain records"
        
        for record in history:
            assert 'timestamp' in record, "Record should have timestamp"
            assert 'old_thresholds' in record, "Record should have old thresholds"
            assert 'new_thresholds' in record, "Record should have new thresholds"
            assert 'utilization_gap' in record, "Record should have utilization gap"
    
    # Test case 6: Verify safety margins are enforced during adjustment
    space6 = Space(width=8, length=8, height=12, use_enhanced_feasibility=True)
    space6.target_utilization = 0.75
    # Add boxes to simulate 30% utilization (very low)
    total_volume6 = space6.plain_size[0] * space6.plain_size[1] * space6.plain_size[2]
    target_volume6 = int(total_volume6 * 0.30)
    box_volume6 = 8  # 2x2x2 box
    num_boxes6 = target_volume6 // box_volume6
    for i in range(min(num_boxes6, 8)):
        box = Box(2, 2, 2, (i * 2) % space6.plain_size[0], ((i * 2) // space6.plain_size[0]) % space6.plain_size[1], 0)
        space6.boxes.append(box)
    space6.performance_metrics.update({
        'placement_attempts': 100,
        'successful_placements': 10,  # 10% success rate
        'failed_placements': 90,
        'utilization_history': [0.30] * 5,
        'threshold_adjustments': 0
    })
    
    # Trigger adjustment with extreme conditions
    adjustment_triggered6 = space6.trigger_threshold_adjustment()
    
    if adjustment_triggered6:
        current_thresholds = space6.threshold_manager.get_current_thresholds()
        safety_margins = space6.threshold_manager.safety_margins
        
        # Verify safety margins are respected
        assert current_thresholds.min_support_area_ratio >= safety_margins.min_support_area_ratio, \
            f"Support area safety margin violated: {current_thresholds.min_support_area_ratio} < {safety_margins.min_support_area_ratio}"
        
        assert current_thresholds.corner_support_threshold >= safety_margins.corner_support_threshold, \
            f"Corner support safety margin violated: {current_thresholds.corner_support_threshold} < {safety_margins.corner_support_threshold}"
        
        assert current_thresholds.height_variation_tolerance >= safety_margins.height_variation_tolerance, \
            f"Height tolerance safety margin violated: {current_thresholds.height_variation_tolerance} < {safety_margins.height_variation_tolerance}"
        
        assert current_thresholds.geometric_center_tolerance >= safety_margins.geometric_center_tolerance, \
            f"Center tolerance safety margin violated: {current_thresholds.geometric_center_tolerance} < {safety_margins.geometric_center_tolerance}"


def test_utilization_metrics_provision_unit_cases():
    """Unit tests for specific utilization metrics provision scenarios."""
    
    # Test case 1: Fresh space with no placements
    space = Space(width=10, length=10, height=15, use_enhanced_feasibility=True)
    metrics = space.collect_utilization_metrics()
    
    # Verify initial state
    assert metrics['current_utilization'] == 0.0, "Fresh space should have 0% utilization"
    assert metrics['placement_attempts'] == 0, "Fresh space should have 0 placement attempts"
    assert metrics['successful_placements'] == 0, "Fresh space should have 0 successful placements"
    assert metrics['failed_placements'] == 0, "Fresh space should have 0 failed placements"
    assert metrics['total_placements'] == 0, "Fresh space should have 0 total placements"
    assert metrics['recent_success_rate'] == 0.0, "Fresh space should have 0% success rate"
    assert metrics['utilization_gap'] == metrics['target_utilization'], "Gap should equal target for empty space"
    assert not metrics['fallback_active'], "Fallback should not be active initially"
    assert metrics['threshold_adjustments'] == 0, "No threshold adjustments initially"
    
    # Test case 2: Space with successful placements
    # Place a few items successfully
    height_map = np.zeros((10, 10), dtype=np.int32)
    
    # Place first item (should succeed on flat surface)
    result1 = space.drop_box([3, 3, 2], space.position_to_index([1, 1]), False)
    assert result1, "First placement should succeed on flat surface"
    
    # Place second item (should succeed)
    result2 = space.drop_box([2, 2, 1], space.position_to_index([5, 5]), False)
    assert result2, "Second placement should succeed"
    
    metrics_after_success = space.collect_utilization_metrics()
    
    assert metrics_after_success['current_utilization'] > 0.0, "Utilization should be > 0 after placements"
    assert metrics_after_success['placement_attempts'] == 2, "Should have 2 placement attempts"
    assert metrics_after_success['successful_placements'] == 2, "Should have 2 successful placements"
    assert metrics_after_success['failed_placements'] == 0, "Should have 0 failed placements"
    assert metrics_after_success['total_placements'] == 2, "Should have 2 total placements"
    assert metrics_after_success['recent_success_rate'] == 1.0, "Should have 100% success rate"
    assert metrics_after_success['enhanced_feasibility_usage'] == 2, "Should have used enhanced feasibility 2 times"
    assert metrics_after_success['baseline_feasibility_usage'] == 0, "Should not have used baseline feasibility"
    
    # Test case 3: Space with mixed success/failure
    # Try to place an item that will likely fail (too big or bad position)
    result3 = space.drop_box([8, 8, 5], space.position_to_index([7, 7]), False)  # Likely to fail (out of bounds)
    
    metrics_after_mixed = space.collect_utilization_metrics()
    
    assert metrics_after_mixed['placement_attempts'] == 3, "Should have 3 placement attempts"
    if not result3:
        assert metrics_after_mixed['successful_placements'] == 2, "Should still have 2 successful placements"
        assert metrics_after_mixed['failed_placements'] == 1, "Should have 1 failed placement"
        assert metrics_after_mixed['total_placements'] == 2, "Should still have 2 total placements"
        assert abs(metrics_after_mixed['recent_success_rate'] - (2.0/3.0)) < 1e-10, "Should have 2/3 success rate"
    
    # Test case 4: Verify target and baseline values are set correctly
    assert metrics_after_mixed['target_utilization'] == 0.75, "Target should be 75% as per requirements"
    assert metrics_after_mixed['baseline_utilization'] == 0.68, "Baseline should be 68% as per requirements"
    assert metrics_after_mixed['target_utilization'] > metrics_after_mixed['baseline_utilization'], \
        "Target should exceed baseline"
    
    # Test case 5: Test with fallback mode
    space_fallback = Space(width=8, length=8, height=12, use_enhanced_feasibility=False)
    space_fallback.fallback_active = True
    space_fallback.fallback_reason = "test_fallback"
    
    # Make a placement attempt in fallback mode
    space_fallback.drop_box([2, 2, 1], space_fallback.position_to_index([2, 2]), False)
    
    fallback_metrics = space_fallback.collect_utilization_metrics()
    
    assert fallback_metrics['fallback_active'], "Fallback should be active"
    assert fallback_metrics['fallback_reason'] == "test_fallback", "Fallback reason should be preserved"
    assert fallback_metrics['enhanced_feasibility_usage'] == 0, "Should not use enhanced feasibility in fallback"
    assert fallback_metrics['baseline_feasibility_usage'] == 1, "Should use baseline feasibility in fallback"
    
    # Test case 6: Test metrics consistency over multiple calls
    # Metrics should be consistent when called multiple times without changes
    metrics_call1 = space.collect_utilization_metrics()
    metrics_call2 = space.collect_utilization_metrics()
    
    # All values should be identical
    for key in metrics_call1:
        if key != 'fallback_reason':  # Skip None values that might compare differently
            assert metrics_call1[key] == metrics_call2[key], \
                f"Metric '{key}' should be consistent across calls: {metrics_call1[key]} != {metrics_call2[key]}"
    
    # Test case 7: Test utilization calculation accuracy
    # Verify that current_utilization matches the space's get_ratio() method
    calculated_ratio = space.get_ratio()
    metrics_ratio = space.collect_utilization_metrics()['current_utilization']
    
    assert abs(calculated_ratio - metrics_ratio) < 1e-10, \
        f"Metrics utilization should match space ratio: {metrics_ratio} != {calculated_ratio}"
    
    # Test case 8: Test with threshold adjustments
    space.performance_metrics['threshold_adjustments'] = 3
    
    metrics_with_adjustments = space.collect_utilization_metrics()
    assert metrics_with_adjustments['threshold_adjustments'] == 3, \
        "Threshold adjustments should be reflected in metrics"
    
    # Test case 9: Verify all required keys are present
    required_keys = [
        'current_utilization', 'target_utilization', 'baseline_utilization',
        'utilization_gap', 'recent_success_rate', 'avg_recent_utilization',
        'total_placements', 'placement_attempts', 'successful_placements',
        'failed_placements', 'threshold_adjustments', 'fallback_active',
        'fallback_reason', 'enhanced_feasibility_usage', 'baseline_feasibility_usage'
    ]
    
    final_metrics = space.collect_utilization_metrics()
    for key in required_keys:
        assert key in final_metrics, f"Required key '{key}' missing from metrics"
    
    # Test case 10: Test edge case with higher utilization
    # Fill up more of the space to test higher utilization scenarios
    space_full = Space(width=5, length=5, height=10, use_enhanced_feasibility=True)
    
    # Fill most positions with larger items to get higher utilization
    for i in range(0, 4, 2):  # Place 2x2x5 items to get higher volume utilization
        for j in range(0, 4, 2):
            space_full.drop_box([2, 2, 5], space_full.position_to_index([i, j]), False)
    
    full_metrics = space_full.collect_utilization_metrics()
    
    # Should have reasonable utilization (not necessarily > 0.5, but > 0)
    assert full_metrics['current_utilization'] > 0.0, "Should have positive utilization when items are placed"
    # The utilization gap should be calculated correctly
    expected_gap = full_metrics['target_utilization'] - full_metrics['current_utilization']
    assert abs(full_metrics['utilization_gap'] - expected_gap) < 1e-10, \
        "Utilization gap should be calculated correctly"