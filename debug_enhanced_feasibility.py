#!/usr/bin/env python3
"""
Debug script to understand why enhanced feasibility is not improving performance.
"""

import sys
import os
import numpy as np

# Add paths for imports
sys.path.append('acktr')
sys.path.append('envs/bpp0')

from space import Space, Box

def test_single_placement():
    """Test a single placement to understand the difference between baseline and enhanced."""
    
    # Create two identical spaces
    space_baseline = Space(width=10, length=10, height=10, use_enhanced_feasibility=False)
    space_enhanced = Space(width=10, length=10, height=10, use_enhanced_feasibility=True)
    
    # Create a simple height map with some variation
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
    
    # Test item placement at different positions
    test_item = (3, 3, 2)  # 3x3x2 item
    x, y, z = test_item
    
    print("Testing item placement with baseline vs enhanced feasibility checking")
    print(f"Item size: {x}x{y}x{z}")
    print("Height map:")
    print(height_map)
    print()
    
    # Test various positions
    positions_to_test = [
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        (0, 0), (7, 7), (6, 6)
    ]
    
    for lx, ly in positions_to_test:
        if lx + x <= 10 and ly + y <= 10:
            # Test baseline
            baseline_result = space_baseline.check_box(height_map, x, y, lx, ly, z)
            
            # Test enhanced
            enhanced_result = space_enhanced.check_box_enhanced(height_map, x, y, lx, ly, z)
            
            # Get support area for enhanced
            support_area = space_enhanced.calculate_weighted_support_area(height_map, x, y, lx, ly)
            
            print(f"Position ({lx}, {ly}):")
            print(f"  Baseline result: {baseline_result}")
            print(f"  Enhanced result: {enhanced_result}")
            print(f"  Support area ratio: {support_area:.3f}")
            
            if baseline_result != enhanced_result:
                print(f"  *** DIFFERENCE: Baseline={baseline_result}, Enhanced={enhanced_result}")
            
            print()


def test_threshold_behavior():
    """Test how different thresholds affect placement decisions."""
    
    print("Testing threshold behavior...")
    
    space = Space(width=10, length=10, height=10, use_enhanced_feasibility=True)
    
    # Create a challenging height map
    height_map = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],  # Only corners at height 5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],  # Only corners at height 5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int32)
    
    # Test 6x6 item at position (2, 2) - should have exactly 4 corner supports
    x, y, z = 6, 6, 2
    lx, ly = 2, 2
    
    print(f"Testing {x}x{y}x{z} item at position ({lx}, {ly})")
    print("Height map:")
    print(height_map)
    print()
    
    # Calculate support metrics
    support_area = space.calculate_weighted_support_area(height_map, x, y, lx, ly)
    support_polygon = space.calculate_support_polygon(height_map, x, y, lx, ly)
    geometric_center = space.get_geometric_center_projection(x, y, lx, ly)
    
    print(f"Support area ratio: {support_area:.3f}")
    print(f"Support polygon vertices: {support_polygon.vertices}")
    print(f"Geometric center: {geometric_center}")
    
    # Test enhanced feasibility
    enhanced_result = space.check_box_enhanced(height_map, x, y, lx, ly, z)
    baseline_result = space.check_box(height_map, x, y, lx, ly, z)
    
    print(f"Enhanced result: {enhanced_result}")
    print(f"Baseline result: {baseline_result}")
    
    # Check if geometric center is in polygon
    if len(support_polygon.vertices) >= 3:
        from support_calculation import GeometricUtils
        center_in_polygon = GeometricUtils.point_in_polygon(geometric_center, support_polygon.vertices)
        print(f"Geometric center in polygon: {center_in_polygon}")
    
    print()


def test_support_area_calculation():
    """Test support area calculation with different scenarios."""
    
    print("Testing support area calculation...")
    
    space = Space(width=10, length=10, height=10, use_enhanced_feasibility=True)
    
    # Test scenario 1: Uniform height (should give 100% support)
    uniform_map = np.full((10, 10), 5, dtype=np.int32)
    support_area_uniform = space.calculate_weighted_support_area(uniform_map, 3, 3, 2, 2)
    print(f"Uniform height map - Support area: {support_area_uniform:.3f} (expected: 1.000)")
    
    # Test scenario 2: Mixed heights within tolerance
    mixed_map = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 5, 4, 0, 0, 0, 0, 0],  # Heights 4-5 (within tolerance of 1.0)
        [0, 0, 5, 4, 4, 0, 0, 0, 0, 0],
        [0, 0, 3, 4, 5, 0, 0, 0, 0, 0],  # Height 3 is outside tolerance
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int32)
    
    support_area_mixed = space.calculate_weighted_support_area(mixed_map, 3, 3, 2, 2)
    print(f"Mixed height map - Support area: {support_area_mixed:.3f}")
    
    # Manual calculation: max_height = 5, tolerance = 1.0
    # Cells >= 4 are supported: (2,2)=5, (2,3)=5, (2,4)=4, (3,2)=5, (3,3)=4, (3,4)=4, (4,3)=4, (4,4)=5
    # Cell (4,2)=3 is not supported (< 4)
    # So 8 out of 9 cells are supported = 8/9 ≈ 0.889
    print(f"Expected mixed support area: {8/9:.3f}")
    
    # Test scenario 3: No support (all cells much lower)
    no_support_map = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0, 0, 0, 0, 0, 0],  # Only one cell at max height
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int32)
    
    support_area_none = space.calculate_weighted_support_area(no_support_map, 3, 3, 2, 2)
    print(f"No support map - Support area: {support_area_none:.3f} (expected: {1/9:.3f})")
    
    print()


def main():
    """Run all debug tests."""
    print("🔍 Debugging Enhanced Feasibility Checking")
    print("=" * 50)
    
    test_single_placement()
    test_threshold_behavior()
    test_support_area_calculation()
    
    print("Debug tests completed.")


if __name__ == "__main__":
    main()