#!/usr/bin/env python3

import sys
import os
import numpy as np

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs', 'bpp0'))

from space import Space

def debug_support_area_test():
    """Debug the support area threshold enforcement."""
    
    # Test case: Exactly 75% support area
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
    
    print("Height map for placement area:")
    print(height_map[lx:lx+x, ly:ly+y])
    
    # Calculate support area
    support_ratio = space.calculate_weighted_support_area(height_map, x, y, lx, ly)
    print(f"Support area ratio: {support_ratio:.3f}")
    
    # Get corner heights
    rec = height_map[lx:lx+x, ly:ly+y]
    r00 = rec[0,0]
    r10 = rec[x-1,0] if x > 1 else r00
    r01 = rec[0,y-1] if y > 1 else r00
    r11 = rec[x-1,y-1] if x > 1 and y > 1 else r00
    
    print(f"Corner heights: r00={r00}, r10={r10}, r01={r01}, r11={r11}")
    
    rm = max(r00,r10,r01,r11)
    sc = int(r00==rm)+int(r10==rm)+int(r01==rm)+int(r11==rm)
    
    print(f"Max corner height: {rm}, Corners at max height: {sc}")
    print(f"Max height in area: {np.max(rec)}")
    
    # Test geometric center validation
    geometric_center = space.get_geometric_center_projection(x, y, lx, ly)
    support_polygon = space.calculate_support_polygon(height_map, x, y, lx, ly)
    
    print(f"Geometric center: {geometric_center}")
    print(f"Support polygon vertices: {support_polygon.vertices}")
    
    # Test the enhanced feasibility checking
    result = space.check_box_enhanced(height_map, x, y, lx, ly, 3)
    print(f"Enhanced feasibility result: {result}")
    
    # Test original feasibility checking for comparison
    space_original = Space(width=10, length=10, height=15, use_enhanced_feasibility=False)
    result_original = space_original.check_box(height_map, x, y, lx, ly, 3)
    print(f"Original feasibility result: {result_original}")

if __name__ == "__main__":
    debug_support_area_test()