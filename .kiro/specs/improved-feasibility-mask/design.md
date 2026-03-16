# Design Document

## Overview

This design document outlines the implementation of improved feasibility mask rules for the 3D bin packing system. The current system achieves 68% space utilization, and our goal is to improve this to over 75% by implementing more sophisticated stability checking based on static stability principles from the research paper.

The key insight from the paper is that "an item can be considered stable if the projection of its geometric center lies inside its support polygon," where the support polygon is the convex hull constructed from the support points of the item.

## Architecture

The improved feasibility mask system will enhance the existing `Space.check_box()` method with:

1. **Enhanced Support Calculation**: Replace simple corner checking with comprehensive support polygon analysis
2. **Adaptive Thresholds**: Dynamic adjustment of stability thresholds based on container utilization
3. **Weighted Support Areas**: Consider both contact area and support point strength
4. **Geometric Center Validation**: Ensure geometric center projection falls within support polygon

## Components and Interfaces

### Enhanced Space Class
- **Method**: `check_box_enhanced(plain, x, y, lx, ly, z)` - Improved feasibility checking
- **Method**: `calculate_support_polygon(plain, x, y, lx, ly)` - Compute support polygon from support points
- **Method**: `get_geometric_center_projection(x, y, lx, ly)` - Calculate center projection
- **Method**: `calculate_weighted_support_area(plain, x, y, lx, ly)` - Compute support area with weights
- **Method**: `get_adaptive_thresholds(utilization_ratio)` - Dynamic threshold adjustment

### Support Polygon Calculator
- **Method**: `find_support_points(height_map, x, y, lx, ly)` - Identify all support points
- **Method**: `compute_convex_hull(support_points)` - Calculate convex hull of support points
- **Method**: `point_in_polygon(point, polygon)` - Check if geometric center is within support polygon

### Threshold Manager
- **Method**: `get_stability_thresholds(utilization_ratio)` - Return appropriate thresholds
- **Method**: `adjust_thresholds(current_performance)` - Adapt thresholds based on performance
- **Property**: `min_support_area_ratio` - Minimum required support area
- **Property**: `corner_support_weight` - Weight for corner support contribution

## Data Models

### SupportPoint
```python
class SupportPoint:
    x: float          # X coordinate
    y: float          # Y coordinate  
    height: float     # Support height
    weight: float     # Support strength weight
```

### SupportPolygon
```python
class SupportPolygon:
    vertices: List[Tuple[float, float]]  # Polygon vertices
    area: float                          # Polygon area
    centroid: Tuple[float, float]       # Polygon centroid
```

### StabilityThresholds
```python
class StabilityThresholds:
    min_support_area_ratio: float       # Minimum support area (0.5-0.95)
    corner_support_threshold: float     # Corner support requirement (0.75-0.95)
    height_variation_tolerance: float   # Allowed height variation
    geometric_center_tolerance: float   # Center projection tolerance
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Property 1: Geometric center validation
*For any* item placement with computed support polygon, the geometric center projection should lie within the support polygon boundaries when placement is approved
**Validates: Requirements 1.1**

Property 2: Support area threshold enforcement
*For any* item placement, when the support area ratio is at least 75%, the placement should be considered feasible
**Validates: Requirements 1.2**

Property 3: Comprehensive support calculation
*For any* support area calculation, both direct surface contact and corner support points should contribute to the total support measurement
**Validates: Requirements 1.3**

Property 4: Weighted support with varying heights
*For any* placement area with multiple support heights, the support calculation should use weighted values based on contact area distribution
**Validates: Requirements 1.4**

Property 5: Support threshold rejection
*For any* placement where support area ratio falls below the minimum threshold, the placement should be rejected as infeasible
**Validates: Requirements 1.5**

Property 6: Corner point identification completeness
*For any* rectangular placement area, all four corner support points should be identified and included in stability calculations
**Validates: Requirements 2.1**

Property 7: Convex hull support polygon calculation
*For any* set of corner points with uneven heights, the support polygon should be computed as the convex hull of valid support points
**Validates: Requirements 2.2**

Property 8: Geometric center approval rule
*For any* placement where the geometric center falls within the computed support polygon, corner placement should be approved
**Validates: Requirements 2.3**

Property 9: Additional area support requirement
*For any* placement with insufficient corner support alone, additional area support validation should be required
**Validates: Requirements 2.4**

Property 10: Corner height weighting
*For any* corner support calculation, corners should be weighted according to their relative heights in the support strength calculation
**Validates: Requirements 2.5**

Property 11: Continuous support area measurement
*For any* support area calculation, the measurement should use continuous area analysis rather than discrete corner-only checking
**Validates: Requirements 3.1**

Property 12: Height-weighted support calculation
*For any* placement area with varying support heights, the support area calculation should apply height-based weighting
**Validates: Requirements 3.2**

Property 13: High support area approval
*For any* placement with maximum support area exceeding 85%, the placement should be approved regardless of corner configuration
**Validates: Requirements 3.3**

Property 14: Medium support area validation
*For any* placement with support area between 50-85%, additional corner support validation should be required
**Validates: Requirements 3.4**

Property 15: Low support area rejection
*For any* placement with support area below 50%, the placement should be rejected as unstable
**Validates: Requirements 3.5**

Property 16: Strict thresholds at low utilization
*For any* container with low utilization rate, stricter stability thresholds should be applied
**Validates: Requirements 4.1**

Property 17: Threshold relaxation at high utilization
*For any* container with utilization exceeding 60%, stability thresholds should be gradually relaxed
**Validates: Requirements 4.2**

Property 18: Minimum safety margin maintenance
*For any* threshold relaxation, minimum safety margins should be maintained regardless of utilization pressure
**Validates: Requirements 4.3**

Property 19: Threshold adjustment logging
*For any* threshold adjustment operation, the adjustment should be logged for performance analysis
**Validates: Requirements 4.4**

Property 20: Automatic threshold adjustment
*For any* scenario where utilization targets are not met, thresholds should be automatically adjusted within safe operational bounds
**Validates: Requirements 4.5**

Property 21: Target utilization achievement
*For any* feasibility mask generation using the improved algorithm, space utilization rates should exceed 75%
**Validates: Requirements 5.1**

Property 22: Performance improvement over baseline
*For any* test run with cut_2 dataset, the improved system should demonstrate higher utilization than the baseline 68%
**Validates: Requirements 5.2**

Property 23: Utilization metrics provision
*For any* completed mask generation process, utilization metrics should be provided as output
**Validates: Requirements 5.3**

Property 24: Threshold adjustment triggering
*For any* scenario where utilization falls below target, automatic threshold adjustment should be triggered
**Validates: Requirements 5.4**

Property 25: Performance degradation fallback
*For any* scenario where performance degrades below acceptable levels, the system should revert to the previous stable configuration
**Validates: Requirements 5.5**

## Error Handling

The improved feasibility mask system will handle the following error conditions:

1. **Invalid Geometry**: When support polygon calculation fails due to degenerate cases
2. **Numerical Instability**: When floating-point calculations produce unreliable results
3. **Threshold Violations**: When adaptive thresholds exceed safe operational bounds
4. **Performance Degradation**: When utilization falls significantly below baseline

## Testing Strategy

### Unit Testing
- Test individual geometric calculations (support polygon, convex hull)
- Test threshold adjustment logic with various utilization scenarios
- Test support area calculations with known configurations
- Test corner detection and weighting algorithms

### Property-Based Testing
The system will use Hypothesis (Python property-based testing library) to verify correctness properties. Each property-based test will run a minimum of 100 iterations to ensure robust validation across diverse input scenarios.

Property-based tests will be tagged with comments explicitly referencing the correctness property in the design document using the format: '**Feature: improved-feasibility-mask, Property {number}: {property_text}**'

### Integration Testing
- Test complete feasibility mask generation with cut_2 dataset
- Test performance against baseline system
- Test adaptive threshold behavior under various packing scenarios
- Test system stability under edge cases and stress conditions

The dual testing approach ensures both specific functionality (unit tests) and general correctness across all valid inputs (property tests), providing comprehensive coverage for the improved feasibility mask system.