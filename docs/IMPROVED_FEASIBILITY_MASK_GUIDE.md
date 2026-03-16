# Improved Feasibility Mask System - Comprehensive Guide

## Overview

The Improved Feasibility Mask System enhances the 3D bin packing algorithm with sophisticated stability checking based on static stability principles. This system achieves higher space utilization rates (target: >75% from baseline 68%) while maintaining physical feasibility.

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture Overview](#architecture-overview)
3. [Getting Started](#getting-started)
4. [Usage Examples](#usage-examples)
5. [Configuration and Tuning](#configuration-and-tuning)
6. [Performance Optimization](#performance-optimization)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

## Key Features

### Enhanced Stability Checking
- **Geometric Center Validation**: Ensures item geometric center projection lies within support polygon
- **Weighted Support Area Calculation**: Considers both direct surface contact and corner support
- **Adaptive Threshold Management**: Dynamic adjustment based on container utilization
- **Corner Support Detection**: Enhanced corner placement validation

### Performance Improvements
- **Optimized Geometric Calculations**: Cached convex hull computation and vectorized operations
- **Adaptive Algorithms**: Automatic threshold adjustment for optimal performance
- **Fallback Mechanisms**: Graceful degradation when performance drops
- **Comprehensive Monitoring**: Real-time performance metrics and logging

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Space Class                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Enhanced        │  │ Performance     │  │ Threshold    │ │
│  │ Feasibility     │  │ Monitoring      │  │ Management   │ │
│  │ Checking        │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
           │                      │                    │
           ▼                      ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│ SupportCalculator│  │ GeometricUtils  │  │ ThresholdManager │
│                 │  │                 │  │                  │
│ • Support Points│  │ • Convex Hull   │  │ • Adaptive       │
│ • Support Polygon│  │ • Point-in-     │  │   Thresholds     │
│ • Weighted Area │  │   Polygon       │  │ • Safety Margins │
└─────────────────┘  └─────────────────┘  └──────────────────┘
```

## Getting Started

### Basic Usage

```python
import sys
sys.path.append('envs/bpp0')
from space import Space

# Create a space with enhanced feasibility checking
space = Space(width=10, length=10, height=20, use_enhanced_feasibility=True)

# Place items using enhanced stability checking
item_size = (3, 3, 2)  # width, length, height
position_index = 0     # position in the container
rotation_flag = False  # no rotation

success = space.drop_box(item_size, position_index, rotation_flag)
if success:
    print("Item placed successfully!")
    print(f"Current utilization: {space.get_ratio():.3f}")
else:
    print("Item placement failed - insufficient stability")
```

### Advanced Configuration

```python
from space import Space
from support_calculation import StabilityThresholds

# Create custom stability thresholds
custom_thresholds = StabilityThresholds(
    min_support_area_ratio=0.80,      # Require 80% support area
    corner_support_threshold=0.90,     # Require 90% corner support
    height_variation_tolerance=1.5,    # Allow 1.5 unit height variation
    geometric_center_tolerance=0.15    # 15% tolerance for center projection
)

# Create space with custom configuration
space = Space(width=15, length=15, height=25, use_enhanced_feasibility=True)
space.support_calculator.thresholds = custom_thresholds

# Monitor performance
metrics = space.collect_utilization_metrics()
print(f"Current utilization: {metrics['current_utilization']:.3f}")
print(f"Target utilization: {metrics['target_utilization']:.3f}")
print(f"Success rate: {metrics['recent_success_rate']:.3f}")
```

## Usage Examples

### Example 1: Basic Packing Sequence

```python
from space import Space
import numpy as np

def pack_items_with_monitoring():
    """Example of packing items with performance monitoring."""
    
    # Create space
    space = Space(width=12, length=12, height=20, use_enhanced_feasibility=True)
    
    # Define item sequence
    items = [
        (3, 3, 2), (2, 4, 3), (4, 2, 2), (3, 3, 3),
        (2, 2, 4), (5, 2, 2), (2, 3, 3), (4, 4, 2)
    ]
    
    placed_items = 0
    total_attempts = 0
    
    for item_idx, (x, y, z) in enumerate(items):
        print(f"\nPlacing item {item_idx + 1}: {x}x{y}x{z}")
        
        # Try different positions and rotations
        placed = False
        
        for rotation in [False, True]:  # Try both orientations
            if placed:
                break
                
            item_dims = (x, y, z) if not rotation else (y, x, z)
            
            for pos_idx in range(space.get_action_space()):
                total_attempts += 1
                
                if space.drop_box(item_dims, pos_idx, rotation):
                    placed_items += 1
                    placed = True
                    
                    # Get current metrics
                    metrics = space.collect_utilization_metrics()
                    print(f"  ✅ Placed at position {pos_idx} "
                          f"(rotation: {rotation})")
                    print(f"  📊 Utilization: {metrics['current_utilization']:.3f}")
                    
                    # Monitor performance every few items
                    if item_idx % 3 == 0:
                        monitoring_result = space.monitor_and_adjust_performance()
                        if monitoring_result['threshold_adjusted']:
                            print(f"  ⚙️ Thresholds adjusted for optimization")
                        if monitoring_result['fallback_activated']:
                            print(f"  ⚠️ Fallback activated: {monitoring_result['degradation_reason']}")
                    
                    break
        
        if not placed:
            print(f"  ❌ Could not place item {item_idx + 1}")
    
    # Final results
    final_metrics = space.collect_utilization_metrics()
    print(f"\n📈 Final Results:")
    print(f"  Items placed: {placed_items}/{len(items)}")
    print(f"  Success rate: {placed_items/len(items):.1%}")
    print(f"  Final utilization: {final_metrics['current_utilization']:.3f}")
    print(f"  Target achieved: {'✅' if final_metrics['current_utilization'] >= 0.75 else '❌'}")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Threshold adjustments: {final_metrics['threshold_adjustments']}")
    
    return space, final_metrics

# Run the example
space, metrics = pack_items_with_monitoring()
```

### Example 2: Performance Comparison

```python
def compare_enhanced_vs_baseline():
    """Compare enhanced vs baseline feasibility checking."""
    
    # Test items
    test_items = [(2, 3, 2), (3, 2, 3), (4, 4, 2), (2, 2, 4), (3, 3, 3)]
    
    # Test baseline system
    baseline_space = Space(width=10, length=10, height=15, use_enhanced_feasibility=False)
    baseline_placed = 0
    
    for item in test_items:
        for pos_idx in range(baseline_space.get_action_space()):
            if baseline_space.drop_box(item, pos_idx, False):
                baseline_placed += 1
                break
    
    baseline_utilization = baseline_space.get_ratio()
    
    # Test enhanced system
    enhanced_space = Space(width=10, length=10, height=15, use_enhanced_feasibility=True)
    enhanced_placed = 0
    
    for item in test_items:
        for pos_idx in range(enhanced_space.get_action_space()):
            if enhanced_space.drop_box(item, pos_idx, False):
                enhanced_placed += 1
                break
    
    enhanced_utilization = enhanced_space.get_ratio()
    enhanced_metrics = enhanced_space.collect_utilization_metrics()
    
    # Results
    print("🔍 Performance Comparison Results:")
    print(f"  Baseline System:")
    print(f"    Items placed: {baseline_placed}/{len(test_items)}")
    print(f"    Utilization: {baseline_utilization:.3f}")
    
    print(f"  Enhanced System:")
    print(f"    Items placed: {enhanced_placed}/{len(test_items)}")
    print(f"    Utilization: {enhanced_utilization:.3f}")
    print(f"    Threshold adjustments: {enhanced_metrics['threshold_adjustments']}")
    
    improvement = enhanced_utilization - baseline_utilization
    print(f"  📊 Improvement: {improvement:+.3f} ({improvement/baseline_utilization*100:+.1f}%)")
    
    return {
        'baseline': {'placed': baseline_placed, 'utilization': baseline_utilization},
        'enhanced': {'placed': enhanced_placed, 'utilization': enhanced_utilization},
        'improvement': improvement
    }

# Run comparison
results = compare_enhanced_vs_baseline()
```

### Example 3: Custom Threshold Optimization

```python
from support_calculation import StabilityThresholds

def optimize_thresholds_for_scenario():
    """Example of optimizing thresholds for a specific scenario."""
    
    # Define test scenario
    test_items = [(2, 2, 3), (3, 3, 2), (4, 2, 2), (2, 4, 3), (3, 2, 4)]
    
    # Test different threshold configurations
    threshold_configs = [
        StabilityThresholds(0.70, 0.80, 1.0, 0.1),  # Relaxed
        StabilityThresholds(0.75, 0.85, 1.0, 0.1),  # Default
        StabilityThresholds(0.80, 0.90, 1.0, 0.1),  # Strict
        StabilityThresholds(0.75, 0.85, 1.5, 0.15), # Flexible height
    ]
    
    config_names = ["Relaxed", "Default", "Strict", "Flexible"]
    best_config = None
    best_utilization = 0.0
    
    print("🎯 Testing Threshold Configurations:")
    
    for i, (config, name) in enumerate(zip(threshold_configs, config_names)):
        space = Space(width=10, length=10, height=15, use_enhanced_feasibility=True)
        space.support_calculator.thresholds = config
        
        placed = 0
        for item in test_items:
            for pos_idx in range(space.get_action_space()):
                if space.drop_box(item, pos_idx, False):
                    placed += 1
                    break
        
        utilization = space.get_ratio()
        print(f"  {name}: {placed}/{len(test_items)} items, "
              f"utilization: {utilization:.3f}")
        
        if utilization > best_utilization:
            best_utilization = utilization
            best_config = (config, name)
    
    print(f"\n🏆 Best Configuration: {best_config[1]}")
    print(f"  Support area ratio: {best_config[0].min_support_area_ratio}")
    print(f"  Corner support: {best_config[0].corner_support_threshold}")
    print(f"  Height tolerance: {best_config[0].height_variation_tolerance}")
    print(f"  Utilization achieved: {best_utilization:.3f}")
    
    return best_config[0]

# Run optimization
optimal_thresholds = optimize_thresholds_for_scenario()
```

## Configuration and Tuning

### Stability Thresholds

The system uses four main threshold parameters:

1. **min_support_area_ratio** (0.5-0.95)
   - Minimum required support area as fraction of item base
   - Lower values = more permissive placement
   - Higher values = stricter stability requirements

2. **corner_support_threshold** (0.65-0.95)
   - Required corner support strength
   - Affects corner placement validation
   - Balance between stability and utilization

3. **height_variation_tolerance** (0.5-3.0)
   - Allowed height variation in support area
   - Higher values = more flexible height requirements
   - Lower values = stricter level surface requirements

4. **geometric_center_tolerance** (0.05-0.25)
   - Tolerance for geometric center projection
   - Affects support polygon validation
   - Balance between precision and flexibility

### Adaptive Behavior Configuration

```python
# Configure adaptive threshold behavior
space = Space(width=10, length=10, height=20, use_enhanced_feasibility=True)

# Set utilization targets
space.target_utilization = 0.80  # Target 80% utilization
space.baseline_utilization = 0.68  # Baseline for comparison
space.degradation_threshold = 0.05  # 5% degradation triggers fallback

# Configure performance monitoring
space.performance_window_size = 100  # Track last 100 operations

# Enable/disable fallback mechanism
space.fallback_active = False  # Start with enhanced mode
```

### Performance Monitoring Configuration

```python
# Configure detailed performance logging
threshold_manager = space.threshold_manager

# Set logging levels
import logging
logging.basicConfig(level=logging.INFO)

# Configure performance trend tracking
threshold_manager.performance_trends = {
    'utilization_trend': [],
    'success_rate_trend': [],
    'adjustment_frequency': 0
}

# Export performance data
performance_data = threshold_manager.export_performance_logs()
```

## Performance Optimization

### Geometric Calculation Optimizations

The system includes several performance optimizations:

1. **Convex Hull Caching**: Frequently computed hulls are cached
2. **Vectorized Operations**: NumPy operations for support area calculation
3. **Early Exit Conditions**: Skip expensive calculations when possible
4. **Optimized Algorithms**: Improved point-in-polygon and cross-product calculations

### Memory Usage Optimization

```python
# Configure cache sizes for memory management
from support_calculation import GeometricUtils

# Set convex hull cache size
GeometricUtils._cache_max_size = 500  # Reduce for memory-constrained environments

# Configure performance log retention
space.threshold_manager.detailed_logs = space.threshold_manager.detailed_logs[-200:]  # Keep last 200 logs
```

### Performance Profiling

Use the included performance profiler to analyze system performance:

```python
from performance_profiler import run_comprehensive_performance_analysis

# Run complete performance analysis
results = run_comprehensive_performance_analysis()

# Results include:
# - Geometric calculation performance
# - Feasibility checking overhead
# - Optimal threshold parameters
# - Performance recommendations
```

## API Reference

### Space Class

#### Constructor
```python
Space(width=10, length=10, height=10, use_enhanced_feasibility=True)
```

#### Key Methods

**drop_box(box_size, idx, flag)**
- Place an item in the container
- Returns: `bool` - Success/failure of placement

**check_box_enhanced(plain, x, y, lx, ly, z)**
- Enhanced feasibility checking with stability analysis
- Returns: `int` - Height at which item can be placed, or -1 if infeasible

**collect_utilization_metrics()**
- Get comprehensive performance metrics
- Returns: `dict` - Current utilization and performance data

**monitor_and_adjust_performance()**
- Perform comprehensive performance monitoring and adjustment
- Returns: `dict` - Monitoring results and actions taken

### SupportCalculator Class

#### Key Methods

**find_support_points(height_map, x, y, lx, ly)**
- Identify support points for placement area
- Returns: `List[SupportPoint]`

**compute_support_polygon(support_points)**
- Compute support polygon from support points
- Returns: `SupportPolygon`

**calculate_weighted_support_area(height_map, x, y, lx, ly)**
- Calculate weighted support area ratio
- Returns: `float` (0.0 to 1.0)

### ThresholdManager Class

#### Key Methods

**get_adaptive_thresholds(utilization_ratio)**
- Get thresholds adapted to current utilization
- Returns: `StabilityThresholds`

**adjust_thresholds(current_performance)**
- Adjust thresholds based on performance metrics
- Returns: `StabilityThresholds`

**log_performance_metrics(metrics)**
- Log performance data for analysis
- Returns: `None`

## Troubleshooting

### Common Issues

#### Low Utilization Performance

**Problem**: System not achieving target utilization rates

**Solutions**:
1. Check threshold configuration - may be too strict
2. Enable adaptive threshold adjustment
3. Verify item sequence diversity
4. Consider relaxing safety margins

```python
# Diagnostic code
metrics = space.collect_utilization_metrics()
if metrics['current_utilization'] < 0.70:
    print("Low utilization detected")
    print(f"Current thresholds: {space.get_current_thresholds()}")
    
    # Try relaxing thresholds
    space.trigger_threshold_adjustment()
```

#### High Computational Overhead

**Problem**: Enhanced feasibility checking too slow

**Solutions**:
1. Reduce convex hull cache size
2. Enable fallback mechanism for performance
3. Optimize threshold parameters
4. Use baseline checking for time-critical applications

```python
# Performance optimization
if metrics['recent_success_rate'] < 0.3:
    # Activate fallback to baseline checking
    space.activate_fallback_mechanism("performance_optimization")
```

#### Frequent Threshold Adjustments

**Problem**: System constantly adjusting thresholds

**Solutions**:
1. Check for oscillating patterns
2. Increase adjustment thresholds
3. Verify test scenario diversity
4. Consider fixed threshold configuration

```python
# Detect adjustment patterns
patterns = space.threshold_manager.detect_adjustment_patterns()
if patterns['oscillating_thresholds']:
    print("Oscillating thresholds detected - consider fixed configuration")
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed performance logging
space.threshold_manager.log_detailed_performance(metrics, adjustment_made=True)

# Export comprehensive logs
debug_data = space.threshold_manager.export_performance_logs()
```

### Performance Validation

Validate system performance with test scenarios:

```python
def validate_system_performance():
    """Validate that system meets performance requirements."""
    
    space = Space(width=10, length=10, height=20, use_enhanced_feasibility=True)
    
    # Test with standard item sequence
    test_items = [(2, 3, 2), (3, 2, 3), (4, 4, 2)] * 10
    
    placed = 0
    for item in test_items:
        for pos_idx in range(space.get_action_space()):
            if space.drop_box(item, pos_idx, False):
                placed += 1
                break
    
    metrics = space.collect_utilization_metrics()
    
    # Validation checks
    checks = {
        'utilization_target': metrics['current_utilization'] >= 0.75,
        'success_rate': metrics['recent_success_rate'] >= 0.5,
        'no_degradation': not space.fallback_active
    }
    
    print("🔍 System Validation Results:")
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    return all(checks.values())

# Run validation
system_valid = validate_system_performance()
```

## Best Practices

1. **Start with Default Configuration**: Use default thresholds and enable adaptive adjustment
2. **Monitor Performance**: Regularly check utilization metrics and success rates
3. **Test with Diverse Scenarios**: Validate with various item sequences and container sizes
4. **Enable Fallback**: Always configure fallback mechanism for production use
5. **Profile Performance**: Use performance profiler to identify optimization opportunities
6. **Log Adjustments**: Enable threshold adjustment logging for analysis
7. **Validate Results**: Regularly validate that target utilization is being achieved

## Conclusion

The Improved Feasibility Mask System provides a sophisticated approach to 3D bin packing with enhanced stability checking and adaptive performance optimization. By following this guide and using the provided examples, you can achieve higher space utilization while maintaining physical feasibility and system reliability.

For additional support or advanced configuration options, refer to the source code documentation and performance profiling tools included with the system.