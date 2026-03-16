# Improved Feasibility Mask System - Usage Guide

## Overview

The Improved Feasibility Mask System enhances the 3D bin packing algorithm with sophisticated stability checking based on static stability principles. This guide provides comprehensive examples and best practices for using the system effectively.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Advanced Configuration](#advanced-configuration)
4. [Performance Optimization](#performance-optimization)
5. [Integration Examples](#integration-examples)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation and Setup

```python
# Import required modules
from envs.bpp0.space import Space, Box
from envs.bpp0.support_calculation import StabilityThresholds, GeometricUtils

# Create an optimized space for immediate use
space = Space(width=10, length=10, height=20, use_enhanced_feasibility=True)

# Configure optimal performance settings
GeometricUtils.configure_performance(
    enable_caching=True,
    enable_early_exit=True,
    cache_size=500
)

print("✅ Improved Feasibility Mask System ready!")
```

### First Placement Example

```python
# Create a simple 3x3x2 item placement
item_size = (3, 3, 2)  # (width, length, height)
position = (0, 0)      # (x, y) position

# Try to place the item
success = space.drop_box(item_size, space.position_to_index(position), False)

if success:
    print(f"✅ Item placed successfully!")
    print(f"Current utilization: {space.get_ratio():.3f}")
else:
    print("❌ Placement failed - position not feasible")
```

## Basic Usage

### Creating and Configuring Spaces

#### Standard Configuration

```python
# Create a standard space with enhanced feasibility
space = Space(
    width=10,           # Container width
    length=10,          # Container length  
    height=20,          # Container height
    use_enhanced_feasibility=True  # Enable improved checking
)

# The system uses optimized default thresholds
current_thresholds = space.get_current_thresholds()
print(f"Support area ratio: {current_thresholds.min_support_area_ratio}")
print(f"Corner support: {current_thresholds.corner_support_threshold}")
```

#### Custom Threshold Configuration

```python
from envs.bpp0.support_calculation import StabilityThresholds

# Create custom thresholds for specific requirements
custom_thresholds = StabilityThresholds(
    min_support_area_ratio=0.75,      # Require 75% support area
    corner_support_threshold=0.85,     # Require 85% corner support
    height_variation_tolerance=1.0,    # Allow 1 unit height variation
    geometric_center_tolerance=0.1     # 0.1 unit center tolerance
)

# Apply custom thresholds
space.support_calculator.thresholds = custom_thresholds
print("Custom thresholds applied")
```

### Item Placement Workflow

#### Single Item Placement

```python
def place_single_item(space, item_size, preferred_position=None):
    """
    Place a single item with enhanced feasibility checking.
    
    Args:
        space: Space object with enhanced feasibility enabled
        item_size: Tuple of (width, length, height)
        preferred_position: Optional (x, y) position preference
    
    Returns:
        Tuple of (success: bool, final_position: tuple, utilization: float)
    """
    width, length, height = item_size
    
    # Try preferred position first
    if preferred_position:
        x, y = preferred_position
        if space.drop_box(item_size, space.position_to_index((x, y)), False):
            return True, (x, y), space.get_ratio()
    
    # Search for feasible position
    for x in range(space.plain_size[0] - width + 1):
        for y in range(space.plain_size[1] - length + 1):
            if space.drop_box(item_size, space.position_to_index((x, y)), False):
                return True, (x, y), space.get_ratio()
    
    return False, None, space.get_ratio()

# Example usage
item = (2, 3, 1)
success, position, utilization = place_single_item(space, item)

if success:
    print(f"✅ Item {item} placed at {position}")
    print(f"Container utilization: {utilization:.3f}")
else:
    print(f"❌ Could not place item {item}")
```

#### Batch Item Placement

```python
def place_item_sequence(space, item_list, rotation_allowed=True):
    """
    Place a sequence of items with performance monitoring.
    
    Args:
        space: Space object
        item_list: List of (width, length, height) tuples
        rotation_allowed: Whether to try 90-degree rotation
    
    Returns:
        Dictionary with placement results and performance metrics
    """
    results = {
        'placed_items': [],
        'failed_items': [],
        'final_utilization': 0.0,
        'placement_attempts': 0,
        'performance_metrics': {}
    }
    
    for i, item_size in enumerate(item_list):
        width, length, height = item_size
        placed = False
        
        # Try original orientation
        for x in range(space.plain_size[0] - width + 1):
            for y in range(space.plain_size[1] - length + 1):
                results['placement_attempts'] += 1
                if space.drop_box(item_size, space.position_to_index((x, y)), False):
                    results['placed_items'].append({
                        'item': item_size,
                        'position': (x, y),
                        'rotation': False,
                        'utilization_after': space.get_ratio()
                    })
                    placed = True
                    break
            if placed:
                break
        
        # Try rotated orientation if allowed and not yet placed
        if not placed and rotation_allowed and width != length:
            rotated_size = (length, width, height)
            for x in range(space.plain_size[0] - length + 1):
                for y in range(space.plain_size[1] - width + 1):
                    results['placement_attempts'] += 1
                    if space.drop_box(rotated_size, space.position_to_index((x, y)), True):
                        results['placed_items'].append({
                            'item': item_size,
                            'position': (x, y),
                            'rotation': True,
                            'utilization_after': space.get_ratio()
                        })
                        placed = True
                        break
                if placed:
                    break
        
        if not placed:
            results['failed_items'].append(item_size)
        
        # Monitor performance every 10 items
        if (i + 1) % 10 == 0:
            monitoring_result = space.monitor_and_adjust_performance()
            if monitoring_result['threshold_adjusted']:
                print(f"Thresholds adjusted after item {i + 1}")
    
    results['final_utilization'] = space.get_ratio()
    results['performance_metrics'] = space.collect_utilization_metrics()
    
    return results

# Example usage
items = [(2, 2, 1), (3, 2, 2), (1, 4, 1), (2, 3, 3)]
results = place_item_sequence(space, items)

print(f"Placed: {len(results['placed_items'])}/{len(items)} items")
print(f"Final utilization: {results['final_utilization']:.3f}")
print(f"Total attempts: {results['placement_attempts']}")
```

## Advanced Configuration

### Adaptive Threshold Management

#### Utilization-Based Adjustment

```python
def configure_adaptive_thresholds(space, target_utilization=0.75):
    """
    Configure adaptive threshold management for optimal performance.
    
    Args:
        space: Space object
        target_utilization: Target space utilization (0.0 to 1.0)
    """
    # Set target utilization
    space.target_utilization = target_utilization
    
    # Configure performance monitoring
    space.degradation_threshold = 0.05  # 5% degradation triggers adjustment
    space.performance_window_size = 50  # Monitor last 50 operations
    
    # Enable automatic threshold adjustment
    space.threshold_manager.adjustment_history = []
    
    print(f"Adaptive thresholds configured for {target_utilization:.1%} utilization")
    
    return space

# Apply adaptive configuration
space = configure_adaptive_thresholds(space, target_utilization=0.78)
```

#### Performance-Based Fallback

```python
def setup_performance_fallback(space, fallback_threshold=0.03):
    """
    Setup automatic fallback to baseline checking when performance degrades.
    
    Args:
        space: Space object
        fallback_threshold: Performance degradation threshold
    """
    # Configure fallback sensitivity
    space.degradation_threshold = fallback_threshold
    
    # Store stable configuration
    space.stable_configuration = {
        'thresholds': space.threshold_manager.get_current_thresholds(),
        'utilization': space.get_ratio(),
        'success_rate': 1.0
    }
    
    print(f"Performance fallback configured (threshold: {fallback_threshold:.1%})")

# Setup fallback mechanism
setup_performance_fallback(space, fallback_threshold=0.03)
```

### Performance Optimization Configurations

#### High-Performance Setup

```python
def create_high_performance_space(width=10, length=10, height=20):
    """
    Create a space optimized for maximum computational performance.
    
    Returns:
        Optimized Space object
    """
    # Create space with enhanced feasibility
    space = Space(width=width, length=length, height=height, use_enhanced_feasibility=True)
    
    # Apply performance-optimized thresholds
    performance_thresholds = StabilityThresholds(
        min_support_area_ratio=0.70,      # Optimized based on profiling
        corner_support_threshold=0.80,     # 121.3% overhead acceptable
        height_variation_tolerance=0.5,    # Tightened for performance
        geometric_center_tolerance=0.1     # Balanced for 1.3M ops/sec
    )
    space.support_calculator.thresholds = performance_thresholds
    
    # Configure geometric calculations for performance
    GeometricUtils.configure_performance(
        enable_caching=True,
        enable_early_exit=True,
        cache_size=500  # Optimized cache size
    )
    
    # Performance monitoring settings
    space.target_utilization = 0.75
    space.degradation_threshold = 0.03  # Sensitive to degradation
    space.performance_window_size = 30  # Fast response
    
    return space

# Create high-performance space
hp_space = create_high_performance_space()
print("High-performance space created")
```

#### Memory-Efficient Setup

```python
def create_memory_efficient_space(width=10, length=10, height=20):
    """
    Create a space optimized for minimal memory usage.
    
    Returns:
        Memory-optimized Space object
    """
    space = Space(width=width, length=length, height=height, use_enhanced_feasibility=True)
    
    # Conservative thresholds to reduce computation
    memory_thresholds = StabilityThresholds(
        min_support_area_ratio=0.75,
        corner_support_threshold=0.85,
        height_variation_tolerance=0.5,
        geometric_center_tolerance=0.1
    )
    space.support_calculator.thresholds = memory_thresholds
    
    # Minimal cache configuration
    GeometricUtils.configure_performance(
        enable_caching=True,
        enable_early_exit=True,
        cache_size=100  # Small cache
    )
    
    # Larger performance window for stability
    space.performance_window_size = 20
    space.degradation_threshold = 0.10  # Less sensitive
    
    return space

# Create memory-efficient space
me_space = create_memory_efficient_space()
print("Memory-efficient space created")
```

## Performance Optimization

### Real-Time Performance Monitoring

#### Performance Dashboard

```python
def display_performance_dashboard(space):
    """
    Display a comprehensive performance dashboard.
    
    Args:
        space: Space object to monitor
    """
    metrics = space.collect_utilization_metrics()
    cache_stats = GeometricUtils.get_cache_stats()
    
    print("=" * 60)
    print("🚀 IMPROVED FEASIBILITY MASK - PERFORMANCE DASHBOARD")
    print("=" * 60)
    
    # Utilization Status
    util_status = "✅" if metrics['current_utilization'] >= metrics['target_utilization'] else "⚠️"
    print(f"📊 UTILIZATION {util_status}")
    print(f"   Current: {metrics['current_utilization']:.3f} ({metrics['current_utilization']:.1%})")
    print(f"   Target:  {metrics['target_utilization']:.3f} ({metrics['target_utilization']:.1%})")
    print(f"   Gap:     {metrics['utilization_gap']:+.3f}")
    
    # Performance Status
    perf_status = "✅" if metrics['recent_success_rate'] > 0.7 else "⚠️" if metrics['recent_success_rate'] > 0.4 else "🔴"
    print(f"\n⚡ PERFORMANCE {perf_status}")
    print(f"   Success Rate: {metrics['recent_success_rate']:.3f} ({metrics['recent_success_rate']:.1%})")
    print(f"   Total Placements: {metrics['total_placements']}")
    print(f"   Attempts: {metrics['placement_attempts']}")
    
    # System Status
    sys_status = "🔴" if metrics['fallback_active'] else "✅"
    print(f"\n🔧 SYSTEM {sys_status}")
    print(f"   Enhanced Mode: {'❌ (Fallback Active)' if metrics['fallback_active'] else '✅'}")
    print(f"   Threshold Adjustments: {metrics['threshold_adjustments']}")
    if metrics['fallback_active']:
        print(f"   Fallback Reason: {metrics['fallback_reason']}")
    
    # Cache Status
    cache_status = "⚠️" if cache_stats['cache_usage_ratio'] > 0.8 else "✅"
    print(f"\n💾 CACHE {cache_status}")
    print(f"   Usage: {cache_stats['cache_usage_ratio']:.1%} ({cache_stats['cache_size']}/{cache_stats['cache_max_size']})")
    print(f"   Hit Rate: {cache_stats['cache_hit_rate']:.1%}")
    print(f"   Enabled: {'✅' if cache_stats['caching_enabled'] else '❌'}")
    
    print("=" * 60)

# Example usage
display_performance_dashboard(space)
```

#### Automated Performance Alerts

```python
def setup_performance_alerts(space, alert_callback=None):
    """
    Setup automated performance monitoring with alerts.
    
    Args:
        space: Space object to monitor
        alert_callback: Optional function to call when alerts are triggered
    """
    def default_alert_callback(alert_type, message, metrics):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {alert_type}: {message}")
    
    callback = alert_callback or default_alert_callback
    
    def check_alerts():
        metrics = space.collect_utilization_metrics()
        cache_stats = GeometricUtils.get_cache_stats()
        
        # Utilization alerts
        if metrics['current_utilization'] < 0.60:
            callback("🔴 CRITICAL", f"Very low utilization: {metrics['current_utilization']:.1%}", metrics)
        elif metrics['current_utilization'] < 0.70:
            callback("🟡 WARNING", f"Low utilization: {metrics['current_utilization']:.1%}", metrics)
        
        # Performance alerts
        if metrics['recent_success_rate'] < 0.20:
            callback("🔴 CRITICAL", f"Very low success rate: {metrics['recent_success_rate']:.1%}", metrics)
        elif metrics['recent_success_rate'] < 0.40:
            callback("🟡 WARNING", f"Low success rate: {metrics['recent_success_rate']:.1%}", metrics)
        
        # System alerts
        if metrics['fallback_active']:
            callback("⚠️ SYSTEM", f"Fallback active: {metrics['fallback_reason']}", metrics)
        
        # Memory alerts
        if cache_stats['cache_usage_ratio'] > 0.90:
            callback("🟡 MEMORY", f"High cache usage: {cache_stats['cache_usage_ratio']:.1%}", metrics)
    
    return check_alerts

# Setup alerts
alert_checker = setup_performance_alerts(space)

# Use during operation (call periodically)
alert_checker()
```

### Cache Management and Optimization

#### Intelligent Cache Management

```python
def manage_cache_intelligently(operation_count):
    """
    Intelligent cache management based on operation patterns.
    
    Args:
        operation_count: Current operation number
    """
    cache_stats = GeometricUtils.get_cache_stats()
    
    # Clear cache periodically or when usage is high
    if (operation_count % 1000 == 0 or 
        cache_stats['cache_usage_ratio'] > 0.85):
        
        print(f"Cache maintenance at operation {operation_count}")
        print(f"  Before: {cache_stats['cache_size']} entries ({cache_stats['cache_usage_ratio']:.1%})")
        
        GeometricUtils.clear_cache()
        
        new_stats = GeometricUtils.get_cache_stats()
        print(f"  After: {new_stats['cache_size']} entries ({new_stats['cache_usage_ratio']:.1%})")
    
    # Adjust cache size based on performance
    if cache_stats['cache_hit_rate'] < 0.3 and cache_stats['cache_size'] > 100:
        # Low hit rate, reduce cache size
        GeometricUtils.configure_performance(cache_size=max(100, cache_stats['cache_max_size'] // 2))
        print(f"Reduced cache size to {GeometricUtils._cache_max_size} due to low hit rate")
    elif cache_stats['cache_hit_rate'] > 0.8 and cache_stats['cache_usage_ratio'] > 0.9:
        # High hit rate and usage, increase cache size
        new_size = min(1000, cache_stats['cache_max_size'] * 2)
        GeometricUtils.configure_performance(cache_size=new_size)
        print(f"Increased cache size to {new_size} due to high hit rate")

# Example usage in training loop
for operation in range(5000):
    # Your placement operations here
    # ...
    
    # Intelligent cache management
    if operation % 50 == 0:  # Check every 50 operations
        manage_cache_intelligently(operation)
```

## Integration Examples

### Training Loop Integration

#### Basic Training Integration

```python
def enhanced_training_loop(space, item_generator, max_episodes=1000):
    """
    Training loop with enhanced feasibility mask integration.
    
    Args:
        space: Space object with enhanced feasibility
        item_generator: Function that generates item sequences
        max_episodes: Maximum number of training episodes
    
    Returns:
        Training results and performance metrics
    """
    training_results = {
        'episodes': [],
        'utilization_history': [],
        'performance_metrics': [],
        'threshold_adjustments': 0
    }
    
    for episode in range(max_episodes):
        # Reset space for new episode
        episode_space = Space(
            width=space.plain_size[0], 
            length=space.plain_size[1], 
            height=space.plain_size[2],
            use_enhanced_feasibility=True
        )
        episode_space.support_calculator.thresholds = space.support_calculator.thresholds
        
        # Generate item sequence for this episode
        items = item_generator()
        
        episode_data = {
            'episode': episode,
            'items_attempted': len(items),
            'items_placed': 0,
            'final_utilization': 0.0,
            'performance_events': []
        }
        
        # Place items in sequence
        for i, item in enumerate(items):
            # Try to place item
            placed = False
            for x in range(episode_space.plain_size[0] - item[0] + 1):
                for y in range(episode_space.plain_size[1] - item[1] + 1):
                    if episode_space.drop_box(item, episode_space.position_to_index((x, y)), False):
                        episode_data['items_placed'] += 1
                        placed = True
                        break
                if placed:
                    break
            
            # Monitor performance every 10 items
            if (i + 1) % 10 == 0:
                monitoring_result = episode_space.monitor_and_adjust_performance()
                if monitoring_result['threshold_adjusted']:
                    training_results['threshold_adjustments'] += 1
                    episode_data['performance_events'].append({
                        'item_index': i,
                        'event': 'threshold_adjustment',
                        'utilization': episode_space.get_ratio()
                    })
        
        # Record episode results
        episode_data['final_utilization'] = episode_space.get_ratio()
        training_results['episodes'].append(episode_data)
        training_results['utilization_history'].append(episode_data['final_utilization'])
        
        # Collect performance metrics every 100 episodes
        if (episode + 1) % 100 == 0:
            metrics = episode_space.collect_utilization_metrics()
            training_results['performance_metrics'].append({
                'episode': episode,
                'metrics': metrics
            })
            
            print(f"Episode {episode + 1}: Utilization {episode_data['final_utilization']:.3f}, "
                  f"Placed {episode_data['items_placed']}/{episode_data['items_attempted']} items")
    
    return training_results

# Example item generator
def random_item_generator(num_items=20):
    """Generate random item sequence for training."""
    import random
    items = []
    for _ in range(num_items):
        width = random.randint(1, 3)
        length = random.randint(1, 3)
        height = random.randint(1, 2)
        items.append((width, length, height))
    return items

# Run training with enhanced feasibility
results = enhanced_training_loop(space, random_item_generator, max_episodes=100)
print(f"Training completed: {len(results['episodes'])} episodes")
print(f"Average utilization: {sum(results['utilization_history']) / len(results['utilization_history']):.3f}")
print(f"Threshold adjustments: {results['threshold_adjustments']}")
```

### Evaluation and Benchmarking

#### Performance Comparison

```python
def compare_feasibility_methods(item_sequences, container_size=(10, 10, 20)):
    """
    Compare enhanced vs baseline feasibility checking performance.
    
    Args:
        item_sequences: List of item sequences to test
        container_size: Container dimensions (width, length, height)
    
    Returns:
        Comparison results
    """
    width, length, height = container_size
    
    results = {
        'enhanced': {'utilizations': [], 'times': [], 'placements': []},
        'baseline': {'utilizations': [], 'times': [], 'placements': []}
    }
    
    for sequence_idx, items in enumerate(item_sequences):
        print(f"Testing sequence {sequence_idx + 1}/{len(item_sequences)}")
        
        # Test enhanced method
        enhanced_space = Space(width, length, height, use_enhanced_feasibility=True)
        start_time = time.time()
        
        enhanced_placed = 0
        for item in items:
            placed = False
            for x in range(width - item[0] + 1):
                for y in range(length - item[1] + 1):
                    if enhanced_space.drop_box(item, enhanced_space.position_to_index((x, y)), False):
                        enhanced_placed += 1
                        placed = True
                        break
                if placed:
                    break
        
        enhanced_time = time.time() - start_time
        enhanced_util = enhanced_space.get_ratio()
        
        results['enhanced']['utilizations'].append(enhanced_util)
        results['enhanced']['times'].append(enhanced_time)
        results['enhanced']['placements'].append(enhanced_placed)
        
        # Test baseline method
        baseline_space = Space(width, length, height, use_enhanced_feasibility=False)
        start_time = time.time()
        
        baseline_placed = 0
        for item in items:
            placed = False
            for x in range(width - item[0] + 1):
                for y in range(length - item[1] + 1):
                    if baseline_space.drop_box(item, baseline_space.position_to_index((x, y)), False):
                        baseline_placed += 1
                        placed = True
                        break
                if placed:
                    break
        
        baseline_time = time.time() - start_time
        baseline_util = baseline_space.get_ratio()
        
        results['baseline']['utilizations'].append(baseline_util)
        results['baseline']['times'].append(baseline_time)
        results['baseline']['placements'].append(baseline_placed)
    
    # Calculate summary statistics
    summary = {
        'enhanced': {
            'avg_utilization': sum(results['enhanced']['utilizations']) / len(results['enhanced']['utilizations']),
            'avg_time': sum(results['enhanced']['times']) / len(results['enhanced']['times']),
            'avg_placements': sum(results['enhanced']['placements']) / len(results['enhanced']['placements'])
        },
        'baseline': {
            'avg_utilization': sum(results['baseline']['utilizations']) / len(results['baseline']['utilizations']),
            'avg_time': sum(results['baseline']['times']) / len(results['baseline']['times']),
            'avg_placements': sum(results['baseline']['placements']) / len(results['baseline']['placements'])
        }
    }
    
    # Calculate improvements
    util_improvement = ((summary['enhanced']['avg_utilization'] - summary['baseline']['avg_utilization']) 
                       / summary['baseline']['avg_utilization'] * 100)
    time_overhead = ((summary['enhanced']['avg_time'] - summary['baseline']['avg_time']) 
                    / summary['baseline']['avg_time'] * 100)
    
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)
    print(f"Enhanced Utilization: {summary['enhanced']['avg_utilization']:.3f} ({summary['enhanced']['avg_utilization']:.1%})")
    print(f"Baseline Utilization: {summary['baseline']['avg_utilization']:.3f} ({summary['baseline']['avg_utilization']:.1%})")
    print(f"Utilization Improvement: {util_improvement:+.1f}%")
    print(f"")
    print(f"Enhanced Time: {summary['enhanced']['avg_time']:.4f}s")
    print(f"Baseline Time: {summary['baseline']['avg_time']:.4f}s")
    print(f"Time Overhead: {time_overhead:+.1f}%")
    print(f"")
    print(f"Enhanced Placements: {summary['enhanced']['avg_placements']:.1f}")
    print(f"Baseline Placements: {summary['baseline']['avg_placements']:.1f}")
    
    return results, summary

# Generate test sequences
test_sequences = []
for _ in range(10):
    sequence = random_item_generator(15)
    test_sequences.append(sequence)

# Run comparison
comparison_results, summary = compare_feasibility_methods(test_sequences)
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Low Utilization Performance

**Problem**: System achieves lower utilization than expected (<70%)

**Diagnosis**:
```python
# Check current thresholds
thresholds = space.get_current_thresholds()
print(f"Support area ratio: {thresholds.min_support_area_ratio}")
print(f"Corner support: {thresholds.corner_support_threshold}")

# Check success rate
metrics = space.collect_utilization_metrics()
print(f"Success rate: {metrics['recent_success_rate']:.3f}")
print(f"Current utilization: {metrics['current_utilization']:.3f}")
```

**Solutions**:
```python
# Solution 1: Relax thresholds
relaxed_thresholds = StabilityThresholds(
    min_support_area_ratio=0.65,
    corner_support_threshold=0.75,
    height_variation_tolerance=1.0,
    geometric_center_tolerance=0.15
)
space.support_calculator.thresholds = relaxed_thresholds

# Solution 2: Enable adaptive adjustment
space.trigger_threshold_adjustment()

# Solution 3: Check for fallback activation
if space.fallback_active:
    print(f"Fallback active: {space.fallback_reason}")
    space.deactivate_fallback_mechanism()
```

#### Issue 2: High Memory Usage

**Problem**: Memory consumption increases over time

**Diagnosis**:
```python
# Check cache usage
cache_stats = GeometricUtils.get_cache_stats()
print(f"Cache usage: {cache_stats['cache_usage_ratio']:.2%}")
print(f"Cache size: {cache_stats['cache_size']}")

# Check log sizes
log_count = len(space.threshold_manager.detailed_logs)
print(f"Performance logs: {log_count}")
```

**Solutions**:
```python
# Solution 1: Clear cache
GeometricUtils.clear_cache()

# Solution 2: Reduce cache size
GeometricUtils.configure_performance(cache_size=100)

# Solution 3: Rotate logs
space.threshold_manager.detailed_logs = space.threshold_manager.detailed_logs[-100:]

# Solution 4: Disable caching for memory-critical applications
GeometricUtils.configure_performance(enable_caching=False)
```

#### Issue 3: Poor Performance (High Overhead)

**Problem**: Enhanced feasibility checking is too slow

**Diagnosis**:
```python
from acktr.performance_profiler import FeasibilityCheckProfiler

profiler = FeasibilityCheckProfiler()
results = profiler.profile_enhanced_vs_baseline(100)

overhead = ((results['enhanced'].avg_time_per_operation - 
            results['baseline'].avg_time_per_operation) / 
           results['baseline'].avg_time_per_operation * 100)
print(f"Current overhead: {overhead:.1f}%")
```

**Solutions**:
```python
# Solution 1: Activate fallback for performance-critical sections
space.activate_fallback_mechanism("performance_optimization")

# Solution 2: Use optimized thresholds
optimized_thresholds = StabilityThresholds(0.70, 0.80, 0.5, 0.1)
space.support_calculator.thresholds = optimized_thresholds

# Solution 3: Optimize cache settings
GeometricUtils.configure_performance(
    enable_caching=True,
    enable_early_exit=True,
    cache_size=500
)

# Solution 4: Reduce geometric calculations
space.support_calculator.thresholds.geometric_center_tolerance = 0.2  # Less strict
```

### Debug Tools and Utilities

#### Performance Debug Tool

```python
def debug_performance_comprehensive(space):
    """
    Comprehensive performance debugging tool.
    
    Args:
        space: Space object to debug
    """
    print("🔍 COMPREHENSIVE PERFORMANCE DEBUG")
    print("=" * 50)
    
    # System Information
    print("📋 SYSTEM INFORMATION")
    print(f"   Container Size: {space.plain_size}")
    print(f"   Enhanced Mode: {space.use_enhanced_feasibility}")
    print(f"   Target Utilization: {space.target_utilization:.1%}")
    
    # Current Metrics
    metrics = space.collect_utilization_metrics()
    print(f"\n📊 CURRENT METRICS")
    print(f"   Utilization: {metrics['current_utilization']:.3f} ({metrics['current_utilization']:.1%})")
    print(f"   Success Rate: {metrics['recent_success_rate']:.3f}")
    print(f"   Placements: {metrics['total_placements']}")
    print(f"   Attempts: {metrics['placement_attempts']}")
    
    # Threshold Information
    thresholds = space.get_current_thresholds()
    print(f"\n⚙️ CURRENT THRESHOLDS")
    print(f"   Support Area: {thresholds.min_support_area_ratio:.3f}")
    print(f"   Corner Support: {thresholds.corner_support_threshold:.3f}")
    print(f"   Height Tolerance: {thresholds.height_variation_tolerance:.3f}")
    print(f"   Center Tolerance: {thresholds.geometric_center_tolerance:.3f}")
    
    # Cache Statistics
    cache_stats = GeometricUtils.get_cache_stats()
    print(f"\n💾 CACHE STATISTICS")
    print(f"   Usage: {cache_stats['cache_usage_ratio']:.1%} ({cache_stats['cache_size']}/{cache_stats['cache_max_size']})")
    print(f"   Hit Rate: {cache_stats['cache_hit_rate']:.1%}")
    print(f"   Total Computations: {cache_stats['total_computations']}")
    
    # Performance Trends
    trends = space.threshold_manager.get_performance_trends()
    print(f"\n📈 PERFORMANCE TRENDS")
    if 'recent_utilization_avg' in trends:
        print(f"   Utilization Trend: {trends['utilization_trend_direction']}")
        print(f"   Utilization Change: {trends['utilization_change']:+.3f}")
    if 'recent_success_rate_avg' in trends:
        print(f"   Success Rate Trend: {trends['success_rate_trend_direction']}")
        print(f"   Success Rate Change: {trends['success_rate_change']:+.3f}")
    print(f"   Total Adjustments: {trends['total_adjustments']}")
    
    # System Health
    print(f"\n🏥 SYSTEM HEALTH")
    degradation_detected, reason = space.detect_performance_degradation()
    print(f"   Degradation Detected: {'Yes' if degradation_detected else 'No'}")
    if degradation_detected:
        print(f"   Reason: {reason}")
    print(f"   Fallback Active: {'Yes' if space.fallback_active else 'No'}")
    if space.fallback_active:
        print(f"   Fallback Reason: {space.fallback_reason}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    recommendations = []
    
    if metrics['current_utilization'] < 0.70:
        recommendations.append("Consider relaxing thresholds for better utilization")
    if metrics['recent_success_rate'] < 0.40:
        recommendations.append("Check threshold configuration - success rate is low")
    if cache_stats['cache_usage_ratio'] > 0.85:
        recommendations.append("Clear or reduce cache size to free memory")
    if degradation_detected and not space.fallback_active:
        recommendations.append("Consider activating fallback mechanism")
    if space.fallback_active and not degradation_detected:
        recommendations.append("Performance may have stabilized - consider deactivating fallback")
    
    if not recommendations:
        recommendations.append("System appears to be operating normally")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print("=" * 50)

# Use the debug tool
debug_performance_comprehensive(space)
```

## Best Practices Summary

### Development Best Practices

1. **Start with Optimized Configuration**: Always use the performance-optimized thresholds (0.70, 0.80, 0.5, 0.1)
2. **Enable Performance Monitoring**: Set up automatic performance monitoring and alerts
3. **Use Adaptive Thresholds**: Enable adaptive threshold adjustment for optimal utilization
4. **Monitor Cache Usage**: Implement intelligent cache management for long-running operations
5. **Profile Regularly**: Use the built-in performance profiler to monitor system performance

### Production Best Practices

1. **Conservative Settings**: Use slightly more conservative thresholds in production
2. **Fallback Mechanisms**: Always implement performance fallback for critical applications
3. **Memory Management**: Implement automatic log rotation and cache cleanup
4. **Health Monitoring**: Set up comprehensive health checks and alerting
5. **Performance Baselines**: Establish performance baselines and monitor for degradation

### Optimization Best Practices

1. **Measure First**: Always profile before optimizing
2. **Incremental Changes**: Make small, measured changes to thresholds
3. **Monitor Impact**: Track the impact of changes on both performance and utilization
4. **Document Changes**: Keep records of threshold adjustments and their effects
5. **Test Thoroughly**: Validate changes across different item sequences and scenarios

This comprehensive usage guide provides everything needed to effectively use the Improved Feasibility Mask System. For additional performance optimization details, refer to the Performance Optimization Guide.