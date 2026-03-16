"""
Performance profiler for improved feasibility mask system.

This module provides comprehensive performance analysis and optimization
tools for the enhanced stability checking algorithms.
"""

import cProfile
import pstats
import time
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'envs', 'bpp0'))

try:
    from space import Space, Box
    from support_calculation import SupportCalculator, StabilityThresholds, GeometricUtils
except ImportError as e:
    print(f"Could not import required modules: {e}")
    sys.exit(1)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    operation_name: str
    total_time: float
    iterations: int
    avg_time_per_operation: float
    operations_per_second: float
    memory_usage_mb: Optional[float] = None


class GeometricCalculationProfiler:
    """Profiler for geometric calculation performance."""
    
    def __init__(self):
        self.results = {}
        
    def profile_convex_hull_calculation(self, iterations: int = 1000) -> PerformanceMetrics:
        """Profile convex hull calculation performance."""
        print(f"Profiling convex hull calculation ({iterations} iterations)...")
        
        # Generate test data
        test_points = []
        for _ in range(iterations):
            # Generate random points for convex hull
            num_points = np.random.randint(4, 20)
            points = [(np.random.uniform(0, 10), np.random.uniform(0, 10)) 
                     for _ in range(num_points)]
            test_points.append(points)
        
        # Profile the calculation
        start_time = time.time()
        
        for points in test_points:
            hull = GeometricUtils.compute_convex_hull(points)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        metrics = PerformanceMetrics(
            operation_name="convex_hull_calculation",
            total_time=total_time,
            iterations=iterations,
            avg_time_per_operation=total_time / iterations,
            operations_per_second=iterations / total_time
        )
        
        self.results["convex_hull"] = metrics
        return metrics
    
    def profile_point_in_polygon_check(self, iterations: int = 1000) -> PerformanceMetrics:
        """Profile point-in-polygon check performance."""
        print(f"Profiling point-in-polygon check ({iterations} iterations)...")
        
        # Generate test data
        test_cases = []
        for _ in range(iterations):
            # Generate a simple polygon
            polygon = [(0, 0), (5, 0), (5, 5), (0, 5)]
            point = (np.random.uniform(-1, 6), np.random.uniform(-1, 6))
            test_cases.append((point, polygon))
        
        # Profile the calculation
        start_time = time.time()
        
        for point, polygon in test_cases:
            result = GeometricUtils.point_in_polygon(point, polygon)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        metrics = PerformanceMetrics(
            operation_name="point_in_polygon_check",
            total_time=total_time,
            iterations=iterations,
            avg_time_per_operation=total_time / iterations,
            operations_per_second=iterations / total_time
        )
        
        self.results["point_in_polygon"] = metrics
        return metrics
    
    def profile_support_area_calculation(self, iterations: int = 1000) -> PerformanceMetrics:
        """Profile weighted support area calculation performance."""
        print(f"Profiling support area calculation ({iterations} iterations)...")
        
        calculator = SupportCalculator()
        
        # Generate test data
        test_cases = []
        for _ in range(iterations):
            width, length = np.random.randint(5, 15), np.random.randint(5, 15)
            height_map = np.random.randint(0, 10, size=(width, length))
            x, y = np.random.randint(1, 5), np.random.randint(1, 5)
            lx, ly = np.random.randint(0, max(1, width-x)), np.random.randint(0, max(1, length-y))
            test_cases.append((height_map, x, y, lx, ly))
        
        # Profile the calculation
        start_time = time.time()
        
        for height_map, x, y, lx, ly in test_cases:
            result = calculator.calculate_weighted_support_area(height_map, x, y, lx, ly)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        metrics = PerformanceMetrics(
            operation_name="support_area_calculation",
            total_time=total_time,
            iterations=iterations,
            avg_time_per_operation=total_time / iterations,
            operations_per_second=iterations / total_time
        )
        
        self.results["support_area"] = metrics
        return metrics


class FeasibilityCheckProfiler:
    """Profiler for feasibility checking performance."""
    
    def __init__(self):
        self.results = {}
    
    def profile_enhanced_vs_baseline(self, iterations: int = 1000) -> Dict[str, PerformanceMetrics]:
        """Compare enhanced vs baseline feasibility checking performance."""
        print(f"Profiling enhanced vs baseline feasibility checking ({iterations} iterations)...")
        
        # Create test spaces
        enhanced_space = Space(width=10, length=10, height=20, use_enhanced_feasibility=True)
        baseline_space = Space(width=10, length=10, height=20, use_enhanced_feasibility=False)
        
        # Generate test cases
        test_cases = []
        for _ in range(iterations):
            x, y, z = np.random.randint(1, 4), np.random.randint(1, 4), np.random.randint(1, 3)
            lx, ly = np.random.randint(0, 7), np.random.randint(0, 7)
            test_cases.append((x, y, z, lx, ly))
        
        # Profile enhanced feasibility checking
        start_time = time.time()
        for x, y, z, lx, ly in test_cases:
            result = enhanced_space.check_box_enhanced(enhanced_space.plain, x, y, lx, ly, z)
        end_time = time.time()
        
        enhanced_metrics = PerformanceMetrics(
            operation_name="enhanced_feasibility_check",
            total_time=end_time - start_time,
            iterations=iterations,
            avg_time_per_operation=(end_time - start_time) / iterations,
            operations_per_second=iterations / (end_time - start_time)
        )
        
        # Profile baseline feasibility checking
        start_time = time.time()
        for x, y, z, lx, ly in test_cases:
            result = baseline_space.check_box(baseline_space.plain, x, y, lx, ly, z)
        end_time = time.time()
        
        baseline_metrics = PerformanceMetrics(
            operation_name="baseline_feasibility_check",
            total_time=end_time - start_time,
            iterations=iterations,
            avg_time_per_operation=(end_time - start_time) / iterations,
            operations_per_second=iterations / (end_time - start_time)
        )
        
        self.results["enhanced"] = enhanced_metrics
        self.results["baseline"] = baseline_metrics
        
        return {"enhanced": enhanced_metrics, "baseline": baseline_metrics}


class ThresholdOptimizer:
    """Optimizer for threshold parameters."""
    
    def __init__(self):
        self.optimization_results = {}
    
    def find_optimal_thresholds(self, test_scenarios: List[Dict]) -> StabilityThresholds:
        """
        Find optimal threshold parameters through systematic testing.
        
        Args:
            test_scenarios: List of test scenarios with different item sequences
            
        Returns:
            Optimized StabilityThresholds object
        """
        print("Finding optimal threshold parameters...")
        
        # Define parameter ranges to test
        support_area_ratios = [0.70, 0.75, 0.80, 0.85]
        corner_support_thresholds = [0.80, 0.85, 0.90, 0.95]
        height_tolerances = [0.5, 1.0, 1.5, 2.0]
        
        best_thresholds = None
        best_performance = 0.0
        
        total_combinations = len(support_area_ratios) * len(corner_support_thresholds) * len(height_tolerances)
        combination_count = 0
        
        for support_ratio in support_area_ratios:
            for corner_threshold in corner_support_thresholds:
                for height_tolerance in height_tolerances:
                    combination_count += 1
                    print(f"Testing combination {combination_count}/{total_combinations}: "
                          f"support={support_ratio}, corner={corner_threshold}, height={height_tolerance}")
                    
                    # Create test thresholds
                    test_thresholds = StabilityThresholds(
                        min_support_area_ratio=support_ratio,
                        corner_support_threshold=corner_threshold,
                        height_variation_tolerance=height_tolerance,
                        geometric_center_tolerance=0.1
                    )
                    
                    # Test performance with these thresholds
                    avg_utilization = self._test_threshold_performance(test_thresholds, test_scenarios)
                    
                    if avg_utilization > best_performance:
                        best_performance = avg_utilization
                        best_thresholds = test_thresholds
                        print(f"  New best performance: {avg_utilization:.4f}")
        
        print(f"Optimal thresholds found with performance: {best_performance:.4f}")
        print(f"  Support area ratio: {best_thresholds.min_support_area_ratio}")
        print(f"  Corner support threshold: {best_thresholds.corner_support_threshold}")
        print(f"  Height tolerance: {best_thresholds.height_variation_tolerance}")
        
        return best_thresholds
    
    def _test_threshold_performance(self, thresholds: StabilityThresholds, 
                                  test_scenarios: List[Dict]) -> float:
        """Test performance of specific threshold configuration."""
        total_utilization = 0.0
        
        for scenario in test_scenarios[:3]:  # Test on first 3 scenarios for speed
            space = Space(width=10, length=10, height=20, use_enhanced_feasibility=True)
            space.support_calculator.thresholds = thresholds
            
            # Simulate packing with the test scenario
            for item in scenario['items'][:20]:  # Test first 20 items for speed
                x, y, z = item
                
                # Try to place the item
                placed = False
                for lx in range(space.plain_size[0] - x + 1):
                    for ly in range(space.plain_size[1] - y + 1):
                        if space.drop_box((x, y, z), space.position_to_index((lx, ly)), False):
                            placed = True
                            break
                    if placed:
                        break
            
            total_utilization += space.get_ratio()
        
        return total_utilization / len(test_scenarios[:3])


def run_comprehensive_performance_analysis():
    """Run comprehensive performance analysis and optimization."""
    print("🚀 Starting Comprehensive Performance Analysis")
    print("=" * 60)
    
    results = {
        'timestamp': time.time(),
        'geometric_calculations': {},
        'feasibility_checking': {},
        'optimization_results': {}
    }
    
    # 1. Profile geometric calculations
    print("\n📐 Profiling Geometric Calculations")
    print("-" * 40)
    
    geo_profiler = GeometricCalculationProfiler()
    
    convex_hull_metrics = geo_profiler.profile_convex_hull_calculation(1000)
    print(f"Convex Hull: {convex_hull_metrics.operations_per_second:.0f} ops/sec "
          f"({convex_hull_metrics.avg_time_per_operation*1000:.4f} ms/op)")
    
    point_in_polygon_metrics = geo_profiler.profile_point_in_polygon_check(1000)
    print(f"Point-in-Polygon: {point_in_polygon_metrics.operations_per_second:.0f} ops/sec "
          f"({point_in_polygon_metrics.avg_time_per_operation*1000:.4f} ms/op)")
    
    support_area_metrics = geo_profiler.profile_support_area_calculation(1000)
    print(f"Support Area: {support_area_metrics.operations_per_second:.0f} ops/sec "
          f"({support_area_metrics.avg_time_per_operation*1000:.4f} ms/op)")
    
    results['geometric_calculations'] = {
        'convex_hull': convex_hull_metrics.__dict__,
        'point_in_polygon': point_in_polygon_metrics.__dict__,
        'support_area': support_area_metrics.__dict__
    }
    
    # 2. Profile feasibility checking
    print("\n🔍 Profiling Feasibility Checking")
    print("-" * 40)
    
    feasibility_profiler = FeasibilityCheckProfiler()
    feasibility_metrics = feasibility_profiler.profile_enhanced_vs_baseline(1000)
    
    enhanced_metrics = feasibility_metrics['enhanced']
    baseline_metrics = feasibility_metrics['baseline']
    
    print(f"Enhanced: {enhanced_metrics.operations_per_second:.0f} ops/sec "
          f"({enhanced_metrics.avg_time_per_operation*1000:.4f} ms/op)")
    print(f"Baseline: {baseline_metrics.operations_per_second:.0f} ops/sec "
          f"({baseline_metrics.avg_time_per_operation*1000:.4f} ms/op)")
    
    overhead_pct = ((enhanced_metrics.avg_time_per_operation - baseline_metrics.avg_time_per_operation) 
                   / baseline_metrics.avg_time_per_operation * 100)
    print(f"Enhanced overhead: {overhead_pct:.1f}%")
    
    results['feasibility_checking'] = {
        'enhanced': enhanced_metrics.__dict__,
        'baseline': baseline_metrics.__dict__,
        'overhead_percentage': overhead_pct
    }
    
    # 3. Generate test scenarios for optimization
    print("\n⚙️ Generating Test Scenarios for Optimization")
    print("-" * 40)
    
    test_scenarios = []
    for i in range(5):
        items = []
        for _ in range(30):
            x = np.random.randint(1, 4)
            y = np.random.randint(1, 4)
            z = np.random.randint(1, 3)
            items.append((x, y, z))
        test_scenarios.append({'items': items})
    
    print(f"Generated {len(test_scenarios)} test scenarios with {len(test_scenarios[0]['items'])} items each")
    
    # 4. Optimize thresholds
    print("\n🎯 Optimizing Threshold Parameters")
    print("-" * 40)
    
    optimizer = ThresholdOptimizer()
    optimal_thresholds = optimizer.find_optimal_thresholds(test_scenarios)
    
    results['optimization_results'] = {
        'optimal_thresholds': {
            'min_support_area_ratio': optimal_thresholds.min_support_area_ratio,
            'corner_support_threshold': optimal_thresholds.corner_support_threshold,
            'height_variation_tolerance': optimal_thresholds.height_variation_tolerance,
            'geometric_center_tolerance': optimal_thresholds.geometric_center_tolerance
        }
    }
    
    # 5. Save results
    print("\n💾 Saving Performance Analysis Results")
    print("-" * 40)
    
    results_file = f"performance_analysis_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # 6. Generate performance summary
    print("\n📊 Performance Analysis Summary")
    print("=" * 60)
    
    print(f"Geometric Calculations Performance:")
    print(f"  • Convex Hull: {convex_hull_metrics.operations_per_second:.0f} ops/sec")
    print(f"  • Point-in-Polygon: {point_in_polygon_metrics.operations_per_second:.0f} ops/sec")
    print(f"  • Support Area: {support_area_metrics.operations_per_second:.0f} ops/sec")
    
    print(f"\nFeasibility Checking Performance:")
    print(f"  • Enhanced: {enhanced_metrics.operations_per_second:.0f} ops/sec")
    print(f"  • Baseline: {baseline_metrics.operations_per_second:.0f} ops/sec")
    print(f"  • Overhead: {overhead_pct:.1f}%")
    
    print(f"\nOptimal Thresholds:")
    print(f"  • Support Area Ratio: {optimal_thresholds.min_support_area_ratio}")
    print(f"  • Corner Support: {optimal_thresholds.corner_support_threshold}")
    print(f"  • Height Tolerance: {optimal_thresholds.height_variation_tolerance}")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_performance_analysis()