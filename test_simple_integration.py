#!/usr/bin/env python3
"""
Simple integration test for improved feasibility mask with cut_2 dataset simulation.

This script tests the complete system without complex dependencies.
"""

import sys
import os
import numpy as np
import time
import json
from pathlib import Path

# Add paths for imports
sys.path.append('acktr')
sys.path.append('envs/bpp0')

try:
    from space import Space, Box
    from support_calculation import SupportCalculator, StabilityThresholds, GeometricUtils, ThresholdManager
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)


def generate_cut2_like_dataset(num_sequences=10, items_per_sequence=30):
    """Generate synthetic dataset similar to cut_2 characteristics."""
    np.random.seed(42)  # For reproducible results
    
    sequences = []
    for seq_idx in range(num_sequences):
        sequence = []
        for _ in range(items_per_sequence):
            # Generate items with more challenging sizes for better testing
            # Mix of small, medium, and large items
            size_type = np.random.choice(['small', 'medium', 'large'], p=[0.4, 0.4, 0.2])
            
            if size_type == 'small':
                x = np.random.randint(2, 4)  # Width: 2-3
                y = np.random.randint(2, 4)  # Length: 2-3  
                z = np.random.randint(2, 4)  # Height: 2-3
            elif size_type == 'medium':
                x = np.random.randint(3, 5)  # Width: 3-4
                y = np.random.randint(3, 5)  # Length: 3-4  
                z = np.random.randint(3, 5)  # Height: 3-4
            else:  # large
                x = np.random.randint(4, 6)  # Width: 4-5
                y = np.random.randint(4, 6)  # Length: 4-5  
                z = np.random.randint(3, 6)  # Height: 3-5
            
            sequence.append((x, y, z))
        sequences.append(sequence)
    
    return sequences


def run_packing_test(item_sequence, container_size=(10, 10, 10), use_enhanced=True):
    """Run a packing test with given item sequence."""
    width, length, height = container_size
    space = Space(width=width, length=length, height=height, 
                  use_enhanced_feasibility=use_enhanced)
    
    placed_items = 0
    failed_placements = 0
    placement_attempts = 0
    
    start_time = time.time()
    
    for item_idx, (x, y, z) in enumerate(item_sequence):
        placed = False
        
        # Use a more sophisticated placement strategy
        # Try positions in order of preference (corners first, then edges, then center)
        positions = []
        
        # Add corner positions first
        for lx in [0, max(0, width - x)]:
            for ly in [0, max(0, length - y)]:
                if lx + x <= width and ly + y <= length:
                    positions.append((lx, ly))
        
        # Add edge positions
        for lx in range(1, width - x):
            for ly in [0, max(0, length - y)]:
                if lx + x <= width and ly + y <= length:
                    positions.append((lx, ly))
        for lx in [0, max(0, width - x)]:
            for ly in range(1, length - y):
                if lx + x <= width and ly + y <= length:
                    positions.append((lx, ly))
        
        # Add remaining positions
        for lx in range(width - x + 1):
            for ly in range(length - y + 1):
                if (lx, ly) not in positions:
                    positions.append((lx, ly))
        
        # Try each position
        for lx, ly in positions:
            placement_attempts += 1
            
            if space.drop_box([x, y, z], space.position_to_index([lx, ly]), False):
                placed_items += 1
                placed = True
                break
        
        if not placed:
            failed_placements += 1
        
        # Stop if container is getting very full
        if space.get_ratio() > 0.95:
            break
    
    end_time = time.time()
    
    # Calculate results
    final_utilization = space.get_ratio()
    success_rate = placed_items / len(item_sequence) if item_sequence else 0.0
    performance_metrics = space.collect_utilization_metrics()
    
    return {
        'final_utilization': final_utilization,
        'success_rate': success_rate,
        'placed_items': placed_items,
        'failed_placements': failed_placements,
        'total_items': len(item_sequence),
        'placement_attempts': placement_attempts,
        'execution_time': end_time - start_time,
        'threshold_adjustments': performance_metrics['threshold_adjustments'],
        'fallback_used': bool(space.fallback_active),  # Convert to bool for JSON serialization
        'performance_metrics': {
            'current_utilization': float(performance_metrics['current_utilization']),
            'target_utilization': float(performance_metrics['target_utilization']),
            'utilization_gap': float(performance_metrics['utilization_gap']),
            'recent_success_rate': float(performance_metrics['recent_success_rate']),
            'threshold_adjustments': int(performance_metrics['threshold_adjustments']),
            'fallback_active': bool(performance_metrics['fallback_active'])
        }
    }


def main():
    """Run comprehensive integration tests."""
    print("🚀 Starting Cut_2 Dataset Integration Tests")
    print("=" * 60)
    
    # Generate test dataset
    print("📊 Generating synthetic cut_2-like dataset...")
    test_sequences = generate_cut2_like_dataset(num_sequences=10, items_per_sequence=25)
    print(f"   Generated {len(test_sequences)} sequences with {len(test_sequences[0])} items each")
    
    # Test baseline performance
    print("\n🔍 Testing baseline performance (original feasibility checking)...")
    baseline_results = []
    
    for seq_idx, sequence in enumerate(test_sequences):
        result = run_packing_test(sequence, use_enhanced=False)
        baseline_results.append(result)
        print(f"   Sequence {seq_idx+1}: {result['final_utilization']:.3f} utilization, "
              f"{result['success_rate']:.3f} success rate")
    
    baseline_avg_util = np.mean([r['final_utilization'] for r in baseline_results])
    baseline_avg_success = np.mean([r['success_rate'] for r in baseline_results])
    
    print(f"\n📈 Baseline Results:")
    print(f"   Average Utilization: {baseline_avg_util:.3f}")
    print(f"   Average Success Rate: {baseline_avg_success:.3f}")
    
    # Test enhanced performance
    print("\n⚡ Testing enhanced performance (improved feasibility checking)...")
    enhanced_results = []
    
    for seq_idx, sequence in enumerate(test_sequences):
        result = run_packing_test(sequence, use_enhanced=True)
        enhanced_results.append(result)
        print(f"   Sequence {seq_idx+1}: {result['final_utilization']:.3f} utilization, "
              f"{result['success_rate']:.3f} success rate, "
              f"{result['threshold_adjustments']} adjustments")
    
    enhanced_avg_util = np.mean([r['final_utilization'] for r in enhanced_results])
    enhanced_avg_success = np.mean([r['success_rate'] for r in enhanced_results])
    enhanced_avg_adjustments = np.mean([r['threshold_adjustments'] for r in enhanced_results])
    
    print(f"\n🎯 Enhanced Results:")
    print(f"   Average Utilization: {enhanced_avg_util:.3f}")
    print(f"   Average Success Rate: {enhanced_avg_success:.3f}")
    print(f"   Average Threshold Adjustments: {enhanced_avg_adjustments:.1f}")
    
    # Calculate improvements
    util_improvement = enhanced_avg_util - baseline_avg_util
    util_improvement_pct = (util_improvement / baseline_avg_util) * 100 if baseline_avg_util > 0 else 0
    success_improvement = enhanced_avg_success - baseline_avg_success
    
    print(f"\n📊 Performance Comparison:")
    print(f"   Utilization Improvement: +{util_improvement:.3f} ({util_improvement_pct:.1f}%)")
    print(f"   Success Rate Improvement: +{success_improvement:.3f}")
    
    # Validate requirements
    print(f"\n✅ Requirement Validation:")
    
    # Requirement 5.1: Enhanced system should achieve >75% utilization
    target_achieved = enhanced_avg_util >= 0.75
    print(f"   Target Utilization (≥75%): {'✅ PASS' if target_achieved else '❌ FAIL'} "
          f"({enhanced_avg_util:.3f})")
    
    # Requirement 5.2: Enhanced system should improve over baseline
    improvement_achieved = util_improvement > 0
    print(f"   Performance Improvement: {'✅ PASS' if improvement_achieved else '❌ FAIL'} "
          f"({util_improvement:+.3f})")
    
    # Meaningful improvement (at least 5% relative)
    meaningful_improvement = util_improvement_pct >= 5.0
    print(f"   Meaningful Improvement (≥5%): {'✅ PASS' if meaningful_improvement else '❌ FAIL'} "
          f"({util_improvement_pct:.1f}%)")
    
    # Test system stability
    print(f"\n🔧 Testing System Stability:")
    
    # Test with different container sizes
    stability_results = []
    container_sizes = [(8, 8, 8), (10, 10, 10), (12, 12, 12)]
    
    for container_size in container_sizes:
        result = run_packing_test(test_sequences[0], container_size=container_size, use_enhanced=True)
        stability_results.append(result)
        print(f"   Container {container_size}: {result['final_utilization']:.3f} utilization")
    
    stability_utilizations = [r['final_utilization'] for r in stability_results]
    stability_std = np.std(stability_utilizations)
    stability_min = min(stability_utilizations)
    
    stable_performance = stability_std <= 0.25 and stability_min >= 0.40
    print(f"   System Stability: {'✅ PASS' if stable_performance else '❌ FAIL'} "
          f"(std: {stability_std:.3f}, min: {stability_min:.3f})")
    
    # Test adaptive threshold behavior
    print(f"\n🎛️  Testing Adaptive Thresholds:")
    
    # Create a challenging scenario
    space = Space(width=8, length=8, height=8, use_enhanced_feasibility=True)
    space.target_utilization = 0.85  # High target to trigger adjustments
    
    # Place some items to trigger threshold adjustments
    test_items = [(3, 3, 3)] * 10
    placed = 0
    
    for x, y, z in test_items:
        for lx in range(8 - x + 1):
            for ly in range(8 - y + 1):
                if space.drop_box([x, y, z], space.position_to_index([lx, ly]), False):
                    placed += 1
                    break
            else:
                continue
            break
        
        # Trigger monitoring every few items
        if placed % 3 == 0:
            space.monitor_and_adjust_performance()
    
    final_metrics = space.collect_utilization_metrics()
    adjustments_made = final_metrics['threshold_adjustments'] > 0
    
    print(f"   Threshold Adjustments: {'✅ ACTIVE' if adjustments_made else '⚠️  NONE'} "
          f"({final_metrics['threshold_adjustments']} adjustments)")
    print(f"   Fallback Mechanism: {'🔄 ACTIVE' if space.fallback_active else '✅ STABLE'}")
    
    # Save detailed results
    results_data = {
        'test_summary': {
            'baseline_avg_utilization': baseline_avg_util,
            'enhanced_avg_utilization': enhanced_avg_util,
            'utilization_improvement': util_improvement,
            'utilization_improvement_pct': util_improvement_pct,
            'target_achieved': target_achieved,
            'improvement_achieved': improvement_achieved,
            'meaningful_improvement': meaningful_improvement,
            'stable_performance': stable_performance
        },
        'baseline_results': baseline_results,
        'enhanced_results': enhanced_results,
        'stability_results': stability_results,
        'test_metadata': {
            'timestamp': time.time(),
            'num_sequences': len(test_sequences),
            'items_per_sequence': len(test_sequences[0]),
            'container_sizes_tested': container_sizes
        }
    }
    
    # Save results to file
    results_dir = Path("integration_test_results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"cut2_integration_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Final summary
    print(f"\n🏁 Final Test Results:")
    print(f"=" * 60)
    
    all_tests_passed = (target_achieved and improvement_achieved and 
                       meaningful_improvement and stable_performance)
    
    if all_tests_passed:
        print(f"🎉 ALL TESTS PASSED!")
        print(f"   ✅ Enhanced system achieves {enhanced_avg_util:.1%} utilization (target: ≥75%)")
        print(f"   ✅ Improvement of {util_improvement_pct:.1f}% over baseline")
        print(f"   ✅ System demonstrates stability across configurations")
        return True
    else:
        print(f"⚠️  SOME TESTS FAILED!")
        if not target_achieved:
            print(f"   ❌ Target utilization not met: {enhanced_avg_util:.1%} < 75%")
        if not improvement_achieved:
            print(f"   ❌ No improvement over baseline")
        if not meaningful_improvement:
            print(f"   ❌ Improvement not meaningful: {util_improvement_pct:.1f}% < 5%")
        if not stable_performance:
            print(f"   ❌ System performance not stable")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)