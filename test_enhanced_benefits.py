#!/usr/bin/env python3
"""
Targeted test to demonstrate the benefits of enhanced feasibility checking.

This test creates specific scenarios where enhanced feasibility should significantly
outperform baseline feasibility checking.
"""

import sys
import os
import numpy as np
import time

# Add paths for imports
sys.path.append('acktr')
sys.path.append('envs/bpp0')

from space import Space, Box

def create_challenging_scenario_1():
    """
    Create a scenario where enhanced feasibility should excel:
    Uneven surface with good support area but poor corner alignment.
    """
    space_baseline = Space(width=10, length=10, height=15, use_enhanced_feasibility=False)
    space_enhanced = Space(width=10, length=10, height=15, use_enhanced_feasibility=True)
    
    # Create an uneven surface by placing some initial boxes
    initial_boxes = [
        (2, 2, 3, 1, 1),  # (x, y, z, lx, ly)
        (2, 2, 3, 3, 1),
        (2, 2, 3, 5, 1),
        (2, 2, 3, 7, 1),
        (2, 2, 3, 1, 3),
        (2, 2, 3, 3, 3),
        (2, 2, 3, 5, 3),
        (2, 2, 3, 7, 3),
        (2, 2, 3, 1, 5),
        (2, 2, 3, 3, 5),
        (2, 2, 3, 5, 5),
        (2, 2, 3, 7, 5),
    ]
    
    # Place initial boxes in both spaces
    for x, y, z, lx, ly in initial_boxes:
        space_baseline.drop_box([x, y, z], space_baseline.position_to_index([lx, ly]), False)
        space_enhanced.drop_box([x, y, z], space_enhanced.position_to_index([lx, ly]), False)
    
    print("Scenario 1: Uneven surface with good support area")
    print(f"Initial utilization: {space_baseline.get_ratio():.3f}")
    
    # Now try to place items that should benefit from enhanced feasibility
    test_items = [
        (4, 4, 2),  # Large item that spans multiple support areas
        (3, 3, 2),
        (4, 3, 2),
        (3, 4, 2),
        (5, 3, 2),
        (3, 5, 2),
    ]
    
    baseline_placed = 0
    enhanced_placed = 0
    
    for item in test_items:
        x, y, z = item
        
        # Try baseline placement
        baseline_success = False
        for lx in range(10 - x + 1):
            for ly in range(10 - y + 1):
                if space_baseline.drop_box([x, y, z], space_baseline.position_to_index([lx, ly]), False):
                    baseline_placed += 1
                    baseline_success = True
                    break
            if baseline_success:
                break
        
        # Try enhanced placement
        enhanced_success = False
        for lx in range(10 - x + 1):
            for ly in range(10 - y + 1):
                if space_enhanced.drop_box([x, y, z], space_enhanced.position_to_index([lx, ly]), False):
                    enhanced_placed += 1
                    enhanced_success = True
                    break
            if enhanced_success:
                break
        
        print(f"  Item {x}x{y}x{z}: Baseline={'✅' if baseline_success else '❌'}, Enhanced={'✅' if enhanced_success else '❌'}")
    
    baseline_final = space_baseline.get_ratio()
    enhanced_final = space_enhanced.get_ratio()
    
    print(f"Final utilization - Baseline: {baseline_final:.3f}, Enhanced: {enhanced_final:.3f}")
    print(f"Items placed - Baseline: {baseline_placed}/{len(test_items)}, Enhanced: {enhanced_placed}/{len(test_items)}")
    print(f"Improvement: {enhanced_final - baseline_final:.3f} ({((enhanced_final - baseline_final) / baseline_final * 100):.1f}%)")
    
    return baseline_final, enhanced_final, baseline_placed, enhanced_placed


def create_challenging_scenario_2():
    """
    Create a scenario with partial support that should benefit from 
    weighted support area calculation.
    """
    print("\nScenario 2: Partial support with weighted area calculation")
    
    space_baseline = Space(width=12, length=12, height=15, use_enhanced_feasibility=False)
    space_enhanced = Space(width=12, length=12, height=15, use_enhanced_feasibility=True)
    
    # Create a stepped surface
    step_boxes = [
        # Bottom level
        (3, 3, 2, 0, 0), (3, 3, 2, 3, 0), (3, 3, 2, 6, 0), (3, 3, 2, 9, 0),
        (3, 3, 2, 0, 3), (3, 3, 2, 3, 3), (3, 3, 2, 6, 3), (3, 3, 2, 9, 3),
        (3, 3, 2, 0, 6), (3, 3, 2, 3, 6), (3, 3, 2, 6, 6), (3, 3, 2, 9, 6),
        (3, 3, 2, 0, 9), (3, 3, 2, 3, 9), (3, 3, 2, 6, 9), (3, 3, 2, 9, 9),
        
        # Second level (partial coverage)
        (3, 3, 2, 1, 1), (3, 3, 2, 4, 1), (3, 3, 2, 7, 1),
        (3, 3, 2, 1, 4), (3, 3, 2, 4, 4), (3, 3, 2, 7, 4),
        (3, 3, 2, 1, 7), (3, 3, 2, 4, 7), (3, 3, 2, 7, 7),
    ]
    
    # Place step boxes
    for x, y, z, lx, ly in step_boxes:
        space_baseline.drop_box([x, y, z], space_baseline.position_to_index([lx, ly]), False)
        space_enhanced.drop_box([x, y, z], space_enhanced.position_to_index([lx, ly]), False)
    
    print(f"Initial utilization: {space_baseline.get_ratio():.3f}")
    
    # Try to place items that span multiple height levels
    test_items = [
        (6, 6, 2),  # Large item spanning multiple levels
        (5, 5, 2),
        (4, 6, 2),
        (6, 4, 2),
        (7, 5, 2),
        (5, 7, 2),
    ]
    
    baseline_placed = 0
    enhanced_placed = 0
    
    for item in test_items:
        x, y, z = item
        
        # Try baseline placement
        baseline_success = False
        for lx in range(12 - x + 1):
            for ly in range(12 - y + 1):
                if space_baseline.drop_box([x, y, z], space_baseline.position_to_index([lx, ly]), False):
                    baseline_placed += 1
                    baseline_success = True
                    break
            if baseline_success:
                break
        
        # Try enhanced placement
        enhanced_success = False
        for lx in range(12 - x + 1):
            for ly in range(12 - y + 1):
                if space_enhanced.drop_box([x, y, z], space_enhanced.position_to_index([lx, ly]), False):
                    enhanced_placed += 1
                    enhanced_success = True
                    break
            if enhanced_success:
                break
        
        print(f"  Item {x}x{y}x{z}: Baseline={'✅' if baseline_success else '❌'}, Enhanced={'✅' if enhanced_success else '❌'}")
    
    baseline_final = space_baseline.get_ratio()
    enhanced_final = space_enhanced.get_ratio()
    
    print(f"Final utilization - Baseline: {baseline_final:.3f}, Enhanced: {enhanced_final:.3f}")
    print(f"Items placed - Baseline: {baseline_placed}/{len(test_items)}, Enhanced: {enhanced_placed}/{len(test_items)}")
    print(f"Improvement: {enhanced_final - baseline_final:.3f} ({((enhanced_final - baseline_final) / baseline_final * 100):.1f}%)")
    
    return baseline_final, enhanced_final, baseline_placed, enhanced_placed


def create_challenging_scenario_3():
    """
    Create a scenario that tests geometric center validation.
    """
    print("\nScenario 3: Geometric center validation with corner supports")
    
    space_baseline = Space(width=10, length=10, height=15, use_enhanced_feasibility=False)
    space_enhanced = Space(width=10, length=10, height=15, use_enhanced_feasibility=True)
    
    # Create corner supports at different heights
    corner_supports = [
        (1, 1, 4, 2, 2),  # Corner support at (2,2)
        (1, 1, 4, 6, 2),  # Corner support at (6,2)
        (1, 1, 4, 2, 6),  # Corner support at (2,6)
        (1, 1, 4, 6, 6),  # Corner support at (6,6)
        
        # Add some intermediate supports
        (1, 1, 3, 4, 2),
        (1, 1, 3, 2, 4),
        (1, 1, 3, 6, 4),
        (1, 1, 3, 4, 6),
    ]
    
    # Place corner supports
    for x, y, z, lx, ly in corner_supports:
        space_baseline.drop_box([x, y, z], space_baseline.position_to_index([lx, ly]), False)
        space_enhanced.drop_box([x, y, z], space_enhanced.position_to_index([lx, ly]), False)
    
    print(f"Initial utilization: {space_baseline.get_ratio():.3f}")
    
    # Try to place items that should benefit from geometric center validation
    test_items = [
        (5, 5, 2),  # Item that spans the corner supports
        (4, 4, 2),
        (6, 4, 2),
        (4, 6, 2),
        (3, 5, 2),
        (5, 3, 2),
    ]
    
    baseline_placed = 0
    enhanced_placed = 0
    
    for item in test_items:
        x, y, z = item
        
        # Try baseline placement
        baseline_success = False
        for lx in range(10 - x + 1):
            for ly in range(10 - y + 1):
                if space_baseline.drop_box([x, y, z], space_baseline.position_to_index([lx, ly]), False):
                    baseline_placed += 1
                    baseline_success = True
                    break
            if baseline_success:
                break
        
        # Try enhanced placement
        enhanced_success = False
        for lx in range(10 - x + 1):
            for ly in range(10 - y + 1):
                if space_enhanced.drop_box([x, y, z], space_enhanced.position_to_index([lx, ly]), False):
                    enhanced_placed += 1
                    enhanced_success = True
                    break
            if enhanced_success:
                break
        
        print(f"  Item {x}x{y}x{z}: Baseline={'✅' if baseline_success else '❌'}, Enhanced={'✅' if enhanced_success else '❌'}")
    
    baseline_final = space_baseline.get_ratio()
    enhanced_final = space_enhanced.get_ratio()
    
    print(f"Final utilization - Baseline: {baseline_final:.3f}, Enhanced: {enhanced_final:.3f}")
    print(f"Items placed - Baseline: {baseline_placed}/{len(test_items)}, Enhanced: {enhanced_placed}/{len(test_items)}")
    print(f"Improvement: {enhanced_final - baseline_final:.3f} ({((enhanced_final - baseline_final) / baseline_final * 100):.1f}%)")
    
    return baseline_final, enhanced_final, baseline_placed, enhanced_placed


def test_threshold_adaptation():
    """Test adaptive threshold behavior under challenging conditions."""
    print("\nScenario 4: Adaptive threshold behavior")
    
    space = Space(width=8, length=8, height=12, use_enhanced_feasibility=True)
    space.target_utilization = 0.80  # High target to trigger adjustments
    
    # Create a challenging packing scenario
    challenging_items = [
        (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3),  # Large items first
        (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2),  # Medium items
        (4, 2, 2), (2, 4, 2), (3, 2, 2), (2, 3, 2),  # Irregular shapes
    ]
    
    placed_items = 0
    initial_thresholds = space.get_current_thresholds()
    
    print(f"Initial thresholds: support_area={initial_thresholds.min_support_area_ratio:.3f}")
    
    for item_idx, (x, y, z) in enumerate(challenging_items):
        placed = False
        
        # Try to place item
        for lx in range(8 - x + 1):
            for ly in range(8 - y + 1):
                if space.drop_box([x, y, z], space.position_to_index([lx, ly]), False):
                    placed_items += 1
                    placed = True
                    break
            if placed:
                break
        
        # Monitor and adjust every few items
        if (item_idx + 1) % 3 == 0:
            monitoring_result = space.monitor_and_adjust_performance()
            current_thresholds = space.get_current_thresholds()
            
            if monitoring_result['threshold_adjusted']:
                print(f"  After item {item_idx + 1}: Thresholds adjusted")
                print(f"    New support_area threshold: {current_thresholds.min_support_area_ratio:.3f}")
            
            if monitoring_result['fallback_activated']:
                print(f"  After item {item_idx + 1}: Fallback activated")
    
    final_utilization = space.get_ratio()
    final_metrics = space.collect_utilization_metrics()
    final_thresholds = space.get_current_thresholds()
    
    print(f"Final results:")
    print(f"  Utilization: {final_utilization:.3f}")
    print(f"  Items placed: {placed_items}/{len(challenging_items)}")
    print(f"  Threshold adjustments: {final_metrics['threshold_adjustments']}")
    print(f"  Final support_area threshold: {final_thresholds.min_support_area_ratio:.3f}")
    print(f"  Fallback active: {space.fallback_active}")
    
    return final_utilization, placed_items, final_metrics['threshold_adjustments']


def main():
    """Run all targeted benefit tests."""
    print("🎯 Testing Enhanced Feasibility Benefits")
    print("=" * 60)
    
    # Run challenging scenarios
    results = []
    
    # Scenario 1: Uneven surfaces
    baseline1, enhanced1, placed_b1, placed_e1 = create_challenging_scenario_1()
    results.append(('Uneven Surface', baseline1, enhanced1, placed_b1, placed_e1))
    
    # Scenario 2: Partial support
    baseline2, enhanced2, placed_b2, placed_e2 = create_challenging_scenario_2()
    results.append(('Partial Support', baseline2, enhanced2, placed_b2, placed_e2))
    
    # Scenario 3: Geometric center validation
    baseline3, enhanced3, placed_b3, placed_e3 = create_challenging_scenario_3()
    results.append(('Geometric Center', baseline3, enhanced3, placed_b3, placed_e3))
    
    # Scenario 4: Adaptive thresholds
    final_util, placed_items, adjustments = test_threshold_adaptation()
    
    # Summary
    print("\n📊 Summary of Results")
    print("=" * 60)
    
    total_baseline_util = 0
    total_enhanced_util = 0
    total_baseline_placed = 0
    total_enhanced_placed = 0
    
    for scenario, baseline, enhanced, placed_b, placed_e in results:
        improvement = enhanced - baseline
        improvement_pct = (improvement / baseline * 100) if baseline > 0 else 0
        
        print(f"{scenario}:")
        print(f"  Baseline: {baseline:.3f} utilization, {placed_b} items")
        print(f"  Enhanced: {enhanced:.3f} utilization, {placed_e} items")
        print(f"  Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")
        
        total_baseline_util += baseline
        total_enhanced_util += enhanced
        total_baseline_placed += placed_b
        total_enhanced_placed += placed_e
    
    # Overall improvement
    avg_baseline = total_baseline_util / len(results)
    avg_enhanced = total_enhanced_util / len(results)
    overall_improvement = avg_enhanced - avg_baseline
    overall_improvement_pct = (overall_improvement / avg_baseline * 100) if avg_baseline > 0 else 0
    
    print(f"\nOverall Performance:")
    print(f"  Average Baseline: {avg_baseline:.3f}")
    print(f"  Average Enhanced: {avg_enhanced:.3f}")
    print(f"  Overall Improvement: {overall_improvement:.3f} ({overall_improvement_pct:.1f}%)")
    print(f"  Total Items - Baseline: {total_baseline_placed}, Enhanced: {total_enhanced_placed}")
    
    print(f"\nAdaptive Behavior:")
    print(f"  Final utilization with adaptation: {final_util:.3f}")
    print(f"  Items placed with adaptation: {placed_items}")
    print(f"  Threshold adjustments made: {adjustments}")
    
    # Validate requirements
    print(f"\n✅ Requirement Validation:")
    
    # Check if enhanced system shows meaningful improvement
    meaningful_improvement = overall_improvement_pct >= 5.0
    print(f"  Meaningful Improvement (≥5%): {'✅ PASS' if meaningful_improvement else '❌ FAIL'} "
          f"({overall_improvement_pct:.1f}%)")
    
    # Check if enhanced system places more items
    more_items_placed = total_enhanced_placed > total_baseline_placed
    print(f"  More Items Placed: {'✅ PASS' if more_items_placed else '❌ FAIL'} "
          f"(+{total_enhanced_placed - total_baseline_placed} items)")
    
    # Check adaptive behavior
    adaptive_behavior = adjustments > 0
    print(f"  Adaptive Behavior: {'✅ ACTIVE' if adaptive_behavior else '⚠️  INACTIVE'} "
          f"({adjustments} adjustments)")
    
    # Check if any scenario achieves high utilization
    high_utilization_achieved = any(enhanced >= 0.75 for _, _, enhanced, _, _ in results)
    print(f"  High Utilization Achieved: {'✅ YES' if high_utilization_achieved else '❌ NO'}")
    
    success = meaningful_improvement and more_items_placed
    
    if success:
        print(f"\n🎉 Enhanced feasibility checking demonstrates clear benefits!")
        print(f"   The system shows {overall_improvement_pct:.1f}% improvement in challenging scenarios")
        print(f"   and successfully places {total_enhanced_placed - total_baseline_placed} more items")
    else:
        print(f"\n⚠️  Enhanced feasibility checking shows limited benefits in these tests")
        print(f"   Consider adjusting thresholds or testing with more challenging scenarios")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)