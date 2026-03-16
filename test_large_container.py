#!/usr/bin/env python3
"""
Test with larger containers to better demonstrate the benefits of enhanced feasibility checking.
"""

import sys
import os
sys.path.append('envs/bpp0')

import numpy as np
from space import Space
import time

def test_large_container_performance():
    """Test with larger containers that should better show the benefits."""
    print("🚀 Testing Enhanced Feasibility with Larger Containers")
    print("=" * 60)
    
    # Test with larger container sizes
    container_sizes = [
        (20, 20, 20),  # Original size
        (25, 25, 25),  # Larger
        (30, 30, 30),  # Even larger
    ]
    
    # Generate more realistic item sequences
    def generate_realistic_items(num_items=50):
        """Generate realistic item sizes for larger containers."""
        items = []
        np.random.seed(42)  # For reproducible results
        
        for _ in range(num_items):
            # Generate items with realistic size distributions
            size_type = np.random.choice(['small', 'medium', 'large'], p=[0.5, 0.3, 0.2])
            
            if size_type == 'small':
                x = np.random.randint(2, 6)
                y = np.random.randint(2, 6)
                z = np.random.randint(1, 4)
            elif size_type == 'medium':
                x = np.random.randint(4, 10)
                y = np.random.randint(4, 10)
                z = np.random.randint(2, 6)
            else:  # large
                x = np.random.randint(6, 15)
                y = np.random.randint(6, 15)
                z = np.random.randint(3, 8)
            
            items.append((x, y, z))
        
        return items
    
    def run_packing_test(items, container_size, use_enhanced=True):
        """Run packing test with given parameters."""
        width, length, height = container_size
        space = Space(width=width, length=length, height=height, 
                     use_enhanced_feasibility=use_enhanced)
        
        placed_items = 0
        total_attempts = 0
        
        for item_x, item_y, item_z in items:
            placed = False
            
            # Try to place the item at various positions
            for lx in range(max(1, width - item_x + 1)):
                for ly in range(max(1, length - item_y + 1)):
                    total_attempts += 1
                    
                    if use_enhanced:
                        result = space.check_box_enhanced(space.plain, item_x, item_y, lx, ly, item_z)
                    else:
                        result = space.check_box(space.plain, item_x, item_y, lx, ly, item_z)
                    
                    if result >= 0:
                        # Place the item
                        success = space.drop_box((item_x, item_y, item_z), 
                                               lx * length + ly, False)
                        if success:
                            placed_items += 1
                            placed = True
                            break
                
                if placed:
                    break
        
        final_utilization = space.get_ratio()
        success_rate = placed_items / len(items) if items else 0
        
        return {
            'final_utilization': final_utilization,
            'items_placed': placed_items,
            'total_items': len(items),
            'success_rate': success_rate,
            'total_attempts': total_attempts
        }
    
    # Test each container size
    all_results = []
    
    for container_size in container_sizes:
        print(f"\n📦 Testing Container Size: {container_size}")
        print("-" * 40)
        
        # Generate items for this container size
        items = generate_realistic_items(50)
        
        # Test baseline
        print("🔍 Baseline performance...")
        baseline_result = run_packing_test(items, container_size, use_enhanced=False)
        
        # Test enhanced
        print("⚡ Enhanced performance...")
        enhanced_result = run_packing_test(items, container_size, use_enhanced=True)
        
        # Calculate improvements
        util_improvement = enhanced_result['final_utilization'] - baseline_result['final_utilization']
        util_improvement_pct = (util_improvement / baseline_result['final_utilization']) * 100 if baseline_result['final_utilization'] > 0 else 0
        
        items_improvement = enhanced_result['items_placed'] - baseline_result['items_placed']
        
        print(f"📊 Results for {container_size}:")
        print(f"   Baseline:  {baseline_result['final_utilization']:.3f} utilization, {baseline_result['items_placed']} items")
        print(f"   Enhanced:  {enhanced_result['final_utilization']:.3f} utilization, {enhanced_result['items_placed']} items")
        print(f"   Improvement: +{util_improvement:.3f} ({util_improvement_pct:.1f}%), +{items_improvement} items")
        
        # Check if target achieved
        target_achieved = enhanced_result['final_utilization'] >= 0.75
        print(f"   Target ≥75%: {'✅ ACHIEVED' if target_achieved else '❌ NOT MET'}")
        
        all_results.append({
            'container_size': container_size,
            'baseline': baseline_result,
            'enhanced': enhanced_result,
            'improvement': util_improvement,
            'improvement_pct': util_improvement_pct,
            'items_improvement': items_improvement,
            'target_achieved': target_achieved
        })
    
    # Overall summary
    print(f"\n🏁 Overall Summary")
    print("=" * 60)
    
    avg_improvement = sum(r['improvement'] for r in all_results) / len(all_results)
    avg_improvement_pct = sum(r['improvement_pct'] for r in all_results) / len(all_results)
    total_items_improvement = sum(r['items_improvement'] for r in all_results)
    targets_achieved = sum(1 for r in all_results if r['target_achieved'])
    
    print(f"Average Utilization Improvement: +{avg_improvement:.3f} ({avg_improvement_pct:.1f}%)")
    print(f"Total Additional Items Placed: +{total_items_improvement}")
    print(f"Containers Achieving ≥75%: {targets_achieved}/{len(all_results)}")
    
    # Best performing container
    best_result = max(all_results, key=lambda r: r['enhanced']['final_utilization'])
    print(f"\nBest Performance:")
    print(f"   Container: {best_result['container_size']}")
    print(f"   Utilization: {best_result['enhanced']['final_utilization']:.3f}")
    print(f"   Items Placed: {best_result['enhanced']['items_placed']}")
    
    # Check overall success
    meaningful_improvement = avg_improvement_pct >= 5.0
    any_target_achieved = targets_achieved > 0
    
    print(f"\n✅ Final Assessment:")
    print(f"   Meaningful Improvement (≥5%): {'✅ PASS' if meaningful_improvement else '❌ FAIL'} ({avg_improvement_pct:.1f}%)")
    print(f"   Target Achievement: {'✅ PASS' if any_target_achieved else '❌ FAIL'} ({targets_achieved} containers)")
    
    if meaningful_improvement and any_target_achieved:
        print(f"\n🎉 SUCCESS! Enhanced feasibility checking shows significant benefits!")
        return True
    else:
        print(f"\n⚠️  Enhanced feasibility checking shows limited benefits in these tests")
        return False

if __name__ == "__main__":
    success = test_large_container_performance()
    exit(0 if success else 1)