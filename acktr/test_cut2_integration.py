"""
Comprehensive integration tests for improved feasibility mask with cut_2 dataset.

This module tests the complete system with cut_2 dataset scenarios to validate
75%+ utilization target achievement vs baseline 68% and system stability.
"""

import pytest
import numpy as np
import torch
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tempfile
import shutil

# Add the envs directory to the path to import Space and support_calculation
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'envs', 'bpp0'))

try:
    from space import Space, Box
    from support_calculation import SupportCalculator, StabilityThresholds, GeometricUtils, ThresholdManager
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)

# Skip gym environment import due to compatibility issues
# We'll test the Space class directly instead


class Cut2DatasetLoader:
    """Utility class to load and process cut_2 dataset for testing."""
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to cut_2.pt dataset file
        """
        if dataset_path is None:
            # Try to find the dataset in common locations
            possible_paths = [
                'dataset/cut_2.pt',
                '../dataset/cut_2.pt',
                '../../dataset/cut_2.pt'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_path = path
                    break
            
            if dataset_path is None:
                # Create a synthetic dataset for testing if real one not found
                self.use_synthetic = True
                self.dataset_path = None
            else:
                self.use_synthetic = False
                self.dataset_path = dataset_path
        else:
            self.use_synthetic = False
            self.dataset_path = dataset_path
    
    def load_dataset(self) -> List[List[Tuple[int, int, int]]]:
        """
        Load the cut_2 dataset.
        
        Returns:
            List of item sequences, where each sequence is a list of (x, y, z) tuples
        """
        if self.use_synthetic:
            return self._generate_synthetic_dataset()
        
        try:
            # Load the PyTorch dataset
            dataset = torch.load(self.dataset_path)
            
            # Convert to list of item sequences
            if isinstance(dataset, torch.Tensor):
                # Convert tensor to list of sequences
                sequences = []
                for seq in dataset:
                    items = []
                    for item in seq:
                        if len(item) >= 3:
                            items.append((int(item[0]), int(item[1]), int(item[2])))
                    if items:  # Only add non-empty sequences
                        sequences.append(items)
                return sequences
            elif isinstance(dataset, list):
                # Already in list format
                return dataset
            else:
                # Unknown format, fall back to synthetic
                return self._generate_synthetic_dataset()
                
        except Exception as e:
            print(f"Warning: Could not load cut_2 dataset from {self.dataset_path}: {e}")
            print("Using synthetic dataset for testing")
            return self._generate_synthetic_dataset()
    
    def _generate_synthetic_dataset(self) -> List[List[Tuple[int, int, int]]]:
        """
        Generate a synthetic dataset that mimics cut_2 characteristics.
        
        Returns:
            List of synthetic item sequences
        """
        np.random.seed(42)  # For reproducible tests
        
        sequences = []
        
        # Generate 20 test sequences
        for seq_idx in range(20):
            sequence = []
            num_items = np.random.randint(15, 35)  # 15-35 items per sequence
            
            for _ in range(num_items):
                # Generate items with sizes typical for cut_2 dataset
                # Items should fit in 10x10x10 container
                x = np.random.randint(2, 6)  # Width: 2-5
                y = np.random.randint(2, 6)  # Length: 2-5  
                z = np.random.randint(2, 6)  # Height: 2-5
                
                sequence.append((x, y, z))
            
            sequences.append(sequence)
        
        return sequences


class PerformanceTracker:
    """Track and analyze performance metrics during integration testing."""
    
    def __init__(self):
        self.results = []
        self.baseline_results = []
        self.enhanced_results = []
    
    def record_baseline_result(self, utilization: float, success_rate: float, 
                             total_items: int, placed_items: int, test_name: str):
        """Record results from baseline feasibility checking."""
        result = {
            'utilization': utilization,
            'success_rate': success_rate,
            'total_items': total_items,
            'placed_items': placed_items,
            'test_name': test_name,
            'timestamp': time.time()
        }
        self.baseline_results.append(result)
    
    def record_enhanced_result(self, utilization: float, success_rate: float,
                              total_items: int, placed_items: int, test_name: str,
                              threshold_adjustments: int = 0, fallback_used: bool = False):
        """Record results from enhanced feasibility checking."""
        result = {
            'utilization': utilization,
            'success_rate': success_rate,
            'total_items': total_items,
            'placed_items': placed_items,
            'test_name': test_name,
            'threshold_adjustments': threshold_adjustments,
            'fallback_used': fallback_used,
            'timestamp': time.time()
        }
        self.enhanced_results.append(result)
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """
        Compare performance between baseline and enhanced methods.
        
        Returns:
            Dictionary containing performance comparison metrics
        """
        if not self.baseline_results or not self.enhanced_results:
            return {'error': 'Insufficient data for comparison'}
        
        # Calculate averages for baseline
        baseline_utilization = np.mean([r['utilization'] for r in self.baseline_results])
        baseline_success_rate = np.mean([r['success_rate'] for r in self.baseline_results])
        
        # Calculate averages for enhanced
        enhanced_utilization = np.mean([r['utilization'] for r in self.enhanced_results])
        enhanced_success_rate = np.mean([r['success_rate'] for r in self.enhanced_results])
        
        # Calculate improvements
        utilization_improvement = enhanced_utilization - baseline_utilization
        success_rate_improvement = enhanced_success_rate - baseline_success_rate
        
        # Calculate percentage improvements
        utilization_improvement_pct = (utilization_improvement / baseline_utilization) * 100
        success_rate_improvement_pct = (success_rate_improvement / baseline_success_rate) * 100
        
        return {
            'baseline': {
                'avg_utilization': baseline_utilization,
                'avg_success_rate': baseline_success_rate,
                'num_tests': len(self.baseline_results)
            },
            'enhanced': {
                'avg_utilization': enhanced_utilization,
                'avg_success_rate': enhanced_success_rate,
                'num_tests': len(self.enhanced_results),
                'avg_threshold_adjustments': np.mean([r.get('threshold_adjustments', 0) for r in self.enhanced_results]),
                'fallback_usage_rate': np.mean([r.get('fallback_used', False) for r in self.enhanced_results])
            },
            'improvements': {
                'utilization_absolute': utilization_improvement,
                'utilization_percentage': utilization_improvement_pct,
                'success_rate_absolute': success_rate_improvement,
                'success_rate_percentage': success_rate_improvement_pct
            },
            'target_achievement': {
                'baseline_meets_target': baseline_utilization >= 0.75,
                'enhanced_meets_target': enhanced_utilization >= 0.75,
                'target_utilization': 0.75,
                'baseline_vs_target': baseline_utilization - 0.75,
                'enhanced_vs_target': enhanced_utilization - 0.75
            }
        }
    
    def save_results(self, filepath: str):
        """Save all results to a JSON file for analysis."""
        results_data = {
            'baseline_results': self.baseline_results,
            'enhanced_results': self.enhanced_results,
            'performance_comparison': self.get_performance_comparison(),
            'test_metadata': {
                'total_tests': len(self.baseline_results) + len(self.enhanced_results),
                'test_timestamp': time.time()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)


def run_packing_simulation(item_sequence: List[Tuple[int, int, int]], 
                          container_size: Tuple[int, int, int] = (10, 10, 10),
                          use_enhanced: bool = True,
                          enable_rotation: bool = False) -> Dict[str, Any]:
    """
    Run a packing simulation with a given item sequence.
    
    Args:
        item_sequence: List of (x, y, z) item dimensions
        container_size: Container dimensions (width, length, height)
        use_enhanced: Whether to use enhanced feasibility checking
        enable_rotation: Whether to allow item rotation
        
    Returns:
        Dictionary containing simulation results
    """
    width, length, height = container_size
    space = Space(width=width, length=length, height=height, 
                  use_enhanced_feasibility=use_enhanced)
    
    placed_items = 0
    failed_placements = 0
    placement_attempts = 0
    
    # Track performance metrics
    utilization_history = []
    threshold_adjustments_initial = space.performance_metrics['threshold_adjustments']
    
    for item_idx, (x, y, z) in enumerate(item_sequence):
        placed = False
        
        # Try all possible positions
        for lx in range(width):
            for ly in range(length):
                if lx + x <= width and ly + y <= length:
                    placement_attempts += 1
                    
                    # Try without rotation
                    if space.drop_box([x, y, z], space.position_to_index([lx, ly]), False):
                        placed_items += 1
                        placed = True
                        break
                    
                    # Try with rotation if enabled and item is not square
                    if enable_rotation and x != y:
                        if space.drop_box([x, y, z], space.position_to_index([lx, ly]), True):
                            placed_items += 1
                            placed = True
                            break
            
            if placed:
                break
        
        if not placed:
            failed_placements += 1
        
        # Record utilization after each item
        current_utilization = space.get_ratio()
        utilization_history.append(current_utilization)
        
        # Stop if container is getting full to avoid infinite loops
        if current_utilization > 0.95:
            break
    
    # Calculate final metrics
    final_utilization = space.get_ratio()
    success_rate = placed_items / len(item_sequence) if item_sequence else 0.0
    
    # Get performance metrics from space
    performance_metrics = space.collect_utilization_metrics()
    threshold_adjustments = space.performance_metrics['threshold_adjustments'] - threshold_adjustments_initial
    
    return {
        'final_utilization': final_utilization,
        'success_rate': success_rate,
        'placed_items': placed_items,
        'failed_placements': failed_placements,
        'total_items': len(item_sequence),
        'placement_attempts': placement_attempts,
        'utilization_history': utilization_history,
        'threshold_adjustments': threshold_adjustments,
        'fallback_used': space.fallback_active,
        'performance_metrics': performance_metrics,
        'space_summary': space.get_performance_summary()
    }


class TestCut2Integration:
    """Integration tests for improved feasibility mask with cut_2 dataset."""
    
    @pytest.fixture(scope="class")
    def dataset_loader(self):
        """Fixture to provide dataset loader."""
        return Cut2DatasetLoader()
    
    @pytest.fixture(scope="class")
    def performance_tracker(self):
        """Fixture to provide performance tracker."""
        return PerformanceTracker()
    
    @pytest.fixture(scope="class")
    def test_sequences(self, dataset_loader):
        """Fixture to provide test sequences from cut_2 dataset."""
        sequences = dataset_loader.load_dataset()
        # Use first 10 sequences for testing to keep test time reasonable
        return sequences[:10]
    
    def test_baseline_performance_measurement(self, test_sequences, performance_tracker):
        """
        Test baseline performance with original feasibility checking.
        
        This establishes the baseline performance (expected ~68% utilization)
        that the enhanced system should improve upon.
        """
        baseline_results = []
        
        for seq_idx, sequence in enumerate(test_sequences):
            result = run_packing_simulation(
                sequence, 
                container_size=(10, 10, 10),
                use_enhanced=False,  # Use baseline checking
                enable_rotation=False
            )
            
            baseline_results.append(result)
            
            # Record result in performance tracker
            performance_tracker.record_baseline_result(
                utilization=result['final_utilization'],
                success_rate=result['success_rate'],
                total_items=result['total_items'],
                placed_items=result['placed_items'],
                test_name=f"baseline_sequence_{seq_idx}"
            )
        
        # Verify baseline performance is around expected 68%
        avg_utilization = np.mean([r['final_utilization'] for r in baseline_results])
        avg_success_rate = np.mean([r['success_rate'] for r in baseline_results])
        
        # Baseline should be in reasonable range (50-80% to account for dataset variation)
        assert 0.50 <= avg_utilization <= 0.80, (
            f"Baseline utilization {avg_utilization:.3f} outside expected range [0.50, 0.80]"
        )
        
        assert 0.30 <= avg_success_rate <= 0.90, (
            f"Baseline success rate {avg_success_rate:.3f} outside expected range [0.30, 0.90]"
        )
        
        print(f"Baseline Performance - Utilization: {avg_utilization:.3f}, Success Rate: {avg_success_rate:.3f}")
    
    def test_enhanced_performance_improvement(self, test_sequences, performance_tracker):
        """
        Test enhanced performance with improved feasibility checking.
        
        This should achieve >75% utilization and demonstrate improvement
        over baseline performance.
        """
        enhanced_results = []
        
        for seq_idx, sequence in enumerate(test_sequences):
            result = run_packing_simulation(
                sequence,
                container_size=(10, 10, 10), 
                use_enhanced=True,  # Use enhanced checking
                enable_rotation=False
            )
            
            enhanced_results.append(result)
            
            # Record result in performance tracker
            performance_tracker.record_enhanced_result(
                utilization=result['final_utilization'],
                success_rate=result['success_rate'],
                total_items=result['total_items'],
                placed_items=result['placed_items'],
                test_name=f"enhanced_sequence_{seq_idx}",
                threshold_adjustments=result['threshold_adjustments'],
                fallback_used=result['fallback_used']
            )
        
        # Calculate enhanced performance metrics
        avg_utilization = np.mean([r['final_utilization'] for r in enhanced_results])
        avg_success_rate = np.mean([r['success_rate'] for r in enhanced_results])
        avg_threshold_adjustments = np.mean([r['threshold_adjustments'] for r in enhanced_results])
        
        # Enhanced system should achieve target utilization (>75%)
        assert avg_utilization >= 0.75, (
            f"Enhanced utilization {avg_utilization:.3f} did not meet target of 75%"
        )
        
        # Success rate should be reasonable
        assert avg_success_rate >= 0.40, (
            f"Enhanced success rate {avg_success_rate:.3f} is too low"
        )
        
        print(f"Enhanced Performance - Utilization: {avg_utilization:.3f}, "
              f"Success Rate: {avg_success_rate:.3f}, "
              f"Avg Threshold Adjustments: {avg_threshold_adjustments:.1f}")
    
    def test_performance_comparison_and_improvement(self, performance_tracker):
        """
        Compare enhanced performance against baseline and verify improvement.
        
        This validates Requirements 5.1 and 5.2:
        - Enhanced system achieves >75% utilization
        - Enhanced system demonstrates improvement over baseline
        """
        comparison = performance_tracker.get_performance_comparison()
        
        # Verify we have sufficient data
        assert comparison.get('error') is None, "Insufficient data for performance comparison"
        
        baseline = comparison['baseline']
        enhanced = comparison['enhanced']
        improvements = comparison['improvements']
        target_achievement = comparison['target_achievement']
        
        # Requirement 5.1: Enhanced system should achieve >75% utilization
        assert enhanced['avg_utilization'] >= 0.75, (
            f"Enhanced system utilization {enhanced['avg_utilization']:.3f} "
            f"did not meet 75% target"
        )
        
        # Requirement 5.2: Enhanced system should improve over baseline
        assert improvements['utilization_absolute'] > 0, (
            f"Enhanced system did not improve utilization over baseline. "
            f"Improvement: {improvements['utilization_absolute']:.3f}"
        )
        
        # Enhanced system should show meaningful improvement (at least 5% relative)
        assert improvements['utilization_percentage'] >= 5.0, (
            f"Enhanced system improvement {improvements['utilization_percentage']:.1f}% "
            f"is less than minimum expected 5%"
        )
        
        # Verify target achievement
        assert target_achievement['enhanced_meets_target'], (
            f"Enhanced system failed to meet 75% utilization target. "
            f"Achieved: {enhanced['avg_utilization']:.3f}"
        )
        
        print(f"Performance Comparison:")
        print(f"  Baseline: {baseline['avg_utilization']:.3f} utilization, {baseline['avg_success_rate']:.3f} success rate")
        print(f"  Enhanced: {enhanced['avg_utilization']:.3f} utilization, {enhanced['avg_success_rate']:.3f} success rate")
        print(f"  Improvement: {improvements['utilization_absolute']:.3f} absolute ({improvements['utilization_percentage']:.1f}%)")
        print(f"  Target Achievement: Enhanced={target_achievement['enhanced_meets_target']}, Baseline={target_achievement['baseline_meets_target']}")
    
    def test_system_stability_under_various_scenarios(self, dataset_loader):
        """
        Test system stability under various packing scenarios.
        
        This validates that the enhanced system remains stable and doesn't
        degrade performance under different conditions.
        """
        sequences = dataset_loader.load_dataset()
        
        # Test different container sizes
        container_sizes = [(8, 8, 8), (10, 10, 10), (12, 12, 12)]
        
        stability_results = []
        
        for container_size in container_sizes:
            for seq_idx, sequence in enumerate(sequences[:5]):  # Test 5 sequences per container size
                # Test with enhanced feasibility
                result = run_packing_simulation(
                    sequence,
                    container_size=container_size,
                    use_enhanced=True,
                    enable_rotation=False
                )
                
                stability_results.append({
                    'container_size': container_size,
                    'sequence_idx': seq_idx,
                    'utilization': result['final_utilization'],
                    'success_rate': result['success_rate'],
                    'fallback_used': result['fallback_used'],
                    'threshold_adjustments': result['threshold_adjustments']
                })
        
        # Verify system stability
        utilizations = [r['utilization'] for r in stability_results]
        success_rates = [r['success_rate'] for r in stability_results]
        
        # System should maintain reasonable performance across scenarios
        min_utilization = min(utilizations)
        max_utilization = max(utilizations)
        avg_utilization = np.mean(utilizations)
        
        assert min_utilization >= 0.40, (
            f"Minimum utilization {min_utilization:.3f} is too low, indicating instability"
        )
        
        assert avg_utilization >= 0.60, (
            f"Average utilization {avg_utilization:.3f} across scenarios is too low"
        )
        
        # Utilization variance should not be excessive
        utilization_std = np.std(utilizations)
        assert utilization_std <= 0.25, (
            f"Utilization standard deviation {utilization_std:.3f} is too high, "
            f"indicating unstable performance"
        )
        
        # Success rates should be reasonable
        avg_success_rate = np.mean(success_rates)
        assert avg_success_rate >= 0.30, (
            f"Average success rate {avg_success_rate:.3f} is too low"
        )
        
        print(f"Stability Test Results:")
        print(f"  Utilization - Min: {min_utilization:.3f}, Max: {max_utilization:.3f}, "
              f"Avg: {avg_utilization:.3f}, Std: {utilization_std:.3f}")
        print(f"  Success Rate - Avg: {avg_success_rate:.3f}")
        print(f"  Fallback Usage: {sum(r['fallback_used'] for r in stability_results)}/{len(stability_results)}")
    
    def test_adaptive_threshold_behavior(self, test_sequences):
        """
        Test that adaptive threshold adjustment works correctly during packing.
        
        This validates Requirements 4.1, 4.2, 4.4, and 5.4.
        """
        # Use a sequence that will likely trigger threshold adjustments
        test_sequence = test_sequences[0] if test_sequences else [(3, 3, 3)] * 20
        
        # Create space with enhanced feasibility
        space = Space(width=10, length=10, height=10, use_enhanced_feasibility=True)
        
        # Track threshold changes
        initial_thresholds = space.get_current_thresholds()
        threshold_history = [initial_thresholds]
        
        # Simulate packing with periodic monitoring
        placed_items = 0
        for item_idx, (x, y, z) in enumerate(test_sequence):
            # Try to place item
            placed = False
            for lx in range(10 - x + 1):
                for ly in range(10 - y + 1):
                    if space.drop_box([x, y, z], space.position_to_index([lx, ly]), False):
                        placed_items += 1
                        placed = True
                        break
                if placed:
                    break
            
            # Monitor and adjust every 5 items
            if (item_idx + 1) % 5 == 0:
                monitoring_result = space.monitor_and_adjust_performance()
                
                # Record threshold changes
                current_thresholds = space.get_current_thresholds()
                if current_thresholds != threshold_history[-1]:
                    threshold_history.append(current_thresholds)
        
        # Verify adaptive behavior
        final_thresholds = space.get_current_thresholds()
        performance_metrics = space.collect_utilization_metrics()
        
        # Should have some threshold adjustments if performance was suboptimal
        if performance_metrics['utilization_gap'] > 0.05:
            assert performance_metrics['threshold_adjustments'] > 0, (
                f"Expected threshold adjustments with utilization gap "
                f"{performance_metrics['utilization_gap']:.3f}, but got "
                f"{performance_metrics['threshold_adjustments']} adjustments"
            )
        
        # Verify threshold adjustment logging (Requirement 4.4)
        adjustment_history = space.threshold_manager.get_adjustment_history()
        if performance_metrics['threshold_adjustments'] > 0:
            assert len(adjustment_history) > 0, (
                "Threshold adjustments occurred but no history was logged"
            )
            
            # Verify history entries have required fields
            for entry in adjustment_history:
                required_fields = ['timestamp', 'old_thresholds', 'new_thresholds', 
                                 'utilization_gap', 'recent_success_rate']
                for field in required_fields:
                    assert field in entry, f"Missing required field '{field}' in adjustment history"
        
        print(f"Adaptive Threshold Test Results:")
        print(f"  Placed Items: {placed_items}/{len(test_sequence)}")
        print(f"  Final Utilization: {space.get_ratio():.3f}")
        print(f"  Threshold Adjustments: {performance_metrics['threshold_adjustments']}")
        print(f"  Fallback Active: {space.fallback_active}")
        print(f"  Threshold History Length: {len(threshold_history)}")
    
    def test_performance_degradation_detection_and_fallback(self, test_sequences):
        """
        Test performance degradation detection and fallback mechanism.
        
        This validates Requirement 5.5: Performance degradation fallback.
        """
        # Use a challenging sequence that might trigger degradation
        test_sequence = test_sequences[0] if test_sequences else [(4, 4, 4)] * 15
        
        # Create space with enhanced feasibility
        space = Space(width=8, length=8, height=8, use_enhanced_feasibility=True)
        
        # Artificially create conditions that might trigger degradation
        # by setting a high target utilization
        space.target_utilization = 0.90  # Very high target
        
        # Track degradation detection
        degradation_detected = False
        fallback_activated = False
        
        placed_items = 0
        for item_idx, (x, y, z) in enumerate(test_sequence):
            # Try to place item
            placed = False
            for lx in range(8 - x + 1):
                for ly in range(8 - y + 1):
                    if space.drop_box([x, y, z], space.position_to_index([lx, ly]), False):
                        placed_items += 1
                        placed = True
                        break
                if placed:
                    break
            
            # Monitor performance every few items
            if (item_idx + 1) % 3 == 0:
                monitoring_result = space.monitor_and_adjust_performance()
                
                if monitoring_result['degradation_detected']:
                    degradation_detected = True
                
                if monitoring_result['fallback_activated']:
                    fallback_activated = True
        
        # Verify degradation detection works
        final_utilization = space.get_ratio()
        performance_metrics = space.collect_utilization_metrics()
        
        # With challenging conditions, we should see some adaptive behavior
        if performance_metrics['utilization_gap'] > 0.15:  # Large gap
            # Should have detected degradation or activated fallback
            degradation_check = space.detect_performance_degradation()
            
            # Either degradation should be detected or fallback should be active
            assert degradation_check[0] or space.fallback_active, (
                f"Large utilization gap ({performance_metrics['utilization_gap']:.3f}) "
                f"but no degradation detected and no fallback active"
            )
        
        # If fallback was activated, verify it's working correctly
        if space.fallback_active:
            assert space.fallback_reason is not None, (
                "Fallback is active but no reason was recorded"
            )
            
            assert not space.use_enhanced_feasibility, (
                "Fallback is active but enhanced feasibility is still enabled"
            )
        
        print(f"Degradation Detection Test Results:")
        print(f"  Placed Items: {placed_items}/{len(test_sequence)}")
        print(f"  Final Utilization: {final_utilization:.3f}")
        print(f"  Target Utilization: {space.target_utilization:.3f}")
        print(f"  Utilization Gap: {performance_metrics['utilization_gap']:.3f}")
        print(f"  Degradation Detected: {degradation_detected}")
        print(f"  Fallback Active: {space.fallback_active}")
        print(f"  Fallback Reason: {space.fallback_reason}")
    
    def test_utilization_metrics_provision(self, test_sequences):
        """
        Test that utilization metrics are properly provided.
        
        This validates Requirement 5.3: Utilization metrics provision.
        """
        test_sequence = test_sequences[0] if test_sequences else [(2, 2, 2)] * 10
        
        # Create space and run simulation
        space = Space(width=10, length=10, height=10, use_enhanced_feasibility=True)
        
        # Place some items
        placed_items = 0
        for x, y, z in test_sequence[:5]:  # Place first 5 items
            for lx in range(10 - x + 1):
                for ly in range(10 - y + 1):
                    if space.drop_box([x, y, z], space.position_to_index([lx, ly]), False):
                        placed_items += 1
                        break
                else:
                    continue
                break
        
        # Collect utilization metrics
        metrics = space.collect_utilization_metrics()
        
        # Verify all required metrics are present
        required_metrics = [
            'current_utilization', 'target_utilization', 'baseline_utilization',
            'utilization_gap', 'recent_success_rate', 'total_placements',
            'placement_attempts', 'successful_placements', 'failed_placements',
            'threshold_adjustments', 'fallback_active', 'enhanced_feasibility_usage',
            'baseline_feasibility_usage'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Required metric '{metric}' missing from metrics output"
        
        # Verify metric values are reasonable
        assert 0.0 <= metrics['current_utilization'] <= 1.0, (
            f"Current utilization {metrics['current_utilization']} outside valid range [0, 1]"
        )
        
        assert 0.0 <= metrics['recent_success_rate'] <= 1.0, (
            f"Success rate {metrics['recent_success_rate']} outside valid range [0, 1]"
        )
        
        assert metrics['placement_attempts'] >= 0, (
            f"Placement attempts {metrics['placement_attempts']} should be non-negative"
        )
        
        assert metrics['successful_placements'] + metrics['failed_placements'] == metrics['placement_attempts'], (
            f"Placement counts don't add up: {metrics['successful_placements']} + "
            f"{metrics['failed_placements']} != {metrics['placement_attempts']}"
        )
        
        # Verify utilization gap calculation
        expected_gap = metrics['target_utilization'] - metrics['current_utilization']
        assert abs(metrics['utilization_gap'] - expected_gap) < 1e-10, (
            f"Utilization gap calculation incorrect: {metrics['utilization_gap']} != {expected_gap}"
        )
        
        # Get performance summary and verify it contains expected sections
        summary = space.get_performance_summary()
        
        required_sections = ['current_status', 'performance_metrics', 'recent_performance', 
                           'threshold_info', 'system_health']
        
        for section in required_sections:
            assert section in summary, f"Required section '{section}' missing from performance summary"
        
        print(f"Metrics Provision Test Results:")
        print(f"  Current Utilization: {metrics['current_utilization']:.3f}")
        print(f"  Success Rate: {metrics['recent_success_rate']:.3f}")
        print(f"  Placement Attempts: {metrics['placement_attempts']}")
        print(f"  Threshold Adjustments: {metrics['threshold_adjustments']}")
        print(f"  Fallback Active: {metrics['fallback_active']}")
    
    @pytest.mark.slow
    def test_comprehensive_cut2_dataset_evaluation(self, dataset_loader, performance_tracker):
        """
        Comprehensive evaluation with full cut_2 dataset.
        
        This is a comprehensive test that runs the complete dataset and
        generates detailed performance analysis.
        """
        sequences = dataset_loader.load_dataset()
        
        # Test with larger subset for comprehensive evaluation
        test_sequences = sequences[:20] if len(sequences) >= 20 else sequences
        
        comprehensive_results = {
            'baseline_results': [],
            'enhanced_results': [],
            'container_sizes': [(10, 10, 10), (12, 12, 12)],
            'rotation_settings': [False, True]
        }
        
        # Test multiple configurations
        for container_size in comprehensive_results['container_sizes']:
            for enable_rotation in comprehensive_results['rotation_settings']:
                for seq_idx, sequence in enumerate(test_sequences):
                    
                    # Test baseline
                    baseline_result = run_packing_simulation(
                        sequence,
                        container_size=container_size,
                        use_enhanced=False,
                        enable_rotation=enable_rotation
                    )
                    baseline_result['config'] = {
                        'container_size': container_size,
                        'rotation': enable_rotation,
                        'sequence_idx': seq_idx
                    }
                    comprehensive_results['baseline_results'].append(baseline_result)
                    
                    # Test enhanced
                    enhanced_result = run_packing_simulation(
                        sequence,
                        container_size=container_size,
                        use_enhanced=True,
                        enable_rotation=enable_rotation
                    )
                    enhanced_result['config'] = {
                        'container_size': container_size,
                        'rotation': enable_rotation,
                        'sequence_idx': seq_idx
                    }
                    comprehensive_results['enhanced_results'].append(enhanced_result)
        
        # Analyze comprehensive results
        baseline_utilizations = [r['final_utilization'] for r in comprehensive_results['baseline_results']]
        enhanced_utilizations = [r['final_utilization'] for r in comprehensive_results['enhanced_results']]
        
        baseline_avg = np.mean(baseline_utilizations)
        enhanced_avg = np.mean(enhanced_utilizations)
        
        # Verify comprehensive performance
        assert enhanced_avg >= 0.75, (
            f"Comprehensive enhanced average {enhanced_avg:.3f} did not meet 75% target"
        )
        
        assert enhanced_avg > baseline_avg, (
            f"Enhanced average {enhanced_avg:.3f} did not improve over baseline {baseline_avg:.3f}"
        )
        
        improvement = enhanced_avg - baseline_avg
        improvement_pct = (improvement / baseline_avg) * 100
        
        # Should see meaningful improvement across all configurations
        assert improvement_pct >= 5.0, (
            f"Comprehensive improvement {improvement_pct:.1f}% is less than minimum 5%"
        )
        
        # Calculate success rates
        baseline_success_rates = [r['success_rate'] for r in comprehensive_results['baseline_results']]
        enhanced_success_rates = [r['success_rate'] for r in comprehensive_results['enhanced_results']]
        
        baseline_success_avg = np.mean(baseline_success_rates)
        enhanced_success_avg = np.mean(enhanced_success_rates)
        
        print(f"Comprehensive Evaluation Results:")
        print(f"  Total Configurations Tested: {len(comprehensive_results['baseline_results'])}")
        print(f"  Baseline Performance:")
        print(f"    Avg Utilization: {baseline_avg:.3f} (std: {np.std(baseline_utilizations):.3f})")
        print(f"    Avg Success Rate: {baseline_success_avg:.3f} (std: {np.std(baseline_success_rates):.3f})")
        print(f"  Enhanced Performance:")
        print(f"    Avg Utilization: {enhanced_avg:.3f} (std: {np.std(enhanced_utilizations):.3f})")
        print(f"    Avg Success Rate: {enhanced_success_avg:.3f} (std: {np.std(enhanced_success_rates):.3f})")
        print(f"  Improvement:")
        print(f"    Utilization: +{improvement:.3f} ({improvement_pct:.1f}%)")
        print(f"    Success Rate: +{enhanced_success_avg - baseline_success_avg:.3f}")
        
        # Save comprehensive results for further analysis
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"comprehensive_cut2_results_{int(time.time())}.json"
        
        # Prepare results for JSON serialization
        json_results = {
            'baseline_results': comprehensive_results['baseline_results'],
            'enhanced_results': comprehensive_results['enhanced_results'],
            'summary': {
                'baseline_avg_utilization': baseline_avg,
                'enhanced_avg_utilization': enhanced_avg,
                'improvement_absolute': improvement,
                'improvement_percentage': improvement_pct,
                'baseline_avg_success_rate': baseline_success_avg,
                'enhanced_avg_success_rate': enhanced_success_avg,
                'total_configurations': len(comprehensive_results['baseline_results']),
                'target_achieved': enhanced_avg >= 0.75
            },
            'test_metadata': {
                'timestamp': time.time(),
                'dataset_sequences_tested': len(test_sequences),
                'container_sizes': comprehensive_results['container_sizes'],
                'rotation_settings': comprehensive_results['rotation_settings']
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"  Results saved to: {results_file}")


# Utility functions for running tests independently

def run_integration_test_suite():
    """
    Run the complete integration test suite and return results.
    
    This function can be called independently to run all integration tests
    and get a summary of results.
    """
    # Create temporary directory for test results
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "integration_results"
        results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        dataset_loader = Cut2DatasetLoader()
        performance_tracker = PerformanceTracker()
        
        # Load test sequences
        sequences = dataset_loader.load_dataset()[:10]  # Use first 10 for speed
        
        print("Running Cut_2 Dataset Integration Tests...")
        print("=" * 50)
        
        # Run baseline tests
        print("1. Testing baseline performance...")
        baseline_results = []
        for seq_idx, sequence in enumerate(sequences):
            result = run_packing_simulation(sequence, use_enhanced=False)
            baseline_results.append(result)
            performance_tracker.record_baseline_result(
                result['final_utilization'], result['success_rate'],
                result['total_items'], result['placed_items'],
                f"baseline_{seq_idx}"
            )
        
        baseline_avg = np.mean([r['final_utilization'] for r in baseline_results])
        print(f"   Baseline average utilization: {baseline_avg:.3f}")
        
        # Run enhanced tests
        print("2. Testing enhanced performance...")
        enhanced_results = []
        for seq_idx, sequence in enumerate(sequences):
            result = run_packing_simulation(sequence, use_enhanced=True)
            enhanced_results.append(result)
            performance_tracker.record_enhanced_result(
                result['final_utilization'], result['success_rate'],
                result['total_items'], result['placed_items'],
                f"enhanced_{seq_idx}", result['threshold_adjustments'],
                result['fallback_used']
            )
        
        enhanced_avg = np.mean([r['final_utilization'] for r in enhanced_results])
        print(f"   Enhanced average utilization: {enhanced_avg:.3f}")
        
        # Generate comparison
        print("3. Analyzing performance comparison...")
        comparison = performance_tracker.get_performance_comparison()
        
        # Save results
        results_file = results_dir / "integration_test_results.json"
        performance_tracker.save_results(str(results_file))
        
        # Print summary
        print("\nIntegration Test Results Summary:")
        print("=" * 50)
        print(f"Target Utilization: 75%")
        print(f"Baseline Performance: {comparison['baseline']['avg_utilization']:.3f}")
        print(f"Enhanced Performance: {comparison['enhanced']['avg_utilization']:.3f}")
        print(f"Improvement: {comparison['improvements']['utilization_absolute']:.3f} "
              f"({comparison['improvements']['utilization_percentage']:.1f}%)")
        print(f"Target Achieved: {comparison['target_achievement']['enhanced_meets_target']}")
        
        # Copy results to permanent location if possible
        try:
            permanent_results_dir = Path("integration_test_results")
            permanent_results_dir.mkdir(exist_ok=True)
            shutil.copy2(results_file, permanent_results_dir / f"results_{int(time.time())}.json")
            print(f"Results saved to: {permanent_results_dir}")
        except Exception as e:
            print(f"Could not save permanent results: {e}")
        
        return comparison


if __name__ == "__main__":
    # Run integration tests when script is executed directly
    results = run_integration_test_suite()
    
    # Exit with appropriate code
    if results['target_achievement']['enhanced_meets_target']:
        print("\n✅ All integration tests passed!")
        exit(0)
    else:
        print("\n❌ Integration tests failed to meet targets!")
        exit(1)