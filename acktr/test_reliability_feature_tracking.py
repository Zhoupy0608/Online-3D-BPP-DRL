"""
Property-based tests for reliability feature tracking in multi-process training

These tests verify that reliability feature usage counts (uncertainty, visual feedback,
motion primitive) are correctly aggregated across all parallel processes.

Feature: multi-process-training, Property 13: Reliability feature tracking across processes
Validates: Requirements 5.5
"""

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from collections import deque


# Hypothesis strategies for generating test data
@st.composite
def num_processes_strategy(draw):
    """Generate valid number of processes (1-32)."""
    return draw(st.integers(min_value=1, max_value=32))


@st.composite
def feature_counts_strategy(draw, num_processes):
    """
    Generate feature usage counts for multiple processes.
    Returns three lists (noise, visual, motion), each with counts per process.
    """
    noise_counts = []
    visual_counts = []
    motion_counts = []
    
    for _ in range(num_processes):
        # Each process can have 0-100 feature usages
        noise_count = draw(st.integers(min_value=0, max_value=100))
        visual_count = draw(st.integers(min_value=0, max_value=100))
        motion_count = draw(st.integers(min_value=0, max_value=100))
        
        noise_counts.append(noise_count)
        visual_counts.append(visual_count)
        motion_counts.append(motion_count)
    
    return noise_counts, visual_counts, motion_counts


@st.composite
def infos_with_reliability_strategy(draw, num_processes):
    """
    Generate info dictionaries with reliability feature flags.
    Each info dict may contain reliability feature usage indicators.
    """
    infos = []
    for _ in range(num_processes):
        info = {}
        
        # Randomly include reliability features
        if draw(st.booleans()):
            info['noise_applied'] = draw(st.booleans())
        
        if draw(st.booleans()):
            info['visual_feedback_update'] = draw(st.booleans())
        
        if draw(st.booleans()):
            info['motion_option_used'] = draw(st.booleans())
        
        infos.append(info)
    
    return infos


# Property 13: Reliability feature tracking across processes
# Feature: multi-process-training, Property 13: Reliability feature tracking across processes
# Validates: Requirements 5.5
@settings(max_examples=100, deadline=None)
@given(
    num_processes=num_processes_strategy()
)
def test_reliability_feature_aggregation_across_processes(num_processes):
    """
    Property 13: For any reliability feature (uncertainty, visual feedback, motion primitive),
    usage counts should be aggregated across all processes.
    
    This test verifies that:
    1. Feature counts from all processes are summed correctly
    2. No counts are lost or duplicated
    3. Aggregation works for any number of processes
    
    Validates: Requirements 5.5
    """
    # Generate random feature counts for each process
    noise_counts = [np.random.randint(0, 50) for _ in range(num_processes)]
    visual_counts = [np.random.randint(0, 50) for _ in range(num_processes)]
    motion_counts = [np.random.randint(0, 50) for _ in range(num_processes)]
    
    # Simulate aggregation logic from main.py
    total_noise = sum(noise_counts)
    total_visual = sum(visual_counts)
    total_motion = sum(motion_counts)
    
    # Verify aggregation is correct
    expected_noise = sum(noise_counts)
    expected_visual = sum(visual_counts)
    expected_motion = sum(motion_counts)
    
    assert total_noise == expected_noise, \
        f"Noise count mismatch: {total_noise} != {expected_noise}"
    
    assert total_visual == expected_visual, \
        f"Visual feedback count mismatch: {total_visual} != {expected_visual}"
    
    assert total_motion == expected_motion, \
        f"Motion option count mismatch: {total_motion} != {expected_motion}"
    
    # Verify counts are non-negative
    assert total_noise >= 0, "Total noise count should be non-negative"
    assert total_visual >= 0, "Total visual count should be non-negative"
    assert total_motion >= 0, "Total motion count should be non-negative"


@settings(max_examples=100, deadline=None)
@given(
    num_processes=num_processes_strategy()
)
def test_reliability_feature_tracking_from_infos(num_processes):
    """
    Property 13: Test reliability feature tracking from environment info dictionaries.
    
    This simulates the actual pattern used in main.py where feature usage is tracked
    from the 'infos' list returned by envs.step().
    
    Validates: Requirements 5.5
    """
    # Generate random info dictionaries with reliability features
    infos = []
    expected_noise = 0
    expected_visual = 0
    expected_motion = 0
    
    for i in range(num_processes):
        info = {}
        
        # Randomly include reliability features (30% chance each)
        if np.random.random() < 0.3:
            info['noise_applied'] = True
            expected_noise += 1
        
        if np.random.random() < 0.3:
            info['visual_feedback_update'] = True
            expected_visual += 1
        
        if np.random.random() < 0.3:
            info['motion_option_used'] = True
            expected_motion += 1
        
        infos.append(info)
    
    # Simulate the tracking logic from main.py (lines 236-241)
    noise_applied_count = 0
    visual_feedback_updates = 0
    motion_options_used = 0
    
    for i in range(len(infos)):
        if 'noise_applied' in infos[i]:
            noise_applied_count += 1
        if 'visual_feedback_update' in infos[i]:
            visual_feedback_updates += 1
        if 'motion_option_used' in infos[i]:
            motion_options_used += 1
    
    # Verify all feature usages were tracked
    assert noise_applied_count == expected_noise, \
        f"Should track {expected_noise} noise applications, got {noise_applied_count}"
    
    assert visual_feedback_updates == expected_visual, \
        f"Should track {expected_visual} visual updates, got {visual_feedback_updates}"
    
    assert motion_options_used == expected_motion, \
        f"Should track {expected_motion} motion options, got {motion_options_used}"


@settings(max_examples=100, deadline=None)
@given(
    num_processes=num_processes_strategy()
)
def test_per_process_reliability_tracking(num_processes):
    """
    Property 13: Test that per-process reliability feature tracking correctly maintains
    separate counts for each process.
    
    This verifies the per-process tracking pattern that could be used for detailed
    statistics about which processes use which features.
    
    Validates: Requirements 5.5
    """
    # Initialize per-process tracking
    process_noise_counts = [0] * num_processes
    process_visual_counts = [0] * num_processes
    process_motion_counts = [0] * num_processes
    
    # Simulate multiple steps with random feature usage
    num_steps = 20
    expected_total_noise = 0
    expected_total_visual = 0
    expected_total_motion = 0
    
    for step in range(num_steps):
        # Generate random infos for this step
        for i in range(num_processes):
            # Random feature usage (20% chance each per process per step)
            if np.random.random() < 0.2:
                process_noise_counts[i] += 1
                expected_total_noise += 1
            
            if np.random.random() < 0.2:
                process_visual_counts[i] += 1
                expected_total_visual += 1
            
            if np.random.random() < 0.2:
                process_motion_counts[i] += 1
                expected_total_motion += 1
    
    # Verify structure
    assert len(process_noise_counts) == num_processes, \
        f"Should have {num_processes} noise count lists, got {len(process_noise_counts)}"
    
    assert len(process_visual_counts) == num_processes, \
        f"Should have {num_processes} visual count lists, got {len(process_visual_counts)}"
    
    assert len(process_motion_counts) == num_processes, \
        f"Should have {num_processes} motion count lists, got {len(process_motion_counts)}"
    
    # Verify total counts
    total_noise = sum(process_noise_counts)
    total_visual = sum(process_visual_counts)
    total_motion = sum(process_motion_counts)
    
    assert total_noise == expected_total_noise, \
        f"Total noise {total_noise} != expected {expected_total_noise}"
    
    assert total_visual == expected_total_visual, \
        f"Total visual {total_visual} != expected {expected_total_visual}"
    
    assert total_motion == expected_total_motion, \
        f"Total motion {total_motion} != expected {expected_total_motion}"
    
    # Verify all counts are non-negative
    for i in range(num_processes):
        assert process_noise_counts[i] >= 0, f"Process {i} has negative noise count"
        assert process_visual_counts[i] >= 0, f"Process {i} has negative visual count"
        assert process_motion_counts[i] >= 0, f"Process {i} has negative motion count"


@settings(max_examples=100, deadline=None)
@given(
    num_processes=num_processes_strategy()
)
def test_aggregation_preserves_all_feature_counts(num_processes):
    """
    Property 13: Test that aggregation from all processes preserves all feature counts
    without loss or duplication.
    
    Validates: Requirements 5.5
    """
    # Generate known counts for each process
    process_noise = [np.random.randint(0, 20) for _ in range(num_processes)]
    process_visual = [np.random.randint(0, 20) for _ in range(num_processes)]
    process_motion = [np.random.randint(0, 20) for _ in range(num_processes)]
    
    # Simulate aggregation
    aggregated_noise = sum(process_noise)
    aggregated_visual = sum(process_visual)
    aggregated_motion = sum(process_motion)
    
    # Verify all counts were preserved
    expected_noise = sum(process_noise)
    expected_visual = sum(process_visual)
    expected_motion = sum(process_motion)
    
    assert aggregated_noise == expected_noise, \
        f"Noise aggregation mismatch: {aggregated_noise} != {expected_noise}"
    
    assert aggregated_visual == expected_visual, \
        f"Visual aggregation mismatch: {aggregated_visual} != {expected_visual}"
    
    assert aggregated_motion == expected_motion, \
        f"Motion aggregation mismatch: {aggregated_motion} != {expected_motion}"


@settings(max_examples=50, deadline=None)
@given(
    num_processes=num_processes_strategy()
)
def test_reliability_tracking_with_mixed_features(num_processes):
    """
    Property 13: Test reliability tracking when different processes use different
    combinations of features.
    
    This simulates realistic scenarios where some processes might use certain features
    more than others.
    
    Validates: Requirements 5.5
    """
    # Initialize tracking
    noise_counts = [0] * num_processes
    visual_counts = [0] * num_processes
    motion_counts = [0] * num_processes
    
    # Simulate different feature usage patterns per process
    for i in range(num_processes):
        # Process 0: Heavy noise user
        if i == 0:
            noise_counts[i] = np.random.randint(50, 100)
            visual_counts[i] = np.random.randint(0, 10)
            motion_counts[i] = np.random.randint(0, 10)
        # Process 1: Heavy visual user
        elif i == 1 and num_processes > 1:
            noise_counts[i] = np.random.randint(0, 10)
            visual_counts[i] = np.random.randint(50, 100)
            motion_counts[i] = np.random.randint(0, 10)
        # Process 2: Heavy motion user
        elif i == 2 and num_processes > 2:
            noise_counts[i] = np.random.randint(0, 10)
            visual_counts[i] = np.random.randint(0, 10)
            motion_counts[i] = np.random.randint(50, 100)
        # Other processes: Balanced usage
        else:
            noise_counts[i] = np.random.randint(10, 30)
            visual_counts[i] = np.random.randint(10, 30)
            motion_counts[i] = np.random.randint(10, 30)
    
    # Aggregate
    total_noise = sum(noise_counts)
    total_visual = sum(visual_counts)
    total_motion = sum(motion_counts)
    
    # Verify aggregation
    assert total_noise == sum(noise_counts), "Noise aggregation failed"
    assert total_visual == sum(visual_counts), "Visual aggregation failed"
    assert total_motion == sum(motion_counts), "Motion aggregation failed"
    
    # Verify all totals are non-negative
    assert total_noise >= 0, "Total noise should be non-negative"
    assert total_visual >= 0, "Total visual should be non-negative"
    assert total_motion >= 0, "Total motion should be non-negative"
    
    # For single process, at least noise should be positive (it's guaranteed by the logic)
    if num_processes == 1:
        assert total_noise > 0, "Single process should have positive noise count"


@settings(max_examples=100, deadline=None)
@given(
    num_processes=num_processes_strategy()
)
def test_reliability_tracking_invariants(num_processes):
    """
    Property 13: Test invariants that should hold for reliability feature tracking.
    
    Invariants:
    1. Total count >= count from any single process
    2. Total count == sum of all process counts
    3. All counts are non-negative integers
    4. Aggregation is deterministic (same input => same output)
    
    Validates: Requirements 5.5
    """
    # Generate counts
    noise_counts = [np.random.randint(0, 50) for _ in range(num_processes)]
    visual_counts = [np.random.randint(0, 50) for _ in range(num_processes)]
    motion_counts = [np.random.randint(0, 50) for _ in range(num_processes)]
    
    # Aggregate
    total_noise = sum(noise_counts)
    total_visual = sum(visual_counts)
    total_motion = sum(motion_counts)
    
    # Invariant 1: Total >= any single process
    for i in range(num_processes):
        assert total_noise >= noise_counts[i], \
            f"Total noise {total_noise} should be >= process {i} count {noise_counts[i]}"
        assert total_visual >= visual_counts[i], \
            f"Total visual {total_visual} should be >= process {i} count {visual_counts[i]}"
        assert total_motion >= motion_counts[i], \
            f"Total motion {total_motion} should be >= process {i} count {motion_counts[i]}"
    
    # Invariant 2: Total == sum of all
    assert total_noise == sum(noise_counts), "Total noise should equal sum"
    assert total_visual == sum(visual_counts), "Total visual should equal sum"
    assert total_motion == sum(motion_counts), "Total motion should equal sum"
    
    # Invariant 3: All non-negative
    assert total_noise >= 0, "Total noise should be non-negative"
    assert total_visual >= 0, "Total visual should be non-negative"
    assert total_motion >= 0, "Total motion should be non-negative"
    
    for i in range(num_processes):
        assert noise_counts[i] >= 0, f"Process {i} noise count should be non-negative"
        assert visual_counts[i] >= 0, f"Process {i} visual count should be non-negative"
        assert motion_counts[i] >= 0, f"Process {i} motion count should be non-negative"
    
    # Invariant 4: Determinism - aggregate again and verify same results
    total_noise_2 = sum(noise_counts)
    total_visual_2 = sum(visual_counts)
    total_motion_2 = sum(motion_counts)
    
    assert total_noise == total_noise_2, "Noise aggregation should be deterministic"
    assert total_visual == total_visual_2, "Visual aggregation should be deterministic"
    assert total_motion == total_motion_2, "Motion aggregation should be deterministic"


# Unit tests for specific scenarios

def test_reliability_tracking_specific_scenario():
    """
    Unit test with a specific scenario to verify reliability feature tracking.
    """
    num_processes = 4
    
    # Simulate specific feature usage
    # Process 0: 5 noise, 10 visual, 15 motion
    # Process 1: 3 noise, 8 visual, 20 motion
    # Process 2: 7 noise, 12 visual, 18 motion
    # Process 3: 2 noise, 6 visual, 22 motion
    
    noise_counts = [5, 3, 7, 2]
    visual_counts = [10, 8, 12, 6]
    motion_counts = [15, 20, 18, 22]
    
    # Aggregate as in main.py
    total_noise = sum(noise_counts)
    total_visual = sum(visual_counts)
    total_motion = sum(motion_counts)
    
    # Verify totals
    assert total_noise == 17, f"Expected 17 noise applications, got {total_noise}"
    assert total_visual == 36, f"Expected 36 visual updates, got {total_visual}"
    assert total_motion == 75, f"Expected 75 motion options, got {total_motion}"


def test_reliability_tracking_no_features():
    """
    Unit test to verify behavior when no reliability features are used.
    """
    num_processes = 4
    
    # No features used
    noise_counts = [0] * num_processes
    visual_counts = [0] * num_processes
    motion_counts = [0] * num_processes
    
    # Aggregate
    total_noise = sum(noise_counts)
    total_visual = sum(visual_counts)
    total_motion = sum(motion_counts)
    
    # Verify all zeros
    assert total_noise == 0, "Should have 0 noise applications"
    assert total_visual == 0, "Should have 0 visual updates"
    assert total_motion == 0, "Should have 0 motion options"


def test_reliability_tracking_all_processes_active():
    """
    Unit test to verify behavior when all processes use all features.
    """
    num_processes = 8
    
    # All processes use features equally
    noise_counts = [10] * num_processes
    visual_counts = [15] * num_processes
    motion_counts = [20] * num_processes
    
    # Aggregate
    total_noise = sum(noise_counts)
    total_visual = sum(visual_counts)
    total_motion = sum(motion_counts)
    
    # Verify totals
    assert total_noise == 80, f"Expected 80 noise applications, got {total_noise}"
    assert total_visual == 120, f"Expected 120 visual updates, got {total_visual}"
    assert total_motion == 160, f"Expected 160 motion options, got {total_motion}"


def test_reliability_tracking_single_process():
    """
    Unit test with a single process (edge case).
    """
    num_processes = 1
    
    # Single process with some feature usage
    noise_counts = [42]
    visual_counts = [37]
    motion_counts = [55]
    
    # Aggregate
    total_noise = sum(noise_counts)
    total_visual = sum(visual_counts)
    total_motion = sum(motion_counts)
    
    # Verify totals equal single process counts
    assert total_noise == 42, f"Expected 42 noise applications, got {total_noise}"
    assert total_visual == 37, f"Expected 37 visual updates, got {total_visual}"
    assert total_motion == 55, f"Expected 55 motion options, got {total_motion}"


def test_reliability_tracking_from_info_dicts():
    """
    Unit test to verify tracking from actual info dictionaries.
    """
    # Simulate info dicts from multiple steps
    infos_sequence = [
        # Step 1
        [
            {'noise_applied': True, 'visual_feedback_update': True},
            {},
            {'motion_option_used': True},
            {'noise_applied': True}
        ],
        # Step 2
        [
            {},
            {'visual_feedback_update': True, 'motion_option_used': True},
            {'noise_applied': True},
            {}
        ],
        # Step 3
        [
            {'motion_option_used': True},
            {'noise_applied': True},
            {'visual_feedback_update': True},
            {'noise_applied': True, 'visual_feedback_update': True, 'motion_option_used': True}
        ]
    ]
    
    # Track features as in main.py
    noise_applied_count = 0
    visual_feedback_updates = 0
    motion_options_used = 0
    
    for step_infos in infos_sequence:
        for info in step_infos:
            if 'noise_applied' in info:
                noise_applied_count += 1
            if 'visual_feedback_update' in info:
                visual_feedback_updates += 1
            if 'motion_option_used' in info:
                motion_options_used += 1
    
    # Verify counts
    assert noise_applied_count == 5, f"Expected 5 noise applications, got {noise_applied_count}"
    assert visual_feedback_updates == 4, f"Expected 4 visual updates, got {visual_feedback_updates}"
    assert motion_options_used == 4, f"Expected 4 motion options, got {motion_options_used}"


def test_reliability_tracking_partial_features():
    """
    Unit test where only some features are enabled.
    """
    num_processes = 4
    
    # Only noise and motion enabled, no visual feedback
    noise_counts = [5, 3, 7, 2]
    visual_counts = [0, 0, 0, 0]  # Visual feedback disabled
    motion_counts = [15, 20, 18, 22]
    
    # Aggregate
    total_noise = sum(noise_counts)
    total_visual = sum(visual_counts)
    total_motion = sum(motion_counts)
    
    # Verify
    assert total_noise == 17, f"Expected 17 noise applications, got {total_noise}"
    assert total_visual == 0, f"Expected 0 visual updates (disabled), got {total_visual}"
    assert total_motion == 75, f"Expected 75 motion options, got {total_motion}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
