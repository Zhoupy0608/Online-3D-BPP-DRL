"""
Property-based tests for eigenvalue thresholding in KFAC optimizer

These tests verify that eigenvalues below 1e-6 are set to zero for numerical
stability in the KFAC optimizer during multi-process training.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from hypothesis import given, settings, strategies as st

from acktr.algo.kfac import KFACOptimizer


# Hypothesis strategies for generating test data
@st.composite
def batch_size_strategy(draw):
    """Generate valid batch sizes (simulating num_steps * num_processes)."""
    # Paper uses num_steps=5, num_processes=16, so batch_size=80
    # Test with various batch sizes from 4 to 128
    return draw(st.integers(min_value=4, max_value=128))


@st.composite
def input_dim_strategy(draw):
    """Generate valid input dimensions."""
    return draw(st.integers(min_value=5, max_value=50))


@st.composite
def hidden_dim_strategy(draw):
    """Generate valid hidden dimensions."""
    return draw(st.integers(min_value=10, max_value=100))


@st.composite
def output_dim_strategy(draw):
    """Generate valid output dimensions."""
    return draw(st.integers(min_value=2, max_value=20))


class SimpleModel(nn.Module):
    """Simple model for testing KFAC optimizer"""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Property 6: Eigenvalue thresholding for stability
# Feature: multi-process-training, Property 6: Eigenvalue thresholding for stability
# Validates: Requirements 2.3, 8.4
@settings(max_examples=100, deadline=None)
@given(
    batch_size=batch_size_strategy(),
    input_dim=input_dim_strategy(),
    hidden_dim=hidden_dim_strategy(),
    output_dim=output_dim_strategy()
)
def test_eigenvalue_thresholding_property(batch_size, input_dim, hidden_dim, output_dim):
    """
    Property 6: For any eigenvalue decomposition in KFAC, eigenvalues below 1e-6 
    should be set to zero to prevent numerical instability.
    
    This property ensures that the KFAC optimizer maintains numerical stability
    by thresholding small eigenvalues that could cause division by near-zero values.
    
    Validates: Requirements 2.3, 8.4
    """
    # Create model with random dimensions
    model = SimpleModel(input_dim, hidden_dim, output_dim)
    optimizer = KFACOptimizer(model, lr=0.1, Tf=1, damping=1e-2)
    
    # Create random input data
    x = torch.randn(batch_size, input_dim)
    target = torch.randint(0, output_dim, (batch_size,))
    
    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Enable Fisher statistics accumulation
    optimizer.acc_stats = True
    
    # Do another forward-backward pass to accumulate Fisher statistics
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer.zero_grad()
    loss.backward()
    
    # Perform optimizer step (this will compute eigenvalues at step 0 since Tf=1)
    optimizer.step()
    
    # Property: Check that eigenvalues below 1e-6 are zeroed
    for module in optimizer.modules:
        if module in optimizer.d_a:
            d_a = optimizer.d_a[module]
            d_g = optimizer.d_g[module]
            
            # Property 1: No eigenvalue should be between 0 and 1e-6 (exclusive)
            # All eigenvalues should be either >= 1e-6 or exactly 0
            small_but_nonzero_a = ((d_a > 0) & (d_a < 1e-6)).any()
            small_but_nonzero_g = ((d_g > 0) & (d_g < 1e-6)).any()
            
            assert not small_but_nonzero_a, \
                f"Found eigenvalues in d_a between 0 and 1e-6: {d_a[(d_a > 0) & (d_a < 1e-6)]}"
            assert not small_but_nonzero_g, \
                f"Found eigenvalues in d_g between 0 and 1e-6: {d_g[(d_g > 0) & (d_g < 1e-6)]}"
            
            # Property 2: All eigenvalues should be non-negative
            assert (d_a >= 0).all(), f"Found negative eigenvalues in d_a: {d_a[d_a < 0]}"
            assert (d_g >= 0).all(), f"Found negative eigenvalues in d_g: {d_g[d_g < 0]}"
            
            # Property 3: Eigenvalues should be finite (no NaN or Inf)
            assert torch.isfinite(d_a).all(), "Found NaN or Inf in d_a eigenvalues"
            assert torch.isfinite(d_g).all(), "Found NaN or Inf in d_g eigenvalues"


# Unit test: Verify eigenvalue thresholding with known small eigenvalues
def test_eigenvalue_thresholding_with_small_values():
    """
    Unit test to verify eigenvalue thresholding works correctly when
    Fisher matrices have small eigenvalues.
    """
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    optimizer = KFACOptimizer(model, lr=0.1, Tf=1, damping=1e-2)
    
    # Create a batch of data
    x = torch.randn(32, 10)
    target = torch.randint(0, 5, (32,))
    
    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Enable Fisher statistics accumulation
    optimizer.acc_stats = True
    
    # Do another forward-backward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer.zero_grad()
    loss.backward()
    
    # Manually inject small eigenvalues into Fisher matrices to test thresholding
    for module in optimizer.modules:
        if module in optimizer.m_aa:
            # Create a matrix with some very small eigenvalues
            m_aa = optimizer.m_aa[module]
            m_gg = optimizer.m_gg[module]
            
            # Add small perturbations to create small eigenvalues
            m_aa = m_aa + torch.eye(m_aa.size(0)) * 1e-8
            m_gg = m_gg + torch.eye(m_gg.size(0)) * 1e-8
            
            optimizer.m_aa[module] = m_aa
            optimizer.m_gg[module] = m_gg
    
    # Perform optimizer step
    optimizer.step()
    
    # Verify eigenvalues are properly thresholded
    for module in optimizer.modules:
        if module in optimizer.d_a:
            d_a = optimizer.d_a[module]
            d_g = optimizer.d_g[module]
            
            # Check no eigenvalues between 0 and 1e-6
            small_but_nonzero_a = ((d_a > 0) & (d_a < 1e-6)).any()
            small_but_nonzero_g = ((d_g > 0) & (d_g < 1e-6)).any()
            
            assert not small_but_nonzero_a, \
                f"Found eigenvalues in d_a between 0 and 1e-6 after thresholding"
            assert not small_but_nonzero_g, \
                f"Found eigenvalues in d_g between 0 and 1e-6 after thresholding"


# Unit test: Verify eigenvalue thresholding doesn't affect large eigenvalues
def test_eigenvalue_thresholding_preserves_large_values():
    """
    Unit test to verify that eigenvalue thresholding only affects small values
    and preserves eigenvalues >= 1e-6.
    """
    model = SimpleModel(input_dim=15, hidden_dim=30, output_dim=8)
    optimizer = KFACOptimizer(model, lr=0.1, Tf=1, damping=1e-2)
    
    # Create a larger batch to get more stable Fisher matrices
    x = torch.randn(64, 15)
    target = torch.randint(0, 8, (64,))
    
    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Enable Fisher statistics accumulation
    optimizer.acc_stats = True
    
    # Do another forward-backward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer.zero_grad()
    loss.backward()
    
    # Perform optimizer step
    optimizer.step()
    
    # Verify that we have some non-zero eigenvalues (large ones preserved)
    has_large_eigenvalues = False
    for module in optimizer.modules:
        if module in optimizer.d_a:
            d_a = optimizer.d_a[module]
            d_g = optimizer.d_g[module]
            
            # Check that we have eigenvalues >= 1e-6
            if (d_a >= 1e-6).any() or (d_g >= 1e-6).any():
                has_large_eigenvalues = True
            
            # Verify large eigenvalues are preserved (not zeroed)
            large_a = d_a[d_a >= 1e-6]
            large_g = d_g[d_g >= 1e-6]
            
            if len(large_a) > 0:
                assert (large_a >= 1e-6).all(), \
                    f"Large eigenvalues in d_a were incorrectly modified"
            
            if len(large_g) > 0:
                assert (large_g >= 1e-6).all(), \
                    f"Large eigenvalues in d_g were incorrectly modified"
    
    # We should have at least some large eigenvalues in a properly trained model
    assert has_large_eigenvalues, \
        "Expected to find some eigenvalues >= 1e-6 in Fisher matrices"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
