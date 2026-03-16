"""
Property-based tests for gradient NaN/Inf detection in KFAC optimizer

These tests verify that gradients and parameter updates remain finite (no NaN or Inf)
during multi-process training with the KFAC optimizer.
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


@st.composite
def learning_rate_strategy(draw):
    """Generate valid learning rates."""
    return draw(st.floats(min_value=0.01, max_value=1.0))


@st.composite
def damping_strategy(draw):
    """Generate valid damping factors."""
    return draw(st.floats(min_value=1e-4, max_value=1e-1))


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


# Property 8: No NaN or Inf in gradients
# Feature: multi-process-training, Property 8: No NaN or Inf in gradients
# Validates: Requirements 2.5
@settings(max_examples=100, deadline=None)
@given(
    batch_size=batch_size_strategy(),
    input_dim=input_dim_strategy(),
    hidden_dim=hidden_dim_strategy(),
    output_dim=output_dim_strategy(),
    learning_rate=learning_rate_strategy(),
    damping=damping_strategy()
)
def test_no_nan_inf_in_gradients_property(batch_size, input_dim, hidden_dim, 
                                           output_dim, learning_rate, damping):
    """
    Property 8: For any training step, all gradient tensors and parameter tensors 
    should not contain NaN or Inf values after the update.
    
    This property ensures numerical stability during multi-process training with KFAC,
    preventing gradient explosions or numerical errors that could corrupt training.
    
    Validates: Requirements 2.5
    """
    # Create model with random dimensions
    model = SimpleModel(input_dim, hidden_dim, output_dim)
    optimizer = KFACOptimizer(model, lr=learning_rate, Tf=1, damping=damping)
    
    # Create random input data
    x = torch.randn(batch_size, input_dim)
    target = torch.randint(0, output_dim, (batch_size,))
    
    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Property 1: Gradients should be finite before KFAC update
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), \
                "Gradients contain NaN or Inf before KFAC update"
    
    # Enable Fisher statistics accumulation
    optimizer.acc_stats = True
    
    # Do another forward-backward pass to accumulate Fisher statistics
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer.zero_grad()
    loss.backward()
    
    # Property 2: Gradients should be finite after backward pass
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), \
                "Gradients contain NaN or Inf after backward pass"
    
    # Perform optimizer step (this will compute eigenvalues at step 0 since Tf=1)
    optimizer.step()
    
    # Property 3: Parameters should be finite after KFAC update
    for param in model.parameters():
        assert torch.isfinite(param).all(), \
            f"Parameters contain NaN or Inf after KFAC update"
    
    # Property 4: Gradients should be finite after KFAC update
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), \
                "Gradients contain NaN or Inf after KFAC update"
    
    # Property 5: Fisher matrix eigenvalues should be finite
    for module in optimizer.modules:
        if module in optimizer.d_a:
            assert torch.isfinite(optimizer.d_a[module]).all(), \
                "d_a eigenvalues contain NaN or Inf"
            assert torch.isfinite(optimizer.d_g[module]).all(), \
                "d_g eigenvalues contain NaN or Inf"
    
    # Property 6: Fisher matrix eigenvectors should be finite
    for module in optimizer.modules:
        if module in optimizer.Q_a:
            assert torch.isfinite(optimizer.Q_a[module]).all(), \
                "Q_a eigenvectors contain NaN or Inf"
            assert torch.isfinite(optimizer.Q_g[module]).all(), \
                "Q_g eigenvectors contain NaN or Inf"
    
    # Property 7: Running statistics should be finite
    for module in optimizer.modules:
        if module in optimizer.m_aa:
            assert torch.isfinite(optimizer.m_aa[module]).all(), \
                "m_aa running statistics contain NaN or Inf"
            assert torch.isfinite(optimizer.m_gg[module]).all(), \
                "m_gg running statistics contain NaN or Inf"


# Unit test: Verify gradient checking with multiple training steps
def test_no_nan_inf_multiple_steps():
    """
    Unit test to verify gradients remain finite over multiple training steps.
    """
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    optimizer = KFACOptimizer(model, lr=0.25, Tf=10, damping=1e-2)
    
    # Train for 20 steps
    for step in range(20):
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
        
        # Check gradients before update
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"Gradients contain NaN or Inf at step {step} before update"
        
        # Perform optimizer step
        optimizer.step()
        
        # Check parameters after update
        for param in model.parameters():
            assert torch.isfinite(param).all(), \
                f"Parameters contain NaN or Inf at step {step} after update"
        
        # Check gradients after update
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"Gradients contain NaN or Inf at step {step} after update"


# Unit test: Verify gradient checking with large batch size
def test_no_nan_inf_large_batch():
    """
    Unit test to verify gradients remain finite with large batch sizes
    (simulating multi-process training with num_steps=5, num_processes=16).
    """
    model = SimpleModel(input_dim=15, hidden_dim=30, output_dim=8)
    optimizer = KFACOptimizer(model, lr=0.25, Tf=1, damping=1e-2)
    
    # Use batch size of 80 (5 steps × 16 processes)
    batch_size = 80
    x = torch.randn(batch_size, 15)
    target = torch.randint(0, 8, (batch_size,))
    
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
    
    # Verify all tensors are finite
    for param in model.parameters():
        assert torch.isfinite(param).all(), \
            "Parameters contain NaN or Inf with large batch size"
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), \
                "Gradients contain NaN or Inf with large batch size"


# Unit test: Verify gradient checking with extreme learning rates
def test_no_nan_inf_extreme_learning_rates():
    """
    Unit test to verify gradients remain finite even with extreme learning rates.
    """
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    
    # Test with very small learning rate
    optimizer_small = KFACOptimizer(model, lr=0.001, Tf=1, damping=1e-2)
    
    x = torch.randn(32, 10)
    target = torch.randint(0, 5, (32,))
    
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer_small.zero_grad()
    loss.backward()
    
    optimizer_small.acc_stats = True
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer_small.zero_grad()
    loss.backward()
    
    optimizer_small.step()
    
    # Verify finite with small learning rate
    for param in model.parameters():
        assert torch.isfinite(param).all(), \
            "Parameters contain NaN or Inf with small learning rate"
    
    # Reset model
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    
    # Test with large learning rate
    optimizer_large = KFACOptimizer(model, lr=1.0, Tf=1, damping=1e-2)
    
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer_large.zero_grad()
    loss.backward()
    
    optimizer_large.acc_stats = True
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer_large.zero_grad()
    loss.backward()
    
    optimizer_large.step()
    
    # Verify finite with large learning rate
    for param in model.parameters():
        assert torch.isfinite(param).all(), \
            "Parameters contain NaN or Inf with large learning rate"


# Unit test: Verify gradient checking with different damping values
def test_no_nan_inf_different_damping():
    """
    Unit test to verify gradients remain finite with different damping values.
    """
    model = SimpleModel(input_dim=12, hidden_dim=25, output_dim=6)
    
    # Test with small damping
    optimizer_small_damp = KFACOptimizer(model, lr=0.25, Tf=1, damping=1e-4)
    
    x = torch.randn(48, 12)
    target = torch.randint(0, 6, (48,))
    
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer_small_damp.zero_grad()
    loss.backward()
    
    optimizer_small_damp.acc_stats = True
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer_small_damp.zero_grad()
    loss.backward()
    
    optimizer_small_damp.step()
    
    # Verify finite with small damping
    for param in model.parameters():
        assert torch.isfinite(param).all(), \
            "Parameters contain NaN or Inf with small damping"
    
    # Reset model
    model = SimpleModel(input_dim=12, hidden_dim=25, output_dim=6)
    
    # Test with large damping
    optimizer_large_damp = KFACOptimizer(model, lr=0.25, Tf=1, damping=1e-1)
    
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer_large_damp.zero_grad()
    loss.backward()
    
    optimizer_large_damp.acc_stats = True
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    optimizer_large_damp.zero_grad()
    loss.backward()
    
    optimizer_large_damp.step()
    
    # Verify finite with large damping
    for param in model.parameters():
        assert torch.isfinite(param).all(), \
            "Parameters contain NaN or Inf with large damping"


# Unit test: Verify gradient checking after eigenvalue decomposition
def test_no_nan_inf_after_eigen_decomposition():
    """
    Unit test to verify gradients remain finite after eigenvalue decomposition
    is performed (at Tf intervals).
    """
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    optimizer = KFACOptimizer(model, lr=0.25, Tf=5, damping=1e-2)
    
    # Train for 10 steps to trigger eigenvalue decomposition at step 0 and 5
    for step in range(10):
        x = torch.randn(32, 10)
        target = torch.randint(0, 5, (32,))
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.acc_stats = True
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        # Check if eigenvalue decomposition was performed
        if step % 5 == 0:
            # Verify eigenvalues are finite
            for module in optimizer.modules:
                if module in optimizer.d_a:
                    assert torch.isfinite(optimizer.d_a[module]).all(), \
                        f"d_a eigenvalues contain NaN or Inf at step {step}"
                    assert torch.isfinite(optimizer.d_g[module]).all(), \
                        f"d_g eigenvalues contain NaN or Inf at step {step}"
        
        # Verify parameters are finite
        for param in model.parameters():
            assert torch.isfinite(param).all(), \
                f"Parameters contain NaN or Inf at step {step}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
