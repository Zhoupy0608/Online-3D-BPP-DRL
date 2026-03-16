"""
Property-based tests for device consistency in KFAC optimizer

These tests verify that device mismatches are automatically handled in the KFAC
optimizer during multi-process training, ensuring tensors are moved to the correct
device before matrix operations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from hypothesis import given, settings, strategies as st, assume

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


# Property 7: Device consistency in KFAC
# Feature: multi-process-training, Property 7: Device consistency in KFAC
# Validates: Requirements 2.4
@settings(max_examples=100, deadline=None)
@given(
    batch_size=batch_size_strategy(),
    input_dim=input_dim_strategy(),
    hidden_dim=hidden_dim_strategy(),
    output_dim=output_dim_strategy()
)
def test_device_consistency_property(batch_size, input_dim, hidden_dim, output_dim):
    """
    Property 7: For any KFAC update, if Q_a and Q_g are on different devices, 
    the system should automatically move Q_a and d_a to match Q_g's device 
    before matrix multiplication.
    
    This property ensures that device mismatches don't cause runtime errors
    during multi-process training where tensors may be on different devices.
    
    Validates: Requirements 2.4
    """
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping device consistency test")
    
    # Create model on GPU
    model = SimpleModel(input_dim, hidden_dim, output_dim).cuda()
    optimizer = KFACOptimizer(model, lr=0.1, Tf=1, damping=1e-2)
    
    # Create random input data on GPU
    x = torch.randn(batch_size, input_dim).cuda()
    target = torch.randint(0, output_dim, (batch_size,)).cuda()
    
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
    
    # Manually create device mismatch for testing
    # Move Q_a and d_a to CPU to simulate device mismatch
    device_mismatches_created = []
    for module in optimizer.modules:
        if module in optimizer.Q_a:
            original_device_a = optimizer.Q_a[module].device
            original_device_g = optimizer.Q_g[module].device
            
            # Move Q_a and d_a to CPU to create mismatch
            optimizer.Q_a[module] = optimizer.Q_a[module].cpu()
            optimizer.d_a[module] = optimizer.d_a[module].cpu()
            
            device_mismatches_created.append({
                'module': module,
                'original_device_a': original_device_a,
                'original_device_g': original_device_g,
                'mismatched_device_a': optimizer.Q_a[module].device,
                'mismatched_device_g': optimizer.Q_g[module].device
            })
    
    # Assume we created at least one device mismatch
    assume(len(device_mismatches_created) > 0)
    
    # Property 1: Optimizer step should not crash with device mismatch
    try:
        optimizer.step()
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            pytest.fail(f"Device mismatch not handled: {e}")
        else:
            # Re-raise if it's a different error
            raise
    
    # Property 2: After step, Q_a and Q_g should be on the same device
    for mismatch_info in device_mismatches_created:
        module = mismatch_info['module']
        
        if module in optimizer.Q_a and module in optimizer.Q_g:
            device_a = optimizer.Q_a[module].device
            device_g = optimizer.Q_g[module].device
            
            assert device_a == device_g, \
                f"After KFAC step, Q_a is on {device_a} but Q_g is on {device_g}. " \
                f"Device mismatch not resolved!"
    
    # Property 3: d_a and d_g should also be on the same device
    for mismatch_info in device_mismatches_created:
        module = mismatch_info['module']
        
        if module in optimizer.d_a and module in optimizer.d_g:
            device_d_a = optimizer.d_a[module].device
            device_d_g = optimizer.d_g[module].device
            
            assert device_d_a == device_d_g, \
                f"After KFAC step, d_a is on {device_d_a} but d_g is on {device_d_g}. " \
                f"Device mismatch not resolved!"
    
    # Property 4: Model parameters should still be on GPU
    for param in model.parameters():
        assert param.device.type == 'cuda', \
            f"Model parameter moved off GPU: {param.device}"
    
    # Property 5: Gradients should be finite after device mismatch handling
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), \
                "Gradients contain NaN or Inf after device mismatch handling"


# Unit test: Verify device consistency with CPU-only setup
def test_device_consistency_cpu_only():
    """
    Unit test to verify device consistency works correctly when all tensors
    are on CPU (no device mismatch possible).
    """
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    optimizer = KFACOptimizer(model, lr=0.1, Tf=1, damping=1e-2)
    
    # Create a batch of data on CPU
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
    
    # Perform optimizer step
    optimizer.step()
    
    # Verify all tensors remain on CPU
    for module in optimizer.modules:
        if module in optimizer.Q_a:
            assert optimizer.Q_a[module].device.type == 'cpu', \
                f"Q_a moved off CPU: {optimizer.Q_a[module].device}"
            assert optimizer.Q_g[module].device.type == 'cpu', \
                f"Q_g moved off CPU: {optimizer.Q_g[module].device}"
            assert optimizer.d_a[module].device.type == 'cpu', \
                f"d_a moved off CPU: {optimizer.d_a[module].device}"
            assert optimizer.d_g[module].device.type == 'cpu', \
                f"d_g moved off CPU: {optimizer.d_g[module].device}"


# Unit test: Verify device consistency with GPU setup
def test_device_consistency_gpu():
    """
    Unit test to verify device consistency when model is on GPU.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU device consistency test")
    
    model = SimpleModel(input_dim=15, hidden_dim=30, output_dim=8).cuda()
    optimizer = KFACOptimizer(model, lr=0.1, Tf=1, damping=1e-2)
    
    # Create a batch of data on GPU
    x = torch.randn(64, 15).cuda()
    target = torch.randint(0, 8, (64,)).cuda()
    
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
    
    # Manually create device mismatch
    for module in optimizer.modules:
        if module in optimizer.Q_a:
            # Move Q_a and d_a to CPU
            optimizer.Q_a[module] = optimizer.Q_a[module].cpu()
            optimizer.d_a[module] = optimizer.d_a[module].cpu()
            break
    
    # Perform optimizer step - should handle device mismatch
    optimizer.step()
    
    # Verify devices are now consistent
    for module in optimizer.modules:
        if module in optimizer.Q_a:
            assert optimizer.Q_a[module].device == optimizer.Q_g[module].device, \
                f"Devices still mismatched: Q_a on {optimizer.Q_a[module].device}, " \
                f"Q_g on {optimizer.Q_g[module].device}"
            assert optimizer.d_a[module].device == optimizer.d_g[module].device, \
                f"Devices still mismatched: d_a on {optimizer.d_a[module].device}, " \
                f"d_g on {optimizer.d_g[module].device}"


# Unit test: Verify device consistency with multiple modules
def test_device_consistency_multiple_modules():
    """
    Unit test to verify device consistency works correctly when there are
    multiple modules with device mismatches.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping multi-module device consistency test")
    
    model = SimpleModel(input_dim=20, hidden_dim=40, output_dim=10).cuda()
    optimizer = KFACOptimizer(model, lr=0.1, Tf=1, damping=1e-2)
    
    # Create a batch of data on GPU
    x = torch.randn(48, 20).cuda()
    target = torch.randint(0, 10, (48,)).cuda()
    
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
    
    # Create device mismatches in ALL modules
    mismatch_count = 0
    for module in optimizer.modules:
        if module in optimizer.Q_a:
            # Move Q_a and d_a to CPU
            optimizer.Q_a[module] = optimizer.Q_a[module].cpu()
            optimizer.d_a[module] = optimizer.d_a[module].cpu()
            mismatch_count += 1
    
    assert mismatch_count > 0, "No modules found to create device mismatch"
    
    # Perform optimizer step - should handle all device mismatches
    optimizer.step()
    
    # Verify all devices are now consistent
    for module in optimizer.modules:
        if module in optimizer.Q_a:
            assert optimizer.Q_a[module].device == optimizer.Q_g[module].device, \
                f"Module {module.__class__.__name__}: Devices still mismatched after step"
            assert optimizer.d_a[module].device == optimizer.d_g[module].device, \
                f"Module {module.__class__.__name__}: Eigenvalue devices still mismatched"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
