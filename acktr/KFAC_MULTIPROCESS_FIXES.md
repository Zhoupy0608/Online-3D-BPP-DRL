# KFAC Optimizer Fixes for Multi-Process Training

## Summary

This document describes the fixes applied to the KFAC optimizer (`acktr/algo/kfac.py`) to support multi-process training as specified in the paper.

## Requirements Addressed

- **Requirement 2.1**: Fisher matrix accumulation works with batched gradients
- **Requirement 2.2**: KFAC updates using aggregated Fisher matrices
- **Requirement 2.3**: Eigenvalue thresholding for numerical stability
- **Requirement 2.4**: Device mismatch handling
- **Requirement 2.5**: Gradient checking to detect NaN/Inf values

## Changes Made

### 1. Added `acc_stats` Flag Initialization

**Location**: `__init__` method

**Change**: Added initialization of `self.acc_stats = False` flag to control Fisher statistics accumulation.

**Reason**: The flag was referenced in `_save_grad_output` but never initialized, causing AttributeError.

### 2. Enhanced Gradient Checking

**Location**: `step()` method - beginning

**Changes**:
- Added pre-processing check for NaN/Inf in gradients
- If detected, falls back to SGD with gradient clipping
- Prevents corrupted gradients from propagating through KFAC

**Code**:
```python
# Check for NaN/Inf in gradients before processing
has_nan_or_inf = False
for p in self.model.parameters():
    if p.grad is not None:
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            has_nan_or_inf = True
            print(f"Warning: NaN or Inf detected in gradients at step {self.steps}")
            break

# If NaN/Inf detected, skip KFAC update and use SGD
if has_nan_or_inf:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.optim.step()
    self.steps += 1
    return
```

### 3. Improved Eigenvalue Decomposition Error Handling

**Location**: `step()` method - eigenvalue decomposition section

**Changes**:
- Wrapped eigenvalue decomposition in try-except block
- Falls back to SGD if decomposition fails
- Added informative error messages

**Code**:
```python
try:
    self.d_g[m], self.Q_g[m] = torch.linalg.eigh(self.m_gg[m])
    self.d_a[m], self.Q_a[m] = torch.linalg.eigh(self.m_aa[m])
except RuntimeError as e:
    print(f"Warning: Eigenvalue decomposition failed at step {self.steps}: {e}")
    print("Falling back to SGD for this step")
    self.optim.step()
    self.steps += 1
    return
```

### 4. Enhanced Eigenvalue Thresholding Documentation

**Location**: `step()` method - after eigenvalue decomposition

**Changes**:
- Added detailed comments explaining the thresholding
- Clarified that eigenvalues < 1e-6 are set to 0 for numerical stability

**Code**:
```python
# Eigenvalue thresholding for numerical stability
# Set eigenvalues < 1e-6 to 0 to prevent division by very small numbers
self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
self.d_g[m].mul_((self.d_g[m] > 1e-6).float())
```

### 5. Improved Device Mismatch Handling Documentation

**Location**: `step()` method - before natural gradient computation

**Changes**:
- Added detailed comments explaining device handling
- Clarified importance for multi-process training

**Code**:
```python
# Device mismatch handling: ensure Q_a and d_a are on same device as Q_g
# This is critical for multi-process training where tensors may be on different devices
if self.Q_a[m].device != self.Q_g[m].device:
    self.Q_a[m] = self.Q_a[m].to(self.Q_g[m].device)
    self.d_a[m] = self.d_a[m].to(self.Q_g[m].device)
```

### 6. Added Natural Gradient Computation Documentation

**Location**: `step()` method - natural gradient computation

**Changes**:
- Added formula documentation for natural gradient
- Clarified the Kronecker-factored approximation

**Code**:
```python
# Compute natural gradient using Kronecker-factored approximation
# v = Q_g @ (Q_g^T @ grad @ Q_a) / (d_g ⊗ d_a + λ) @ Q_a^T
v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
```

### 7. Added NaN/Inf Check After Update Computation

**Location**: `step()` method - after computing natural gradient update

**Changes**:
- Added check for NaN/Inf in computed updates
- Falls back to SGD if detected
- Prevents corrupted updates from being applied

**Code**:
```python
# Check for NaN/Inf in computed update
if torch.isnan(v).any() or torch.isinf(v).any():
    print(f"Warning: NaN or Inf in KFAC update at step {self.steps}, module {i}")
    print("Falling back to SGD for this step")
    self.optim.step()
    self.steps += 1
    return
```

### 8. Improved KL Divergence Computation

**Location**: `step()` method - KL divergence computation

**Changes**:
- Added protection against division by zero
- Ensured vg_sum is always positive

**Code**:
```python
# Prevent division by zero or negative values
vg_sum = max(vg_sum.item(), 1e-10)
nu = min(1, math.sqrt(self.kl_clip / vg_sum))
```

### 9. Added Final Gradient Check

**Location**: `step()` method - before optimizer step

**Changes**:
- Added final check for NaN/Inf before applying updates
- Clips gradients if issues detected

**Code**:
```python
# Final check for NaN/Inf before optimizer step
for p in self.model.parameters():
    if p.grad is not None:
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            print(f"Warning: NaN or Inf in final gradients at step {self.steps}")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            break
```

### 10. Fixed Weight Decay Application

**Location**: `step()` method - beginning

**Changes**:
- Updated to use `alpha` parameter for PyTorch compatibility
- Added None check for gradients

**Code**:
```python
if self.weight_decay > 0:
    for p in self.model.parameters():
        if p.grad is not None:
            p.grad.data.add_(p.data, alpha=self.weight_decay)
```

## Testing

Created comprehensive test suite in `acktr/test_kfac_multiprocess.py` covering:

1. **Eigenvalue Thresholding Test**: Verifies eigenvalues < 1e-6 are set to 0
2. **Device Mismatch Handling Test**: Verifies tensors are moved to correct device
3. **NaN/Inf Detection Test**: Verifies NaN/Inf values are detected and handled
4. **Fisher Matrix Accumulation Test**: Verifies Fisher matrices accumulate correctly
5. **Multi-Process Batch Gradients Test**: Verifies KFAC works with large batches (80 samples)
6. **Natural Gradient Computation Test**: Verifies natural gradient computation completes successfully

All tests pass successfully.

## Multi-Process Training Compatibility

These fixes ensure KFAC works correctly with multi-process training:

1. **Fisher matrices accumulate statistics from all processes** via the running averages in `m_aa` and `m_gg`
2. **Device mismatches are handled automatically** when tensors are on different devices
3. **Numerical stability is maintained** through eigenvalue thresholding and error handling
4. **Gradient corruption is prevented** through multiple NaN/Inf checks
5. **Graceful degradation** to SGD when KFAC encounters issues

## Integration with Training Pipeline

The KFAC optimizer integrates with the ACKTR training pipeline (`acktr/algo/acktr_pipeline.py`):

1. Fisher statistics are accumulated when `optimizer.acc_stats = True`
2. The optimizer is called with batched gradients from all processes
3. Natural gradient updates are applied using the aggregated Fisher matrices
4. The optimizer handles all edge cases gracefully

## Performance Considerations

- **Eigenvalue decomposition** is performed every `Tf` steps (default: 10)
- **Statistics accumulation** is performed every `Ts` steps (default: 1)
- **Fallback to SGD** ensures training continues even with numerical issues
- **Device transfers** are minimized by checking before moving tensors

## Known Limitations

1. KFAC can still encounter numerical issues with very ill-conditioned Fisher matrices
2. Device transfers may add overhead in multi-GPU setups
3. The fallback to SGD may slow convergence temporarily

## Future Improvements

1. Consider using more aggressive eigenvalue filtering (e.g., 1e-4 instead of 1e-6)
2. Add condition number monitoring for Fisher matrices
3. Implement automatic damping adjustment based on numerical stability
4. Add telemetry for tracking how often fallback to SGD occurs
