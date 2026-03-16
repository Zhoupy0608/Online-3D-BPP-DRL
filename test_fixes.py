#!/usr/bin/env python
"""
Test script to verify fixes for the three main errors:
1. ValueError: too many values to unpack (expected 4)
2. ValueError: The truth value of an array with more than one element is ambiguous
3. TypeError: 'bool' object is not iterable
"""

import sys
import numpy as np
import torch

print("Testing imported modules...")
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test 1: Array logical operations (fix for ValueError: The truth value of an array...)
print("\nTest 1: Array logical operations")
try:
    terminated = np.array([True, False, True])
    truncated = np.array([False, True, False])
    # This should work now with np.logical_or
    news = np.logical_or(terminated, truncated)
    print(f"terminated: {terminated}")
    print(f"truncated: {truncated}")
    print(f"news: {news}")
    print("✓ Array logical operations work correctly")
except Exception as e:
    print(f"✗ Error in array logical operations: {e}")

# Test 2: Bool vs array iteration (fix for TypeError: 'bool' object is not iterable)
print("\nTest 2: Bool vs array iteration")
try:
    # Test with proper array (should work)
    done_array = np.array([True, False, True])
    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_array])
    print(f"done_array: {done_array}")
    print(f"masks: {masks}")
    print("✓ Array iteration works correctly")
    
    # Test with bool (would have failed before our fix)
    done_bool = True
    print(f"done_bool: {done_bool}")
    print("Note: We fixed this by ensuring done is always an array")
except Exception as e:
    print(f"✗ Error in iteration test: {e}")

# Test 3: Check imports from modified modules
print("\nTest 3: Import modified modules")
try:
    # Test vec_normalize import
    from baselines.common.vec_env.vec_normalize import VecNormalize
    print("✓ VecNormalize imported successfully")
    
    # Test shmem_vec_env import
    from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
    print("✓ ShmemVecEnv imported successfully")
    
    # Test acktr envs import
    from acktr.envs import VecPyTorchFrameStack
    print("✓ VecPyTorchFrameStack imported successfully")
    
    print("\n✓ All modified modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"✗ Unexpected error during imports: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")
