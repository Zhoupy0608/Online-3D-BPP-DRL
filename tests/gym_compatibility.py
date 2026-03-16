# Gymnasium compatibility layer for existing code
import gymnasium as gym
import numpy as np

# Fix for deprecated numpy.maximum_sctype if needed
if not hasattr(np, 'maximum_sctype'):
    def maximum_sctype(dtype):
        return np.float64
    np.maximum_sctype = maximum_sctype

# Export gym as the default for compatibility
globals().update(gym.__dict__)
