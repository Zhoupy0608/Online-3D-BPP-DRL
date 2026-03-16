import glob
import os
import torch.nn as nn
import numpy as np
from acktr.envs import VecNormalize
from acktr.performance_optimizer import get_profiler, CandidateMapCache

# Global cache for candidate maps
_candidate_map_cache = CandidateMapCache(max_size=1000)


def check_box(plain, x, y, lx, ly, z, container_size, buffer_range=(0, 0)):
    """
    Check if a box can be placed at the given position.
    
    Args:
        plain: Height map of the container
        x: Box length
        y: Box width
        lx: X position in container
        ly: Y position in container
        z: Box height
        container_size: Tuple (width, length, height) of container
        buffer_range: Tuple (buffer_x, buffer_y) for minimum clearance distance
    
    Returns:
        Maximum height at placement position if valid, -1 otherwise
    """
    if lx + x > container_size[0] or ly + y > container_size[1]:
        return -1
    if lx < 0 or ly < 0:
        return -1

    rec = plain[lx:lx + x, ly:ly + y]
    max_h = np.max(rec)
    max_area = np.sum(rec == max_h)
    area = x * y

    assert max_h >= 0
    if max_h + z > container_size[2]:
        return -1

    LU = int(rec[0, 0] == max_h)
    LD = int(rec[x - 1, 0] == max_h)
    RU = int(rec[0, y - 1] == max_h)
    RD = int(rec[x - 1, y - 1] == max_h)

    if max_area / area > 0.95:
        return max_h
    if LU + LD + RU + RD == 3 and max_area / area > 0.85:
        return max_h
    if LU + LD + RU + RD == 4 and max_area / area > 0.50:
        return max_h

    return -1


def check_buffer_space(plain, x, y, lx, ly, container_size, buffer_range):
    """
    Check if there is sufficient buffer space around a placement position.
    
    Args:
        plain: Height map of the container
        x: Box length
        y: Box width
        lx: X position in container
        ly: Y position in container
        container_size: Tuple (width, length, height) of container
        buffer_range: Tuple (buffer_x, buffer_y) for minimum clearance distance
    
    Returns:
        True if sufficient buffer space exists, False otherwise
    """
    buffer_x, buffer_y = buffer_range
    
    # If no buffer required, always return True
    if buffer_x == 0 and buffer_y == 0:
        return True
    
    # Check if we have space to check buffer (not at edges)
    # We need at least buffer_x space in x direction and buffer_y space in y direction
    
    # Get the height at the placement position
    placement_height = np.max(plain[lx:lx + x, ly:ly + y])
    
    # Check buffer space in negative x direction
    if lx >= buffer_x:
        buffer_region = plain[lx - buffer_x:lx, ly:ly + y]
        if np.any(buffer_region > placement_height):
            return False
    
    # Check buffer space in positive x direction
    if lx + x + buffer_x <= container_size[0]:
        buffer_region = plain[lx + x:lx + x + buffer_x, ly:ly + y]
        if np.any(buffer_region > placement_height):
            return False
    
    # Check buffer space in negative y direction
    if ly >= buffer_y:
        buffer_region = plain[lx:lx + x, ly - buffer_y:ly]
        if np.any(buffer_region > placement_height):
            return False
    
    # Check buffer space in positive y direction
    if ly + y + buffer_y <= container_size[1]:
        buffer_region = plain[lx:lx + x, ly + y:ly + y + buffer_y]
        if np.any(buffer_region > placement_height):
            return False
    
    return True

def get_possible_position(observation, container_size, buffer_range=(0, 0), use_cache=True):
    """
    Generate candidate map for possible placement positions.
    
    Args:
        observation: State observation containing height map and box dimensions
        container_size: Tuple (width, length, height) of container
        buffer_range: Tuple (buffer_x, buffer_y) for minimum clearance distance
        use_cache: Whether to use caching for candidate maps
    
    Returns:
        Flattened list of binary mask indicating valid positions
    """
    profiler = get_profiler()
    
    with profiler.profile("candidate_map_generation"):
        if not isinstance(observation, np.ndarray):
            box_info = observation.cpu().numpy()
        else:
            box_info = observation
        box_info = box_info.reshape((4,-1))
        x = int(box_info[1][0])
        y = int(box_info[2][0])
        z = int(box_info[3][0])

        plain = box_info[0].reshape((container_size[0],container_size[1]))
        
        # Try to get from cache
        if use_cache:
            cached = _candidate_map_cache.get(
                plain, (x, y, z), container_size, buffer_range, False
            )
            if cached is not None:
                return cached.reshape((-1,)).tolist()

        width = container_size[0]
        length = container_size[1]

        action_mask = np.zeros(shape=(width, length), dtype=np.int32)

        for i in range(width - x + 1):
            for j in range(length - y + 1):
                if check_box(plain, x, y, i, j, z, container_size, buffer_range) >= 0:
                    # Check buffer space if buffer_range is specified
                    if buffer_range[0] > 0 or buffer_range[1] > 0:
                        if check_buffer_space(plain, x, y, i, j, container_size, buffer_range):
                            action_mask[i, j] = 1
                    else:
                        action_mask[i, j] = 1

        # Store in cache
        if use_cache:
            _candidate_map_cache.put(
                plain, (x, y, z), container_size, buffer_range, False, action_mask
            )

        # Note: Removed fallback that marks all positions as valid when none are valid
        # This was causing incorrect behavior - if no valid positions exist, the mask should remain all zeros

        return action_mask.reshape((-1,)).tolist()

def get_rotation_mask(observation, container_size, buffer_range=(0, 0), use_cache=True):
    """
    Generate candidate maps for both original and rotated orientations.
    
    Args:
        observation: State observation containing height map and box dimensions
        container_size: Tuple (width, length, height) of container
        buffer_range: Tuple (buffer_x, buffer_y) for minimum clearance distance
        use_cache: Whether to use caching for candidate maps
    
    Returns:
        Concatenated binary masks for original and rotated orientations
    """
    profiler = get_profiler()
    
    with profiler.profile("rotation_mask_generation"):
        box_info = observation.cpu().numpy()
        box_info = box_info.reshape((4,-1))
        x = int(box_info[1][0])
        y = int(box_info[2][0])
        z = int(box_info[3][0])

        plain = box_info[0].reshape((container_size[0],container_size[1]))
        
        # Try to get from cache
        if use_cache:
            cached = _candidate_map_cache.get(
                plain, (x, y, z), container_size, buffer_range, True
            )
            if cached is not None:
                return cached

        width = container_size[0]
        length = container_size[1]

        action_mask1 = np.zeros(shape=(width, length), dtype=np.int32)
        action_mask2 = np.zeros(shape=(width, length), dtype=np.int32)

        # Original orientation
        for i in range(width - x + 1):
            for j in range(length - y + 1):
                if check_box(plain, x, y, i, j, z, container_size, buffer_range) >= 0:
                    # Check buffer space if buffer_range is specified
                    if buffer_range[0] > 0 or buffer_range[1] > 0:
                        if check_buffer_space(plain, x, y, i, j, container_size, buffer_range):
                            action_mask1[i, j] = 1
                    else:
                        action_mask1[i, j] = 1

        # Rotated orientation (90 degrees)
        for i in range(width - y + 1):
            for j in range(length - x + 1):
                if check_box(plain, y, x, i, j, z, container_size, buffer_range) >= 0:
                    # Check buffer space if buffer_range is specified
                    if buffer_range[0] > 0 or buffer_range[1] > 0:
                        if check_buffer_space(plain, y, x, i, j, container_size, buffer_range):
                            action_mask2[i, j] = 1
                    else:
                        action_mask2[i, j] = 1

        action_mask = np.hstack((action_mask1.reshape((-1,)), action_mask2.reshape((-1,))))
        
        # Store in cache
        if use_cache:
            _candidate_map_cache.put(
                plain, (x, y, z), container_size, buffer_range, True, action_mask
            )

        # Note: Removed fallback that marks all positions as valid when none are valid
        # This was causing incorrect behavior - if no valid positions exist, the mask should remain all zeros

        return action_mask

def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def get_candidate_map_cache():
    """Get the global candidate map cache instance."""
    return _candidate_map_cache


def clear_candidate_map_cache():
    """Clear the global candidate map cache."""
    _candidate_map_cache.clear()


def print_performance_stats():
    """Print performance profiling and cache statistics."""
    profiler = get_profiler()
    cache = _candidate_map_cache
    
    print("\n" + "="*80)
    profiler.print_summary()
    
    print("\n=== Candidate Map Cache Statistics ===")
    stats = cache.get_stats()
    print(f"Cache size: {stats['size']}/{stats['max_size']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print("="*80 + "\n")
