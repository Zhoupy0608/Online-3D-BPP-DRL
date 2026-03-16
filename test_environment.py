import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gym
from gym.envs.registration import register

try:
    gym.envs.registry.spec('Bpp-v0')
except:
    register(id='Bpp-v0', entry_point='envs.bpp0:PackingGame')

# Test importing transforms3d modules
try:
    import transforms3d
    print("transforms3d imported successfully")
    
    import transforms3d.quaternions
    print("transforms3d.quaternions imported successfully")
    
    import transforms3d.axangles
    print("transforms3d.axangles imported successfully")
    
    import transforms3d.euler
    print("transforms3d.euler imported successfully")
    
    import transforms3d.taitbryan
    print("transforms3d.taitbryan imported successfully")
    
    import transforms3d.affines
    print("transforms3d.affines imported successfully")
    
    import transforms3d.utils
    print("transforms3d.utils imported successfully")
    
except Exception as e:
    print(f"Error importing transforms3d modules: {e}")
    import traceback
    traceback.print_exc()

# Test creating environment
try:
    env = gym.make('Bpp-v0', container_size=(10, 10, 10), test=True, data_name='cut_2.pt')
    print("Environment created successfully")
except Exception as e:
    print(f"Error creating environment: {e}")
    import traceback
    traceback.print_exc()