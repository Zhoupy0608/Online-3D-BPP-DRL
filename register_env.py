import gymnasium as gym
from gymnasium.envs.registration import register

# Register the Bpp-v0 environment in Gymnasium
try:
    gym.envs.registry.spec('Bpp-v0')
    print("Bpp-v0 environment already registered in Gymnasium")
except:
    register(
        id='Bpp-v0',
        entry_point='envs.bpp0:PackingGame',
    )
    print("Successfully registered Bpp-v0 environment in Gymnasium")

# Register the BppReliable-v0 environment in Gymnasium
try:
    gym.envs.registry.spec('BppReliable-v0')
    print("BppReliable-v0 environment already registered in Gymnasium")
except:
    register(
        id='BppReliable-v0',
        entry_point='envs.bpp0:ReliablePackingGame',
    )
    print("Successfully registered BppReliable-v0 environment in Gymnasium")
