import sys
import os
import numpy as np

# Fix NumPy 2.0 compatibility issue in transforms3d
if not hasattr(np, 'maximum_sctype'):
    def maximum_sctype(dtype):
        return np.float64
    np.maximum_sctype = maximum_sctype

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_test import unified_test
from acktr.arguments import get_args
from gym.envs.registration import register

# 注册Bpp-v0环境
# Check if environment is already registered to avoid duplicate registration
try:
    import gym
    gym.envs.registry.spec('Bpp-v0')
except:
    # Register original PackingGame environment
    register(
        id='Bpp-v0',
        entry_point='envs.bpp0:PackingGame',
    )

# 创建自定义参数类
class Args:
    def __init__(self):
        self.env_name = 'Bpp-v0'
        self.container_size = (10, 10, 10)
        self.load_model = True
        self.load_name = 'Bpp-v02025.12.15-20-24.pt'
        self.load_dir = 'D:/Online-3D-BPP-DRL/saved_models/test12_12_15/'
        self.data_name = 'dataset/cut_2.pt'
        self.item_seq = 'cut2'
        self.device = 'cpu'
        self.cases = 10
        self.enable_rotation = False
        self.item_size_range = (2, 2, 2, 5, 5, 5)
        self.hidden_size = 256  # 添加hidden_size参数
        self.channel = 4  # 添加channel参数，3D装箱问题通常使用4个通道
        self.pallet_size = self.container_size[0]  # 添加pallet_size参数
        self.box_size_set = []
        for i in range(2, 6):
            for j in range(2, 6):
                for k in range(2, 6):
                    self.box_size_set.append((i, j, k))

def main():
    args = Args()
    model_path = args.load_dir + args.load_name
    print(f"Testing model: {model_path}")
    print(f"Dataset: {args.data_name}")
    unified_test(model_path, args)

if __name__ == "__main__":
    main()