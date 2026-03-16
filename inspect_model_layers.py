import torch
import os

# 加载模型检查点
checkpoint_path = './saved_models/test12_12_15/Bpp-v02025.12.15-20-24.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print('=== Mask Layers ===')
for key in checkpoint['model_state_dict']:
    if 'mask' in key:
        print(f'{key}: {checkpoint["model_state_dict"][key].shape}')

print('\n=== Dist Layers ===')
for key in checkpoint['model_state_dict']:
    if 'dist' in key:
        print(f'{key}: {checkpoint["model_state_dict"][key].shape}')

print('\n=== Action Space Analysis ===')
# 检查mask层的输出维度，这决定了动作空间大小
mask_output_shape = None
for key in checkpoint['model_state_dict']:
    if 'mask' in key and key.endswith('.weight'):
        mask_output_shape = checkpoint['model_state_dict'][key].shape
        print(f'Mask output layer shape: {mask_output_shape}')
        # 动作空间大小等于mask层输出的数量
        action_space_size = mask_output_shape[0]
        print(f'Estimated action space size: {action_space_size}')
        
        # 根据动作空间大小判断是否启用了旋转
        # 容器大小10x10的情况下，没有旋转时的动作空间应该是100
        # 启用旋转时应该是200
        if action_space_size == 100:
            print('Action space suggests rotation was DISABLED during training')
        elif action_space_size == 200:
            print('Action space suggests rotation was ENABLED during training')
        else:
            print('Action space size is unexpected - check container size configuration')
        break