import torch
import os

model_path = 'D:/Online-3D-BPP-DRL/saved_models/test12_12_15/Bpp-v02025.12.15-20-24.pt'

# Check if the file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print('Checkpoint keys:', list(checkpoint.keys()))
        
        if 'reliability_config' in checkpoint:
            print('Reliability config:', checkpoint['reliability_config'])
            
        if 'training_step' in checkpoint:
            print('Training steps:', checkpoint['training_step'])
            
        if isinstance(checkpoint, list) and len(checkpoint) == 2:
            print('Legacy format model - contains model_state_dict and ob_rms')
            print('Model_state_dict keys:', list(checkpoint[0].keys())[:10], '...')
            
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print('Modern format model')
            print('Model_state_dict keys:', list(checkpoint['model_state_dict'].keys())[:10], '...')
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()