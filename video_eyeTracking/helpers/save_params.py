# video_eyeTracking/helpers/save_params.py
import os
import json

def save_params(params, output_path):
    params_file = os.path.join(output_path, 'params.json')
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to {params_file}")
