# video_eyeTracking/helpers/load_params.py
import os
import json

def load_params(output_path):
    params_file = os.path.join(output_path, 'params.json')
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
        return params
    return {}  # Return an empty dictionary if no params file exists
