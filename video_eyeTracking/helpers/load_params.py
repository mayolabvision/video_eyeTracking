import os
import json

def load_params(output_path, new_params=None):
    """
    Load parameters from a JSON file or create a new one if it exists.

    :param output_path: The directory where the params.json file is located.
    :param new_params: A dictionary of parameters to save in the new file. Defaults to an empty dictionary.
    :return: The loaded or newly created parameters.
    """
    params_file = os.path.join(output_path, 'params.json')
    
    # Check if the file exists
    if os.path.exists(params_file):
        os.remove(params_file)  # Delete the existing file
    
    # Use provided new_params or default to an empty dictionary
    if new_params is None:
        new_params = {}
    
    # Save the new params to params.json
    with open(params_file, 'w') as f:
        json.dump(new_params, f, indent=4)
    
    return new_params
