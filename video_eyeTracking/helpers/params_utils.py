import os
import pickle

def save_params_to_pickle(output_path, params):
    """
    Save parameters to a pickle file in the specified output path.

    Parameters:
        output_path (str): Path to the output directory.
        params (dict): Dictionary of parameters to save.
    """
    params_pickle_path = os.path.join(output_path, 'params.pickle')
    with open(params_pickle_path, 'wb') as f:
        pickle.dump(params, f)

def load_params_from_pickle(params_pickle_path):
    """
    Load parameters from a pickle file.

    Parameters:
        params_pickle_path (str): Path to the pickle file.

    Returns:
        dict: Dictionary of loaded parameters.
    """
    with open(params_pickle_path, 'rb') as f:
        params = pickle.load(f)
    return params
