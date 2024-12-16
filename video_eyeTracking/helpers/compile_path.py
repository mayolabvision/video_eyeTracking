import os

def compile_path(data_path, patient_ID, seizure_ID):
    """
    Creates an output directory in the same location as the video file.

    Parameters:
        video_path (str): Full path to the video file.

    Returns:
        str: Path to the created output directory.
    """
    # Extract the directory and the video file name without extension
    patient_path = os.path.join(data_path, patient_ID, f"SZ{seizure_ID}")
    sub_dir = next(d for d in os.listdir(patient_path) if d != ".DS_Store" and os.path.isdir(os.path.join(patient_path, d)))
    full_path = os.path.join(patient_path, sub_dir)

    out_path = os.path.join(full_path, "tracking")
    if os.path.exists(full_path):
        os.makedirs(out_path, exist_ok=True) 
            
    return full_path
