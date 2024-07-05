import os

def make_output_directory(video_path):
    """
    Creates an output directory in the same location as the video file.

    Parameters:
        video_path (str): Full path to the video file.

    Returns:
        str: Path to the created output directory.
    """
    # Extract the directory and the video file name without extension
    dir_name = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)
    video_name, _ = os.path.splitext(base_name)
    
    # Create the output directory name
    output_dir_name = f"out_{video_name}"
    output_path = os.path.join(dir_name, output_dir_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    return output_path

# Example usage:
# video_path = "/Users/blah/video_haha.avi"
# output_path = make_output_directory(video_path)
# print(f"Output directory: {output_path}")
