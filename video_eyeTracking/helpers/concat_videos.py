import os
import glob
import re
import cv2 as cv

def concat_videos(full_path, clip_details):
    """ 
    Finds, orders, and concatenates video files that match the specified indices.

    Parameters:
        full_path (str): Path where video files are located.
        clip_details (list): List containing details, with video indices at the 9th position.

    Returns:
        str: Path to the concatenated video file.
    """

    # Get all .avi files within the directory and subdirectories
    video_files = glob.glob(os.path.join(full_path, '**', '*.avi'), recursive=True)
    video_indices = clip_details[8]

    # Dictionary to store matching video filenames by index
    video_dict = {}

    # Loop through each video file and check if its index matches any in video_indices
    for video_file in video_files:
        match = re.search(r'\((\d+)\)', video_file)
        if match:
            video_number = int(match.group(1))
            if video_number in video_indices:
                video_dict[video_number] = video_file

    # Order the selected files based on video_indices
    selected_files = [video_dict[idx] for idx in video_indices if idx in video_dict]

    # Directory and file path for the concatenated video
    output_path = os.path.join(full_path, "tracking", "close_clips.avi")

    if not os.path.exists(output_path):
        # Read the first video to get the frame size and frame rate
        first_video = cv.VideoCapture(selected_files[0])
        frame_width = int(first_video.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(first_video.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(first_video.get(cv.CAP_PROP_FPS))
        first_video.release()

        # Initialize the video writer with the same properties
        fourcc = cv.VideoWriter_fourcc(*'XVID')  # Choose 'XVID' for .avi format
        out = cv.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

        # Concatenate each video in selected_files
        for video_file in selected_files:
            cap = cv.VideoCapture(video_file)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Write each frame to the output video
                out.write(frame)
            
            cap.release()

        # Release the video writer
        out.release()
