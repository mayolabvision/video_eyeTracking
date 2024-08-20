import cv2 as cv
import os

def merge_videos_side_by_side(output_path):
    """
    Merges two videos side-by-side into one video, with the second video (video3) being twice as wide.
    The shorter video is scaled up to match the height of the taller video, preserving aspect ratios.

    Parameters:
    - output_path: Path where the videos are stored and where the merged video will be saved.
    """
    print('------------ MERGING VIDEOS ------------')

    # Define paths to the input videos
    video2_path = os.path.join(output_path, 'STEP2_face_landmarks_tracking.avi')
    video3_path = os.path.join(output_path, 'STEP3_plot_gaze.avi')

    # Open the videos
    cap1 = cv.VideoCapture(video2_path)
    cap2 = cv.VideoCapture(video3_path)

    # Get properties of the videos
    fps = int(cap1.get(cv.CAP_PROP_FPS))
    width1 = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Determine the maximum height to scale both videos proportionally
    max_height = max(height1, height2)

    # Scale widths to maintain aspect ratio
    width1_scaled = int(width1 * (max_height / height1))
    width2_scaled = int(width2 * (max_height / height2))

    # Calculate the total width of the merged video
    total_width = width1_scaled + width2_scaled

    # Define the output video writer
    output_video_path = os.path.join(output_path, 'SUMMARY_VID.avi')
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_video_path, fourcc, fps, (total_width, max_height))

    while True:
        # Read frames from each video
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Break the loop if any of the videos end
        if not ret1 or not ret2:
            break

        # Resize frames to match the maximum height while maintaining aspect ratio
        frame1_resized = cv.resize(frame1, (width1_scaled, max_height))
        frame2_resized = cv.resize(frame2, (width2_scaled, max_height))

        # Combine the frames side by side
        combined_frame = cv.hconcat([frame1_resized, frame2_resized])

        # Write the combined frame to the output video
        out.write(combined_frame)

    # Release all resources
    cap1.release()
    cap2.release()
    out.release()

    print(f"Merged video saved at: {output_video_path}")
