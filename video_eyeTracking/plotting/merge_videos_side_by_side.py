import cv2 as cv
import os

def merge_videos_side_by_side(output_path):
    """
    Merges three videos side-by-side into one video by resizing them to the same height.

    Parameters:
    - output_path: Path where the videos are stored and where the merged video will be saved.
    """
    print('------------ MERGING VIDEOS ------------')

    # Define paths to the input videos
    video1_path = os.path.join(output_path, 'STEP1_crop_video_to_face.avi')
    video2_path = os.path.join(output_path, 'STEP2_face_landmarks_tracking.avi')
    video3_path = os.path.join(output_path, 'STEP3_plot_gaze.avi')

    # Open the videos
    cap1 = cv.VideoCapture(video1_path)
    cap2 = cv.VideoCapture(video2_path)
    cap3 = cv.VideoCapture(video3_path)

    # Get properties of the videos
    fps = int(cap1.get(cv.CAP_PROP_FPS))
    width1 = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv.CAP_PROP_FRAME_HEIGHT))
    width3 = int(cap3.get(cv.CAP_PROP_FRAME_WIDTH))
    height3 = int(cap3.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Determine the smallest height
    min_height = min(height1, height2, height3)

    # Resize all videos to the smallest height while maintaining aspect ratio
    width1 = int(width1 * (min_height / height1))
    width2 = int(width2 * (min_height / height2))
    width3 = int(width3 * (min_height / height3))

    total_width = width1 + width2 + width3

    # Define the output video writer
    output_video_path = os.path.join(output_path, 'SUMMARY_VID.avi')
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_video_path, fourcc, fps, (total_width, min_height))

    while True:
        # Read frames from each video
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        # Break the loop if any of the videos end
        if not ret1 or not ret2 or not ret3:
            break

        # Resize frames to match the smallest height
        frame1 = cv.resize(frame1, (width1, min_height))
        frame2 = cv.resize(frame2, (width2, min_height))
        frame3 = cv.resize(frame3, (width3, min_height))

        # Combine the frames side by side
        combined_frame = cv.hconcat([frame1, frame2, frame3])

        # Write the combined frame to the output video
        out.write(combined_frame)

    # Release all resources
    cap1.release()
    cap2.release()
    cap3.release()
    out.release()

    print(f"Merged video saved at: {output_video_path}")
