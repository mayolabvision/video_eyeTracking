import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import random

def crop_video_based_on_face_detection(video_path, min_detection_confidence=0.99, duration=10, repeats=10, output_path=None):
    """ 
    This function crops the video based on face detection.
    
    Parameters:
        video_path (str): Path to the input video.
        min_detection_confidence (float): Minimum confidence for face detection.
        duration (int): Duration of the video to process.
        repeats (int): Number of times to repeat the detection process.
        output_path (str or None): Path to save the output video. If None, the video is not saved.
    
    Returns:
        list: Bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Define the number of segments based on repeats
    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration_frames = int(fps * duration)
    segments = max(1, repeats)  # Ensure there is at least one segment
    possible_starts = [i * (num_frames - duration_frames) // (segments - 1) for i in range(segments)]

    # Loop through each snippet of the video 
    all_bboxes = []
    for start_frame in possible_starts:
        # Load the video
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        # Initialize MediaPipe face detection
        with mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence) as face_detection:
            # Iterate through each frame in the video
            for _ in range(duration_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Detect faces in the frame
                results = face_detection.process(frame_rgb)

                # If faces are detected
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x1 = int(bboxC.xmin * w)
                        y1 = int(bboxC.ymin * h)
                        x2 = x1 + int(bboxC.width * w)
                        y2 = y1 + int(bboxC.height * h)

                        # Adjust coordinates to make the bounding box fit better
                        padding_x = int(bboxC.width * w * 0.2)  # 20% padding on the width
                        padding_y = int(bboxC.height * h * 0.2)  # 20% padding on the height                                                                                                                                                                                                                                                                                                                                                                                                                                      
                        x1 = max(0, x1 - padding_x)
                        y1 = max(0, y1 - padding_y)
                        x2 = min(w, x2 + padding_x)
                        y2 = min(h, y2 + padding_y)

                        # Collect bounding boxes
                        all_bboxes.append([x1, y1, x2, y2])

    cap.release()
    cv.destroyAllWindows()

    if not all_bboxes:
        raise ValueError("Could not detect a face in the video.")

    # Convert all bounding boxes to a numpy array
    all_bboxes = np.array(all_bboxes)

    # Calculate percentiles for each coordinate
    percentiles = np.percentile(all_bboxes, [2.5, 97.5], axis=0)

    # Filter out bounding boxes that are outside the 95th percentile
    filtered_bboxes = all_bboxes[
        (all_bboxes[:, 0] >= percentiles[0, 0]) & (all_bboxes[:, 0] <= percentiles[1, 0]) &
        (all_bboxes[:, 1] >= percentiles[0, 1]) & (all_bboxes[:, 1] <= percentiles[1, 1]) &
        (all_bboxes[:, 2] >= percentiles[0, 2]) & (all_bboxes[:, 2] <= percentiles[1, 2]) &
        (all_bboxes[:, 3] >= percentiles[0, 3]) & (all_bboxes[:, 3] <= percentiles[1, 3])
    ]

    # Calculate the min and max of the filtered bounding boxes
    final_bbox = [
        int(np.min(filtered_bboxes[:, 0])),  # min of x1
        int(np.min(filtered_bboxes[:, 1])),  # min of y1
        int(np.max(filtered_bboxes[:, 2])),  # max of x2
        int(np.max(filtered_bboxes[:, 3]))   # max of y2
    ]

    if output_path:
        # Select a random 10-second snippet for cropped_face_detection.avi
        cap = cv.VideoCapture(video_path)
        random_start_frame = random.randint(0, max(0, num_frames - duration_frames))
        cap.set(cv.CAP_PROP_POS_FRAMES, random_start_frame)


        # Define the codec and create VideoWriter objects if saving the videos
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(output_path, 'cropped_face_detection.avi')
        out = cv.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

        full_video_path = os.path.join(output_path, 'cropped_fullVideo.avi')
        out_full = cv.VideoWriter(full_video_path, fourcc, fps, (final_bbox[2] - final_bbox[0], final_bbox[3] - final_bbox[1]))

        # Write the duration snippet for cropped_face_detection.avi
        for _ in range(duration_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Draw the final bounding box on the frame
            cv.rectangle(frame, (final_bbox[0], final_bbox[1]), (final_bbox[2], final_bbox[3]), (0, 255, 0), 2)
            out.write(frame)

        cap.release()
        out.release()

        # Write the full video for cropped_fullVideo.mov
        cap = cv.VideoCapture(video_path)
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Crop the frame based on the final bounding box
            cropped_frame = frame[final_bbox[1]:final_bbox[3], final_bbox[0]:final_bbox[2]]
            out_full.write(cropped_frame)

        cap.release()
        out_full.release()
        cv.destroyAllWindows()

    return

# Example usage:
# video_path = "path/to/your/video.avi"
# output_path = make_output_directory(video_path)
# cropped_coords = crop_video_based_on_face_detection(video_path, output_path=output_path)
# print(f"Cropped coordinates: {cropped_coords}")
