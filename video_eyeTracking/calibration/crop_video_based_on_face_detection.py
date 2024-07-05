import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import random

def crop_video_based_on_face_detection(video_path, min_detection_confidence=0.99, duration=10, repeats=10, output_path=None, showVideo=0):
    """
    This function crops the video based on face detection.
    
    Parameters:
        video_path (str): Path to the input video.
        min_detection_confidence (float): Minimum confidence for face detection.
        duration (int): Duration of the video to process.
        repeats (int): Number of times to repeat the detection process.
        output_path (str or None): Path to save the output video. If None, the video is not saved.
        showVideo (int): If set to 1, display the frames in real-time with face mesh points and bounding box.
    
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
                        padding_x = int(bboxC.width * w * 0.1)  # 10% padding on the width
                        padding_y = int(bboxC.height * h * 0.1)  # 10% padding on the height
                        x1 = max(0, x1 - padding_x)
                        y1 = max(0, y1 - padding_y)
                        x2 = min(w, x2 + padding_x)
                        y2 = min(h, y2 + padding_y)

                        # Collect bounding boxes
                        all_bboxes.append([x1, y1, x2, y2])

    cap.release()
    cv.destroyAllWindows()

    if all_bboxes:
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

        # Calculate the mean of the filtered bounding boxes
        final_bbox = np.mean(filtered_bboxes, axis=0)
        final_bbox = [int(final_bbox[0]), int(final_bbox[1]), int(final_bbox[2]), int(final_bbox[3])]

        if showVideo == 1 or output_path:
            # Select a random 10-second snippet
            cap = cv.VideoCapture(video_path)
            random_start_frame = random.randint(0, max(0, num_frames - duration_frames))
            cap.set(cv.CAP_PROP_POS_FRAMES, random_start_frame)

            # Define the codec and create a VideoWriter object if saving the video
            if output_path:
                fourcc = cv.VideoWriter_fourcc(*'XVID')
                output_video_path = os.path.join(output_path, 'cropped_face_detection.avi')
                out = cv.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

            for _ in range(duration_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw the final bounding box on the frame
                cv.rectangle(frame, (final_bbox[0], final_bbox[1]), (final_bbox[2], final_bbox[3]), (0, 255, 0), 2)
                
                if output_path:
                    out.write(frame)

                # Display the frame with the bounding box if showVideo is set to 1
                if showVideo == 1:
                    cv.imshow('Final Crop', frame)
                    cv.waitKey(1)  # Display each frame for 1 ms

            cap.release()
            if output_path:
                out.release()
            cv.destroyAllWindows()

        return final_bbox
    else:
        return None

# Example usage:
# video_path = "path/to/your/video.avi"
# output_path = make_output_directory(video_path)
# cropped_coords = crop_video_based_on_face_detection(video_path, output_path=output_path, showVideo=1)
# print(f"Cropped coordinates: {cropped_coords}")
