import cv2 as cv
import mediapipe as mp
import numpy as np
import random

def crop_video_based_on_face_detection(video_path, min_detection_confidence=0.99, duration=20, repeats=5, showVideo=0):
    """
    This function crops the video based on face detection.
    
    Parameters:
        video_path (str): Path to the input video.
        min_detection_confidence (float): Minimum confidence for face detection.
        duration (int): Duration of the video to process.
        repeats (int): Number of times to repeat the detection process.
        showVideo (int): If set to 1, display the frames in real-time with face mesh points and bounding box.
    
    Returns:
        list: Bounding box coordinates [xmin, ymin, xmax, ymax].
    """

    # Define the number of segments based on repeats
    cap = cv.VideoCapture(video_path)
    duration_frames = int(int(cap.get(cv.CAP_PROP_FPS)) * duration)
    segments = max(1, repeats)  # Ensure there is at least one segment
    possible_starts = [i * (int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - duration_frames) // (segments - 1) for i in range(segments)]
    cap.release()

    # Loop through each snippet of the video 
    all_bboxes = []
    for start_frame in possible_starts:
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils

        # Load the video
        cap = cv.VideoCapture(video_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        duration_frames = int(fps * duration)
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        # Initialize MediaPipe face detection
        with mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence) as face_detection:

            # Initialize variables to store extreme coordinates
            xmin, ymin = float('inf'), float('inf')
            xmax, ymax = float('-inf'), float('-inf')

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

                        # Update extreme coordinates
                        xmin = min(xmin, x1)
                        ymin = min(ymin, y1)
                        xmax = max(xmax, x2)
                        ymax = max(ymax, y2)

                        # Draw bounding box and face mesh points if showVideo is set to 1
                        if showVideo == 1:
                            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display the frame with detected face
                if showVideo == 1:
                    cv.imshow('Face Detection', frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break

            # Append the extreme coordinates for this repeat
            if xmin < float('inf') and ymin < float('inf') and xmax > float('-inf') and ymax > float('-inf'):
                all_bboxes.append([xmin, ymin, xmax, ymax])

        cap.release()

    # Close all OpenCV windows
    cv.destroyAllWindows()

    if all_bboxes:
        # Calculate the overall min and max coordinates across all repeats
        all_bboxes = np.array(all_bboxes)
        final_xmin = np.min(all_bboxes[:, 0])
        final_ymin = np.min(all_bboxes[:, 1])
        final_xmax = np.max(all_bboxes[:, 2])
        final_ymax = np.max(all_bboxes[:, 3])
        return [final_xmin, final_ymin, final_xmax, final_ymax]
    else:
        return None

# Example usage:
# video_path = "path/to/your/video.avi"
# cropped_coords = crop_video_based_on_face_detection(video_path, showVideo=1)
# print(f"Cropped coordinates: {cropped_coords}")
