import random
import cv2 as cv
import mediapipe as mp
import numpy as np

RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

MIN_CONFIDENCE_THRESHOLD = 0.5
CONFIDENCE_STEP = 0.05
EYE_AR_CONSEC_FRAMES = 2

def euclidean_distance_3D(points):
    """Calculates the Euclidean distance between two points in 3D space."""
    P0, P3, P4, P5, P8, P11, P12, P13 = points

    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )

    denominator = 3 * np.linalg.norm(P0 - P8) ** 3

    distance = numerator / denominator

    return distance

def blinking_ratio(landmarks):
    """Calculates the blinking ratio of a person."""
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])

    ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

    return ratio

def calibrate_blink_threshold(video_path, face_crop_coords, min_detection_confidence=0.75, min_tracking_confidence=0.75,  duration=30, repeats=3, showVideo=0):
    """
    Calibrates the blink threshold based on the input video.

    Parameters:
        video_path (str): Path to the input video.
        face_crop_coords (list): Coordinates to crop the face [xmin, ymin, xmax, ymax].
        min_detection_confidence (float): Minimum confidence for face detection.
        min_tracking_confidence (float): Minimum confidence for face tracking.
        duration (int): Duration of the video to process.
        repeats (int): Number of times to repeat the detection process.
        showVideo (int): If set to 1, display the frames in real-time.

    Returns:
        float: The calculated blink threshold.
    """
    while min_detection_confidence >= MIN_CONFIDENCE_THRESHOLD:
        all_blink_ratios = []

        for _ in range(repeats):
            mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

            cap = cv.VideoCapture(video_path)
            fps = int(cap.get(cv.CAP_PROP_FPS))
            num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

            duration_frames = int(fps * duration)
            start_frame = random.randint(0, max(0, num_frames - duration_frames))
            cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

            xmin, ymin, xmax, ymax = face_crop_coords

            for _ in range(duration_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                cropped_frame = frame[ymin:ymax, xmin:xmax]
                frame_rgb = cv.cvtColor(cropped_frame, cv.COLOR_BGR2RGB)
                results = mp_face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    img_h, img_w, _ = cropped_frame.shape
                    for face_landmarks in results.multi_face_landmarks:
                        mesh_points_3D = np.array([[p.x * img_w, p.y * img_h, p.z * img_w] for p in face_landmarks.landmark])

                        eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
                        all_blink_ratios.append(eyes_aspect_ratio)

                        if showVideo == 1:
                            for landmark in face_landmarks.landmark:
                                x = int(landmark.x * img_w)
                                y = int(landmark.y * img_h)
                                cv.circle(cropped_frame, (x, y), 1, (255, 0, 0), -1)

                            cv.putText(cropped_frame, f'Blink Ratio: {eyes_aspect_ratio:.3f}', (10, 30),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

                            cv.imshow('Blink Calibration', cropped_frame)
                            if cv.waitKey(1) & 0xFF == ord('q'):
                                break

            cap.release()
            cv.destroyAllWindows()

        if all_blink_ratios:
            mean_blink_ratio = np.mean(all_blink_ratios)
            std_blink_ratio = np.std(all_blink_ratios)
            blink_threshold = mean_blink_ratio - std_blink_ratio  # Adaptive threshold based on mean and standard deviation

            # Ensure the threshold is within reasonable bounds
            blink_threshold = max(0, min(blink_threshold, 1))

            # Show a snippet of the video with blink detection
            cap = cv.VideoCapture(video_path)
            cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)  # Start from the same frame

            eyes_blink_frame_counter = 0
            total_blinks = 0

            for _ in range(duration_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                cropped_frame = frame[ymin:ymax, xmin:xmax]
                frame_rgb = cv.cvtColor(cropped_frame, cv.COLOR_BGR2RGB)
                results = mp_face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    img_h, img_w, _ = cropped_frame.shape
                    for face_landmarks in results.multi_face_landmarks:
                        mesh_points_3D = np.array([[p.x * img_w, p.y * img_h, p.z * img_w] for p in face_landmarks.landmark])

                        eyes_aspect_ratio = blinking_ratio(mesh_points_3D)

                        if eyes_aspect_ratio <= blink_threshold:
                            eyes_blink_frame_counter += 1
                        else:
                            if eyes_blink_frame_counter > EYE_AR_CONSEC_FRAMES:
                                total_blinks += 1
                                cv.putText(cropped_frame, "BLINK", (img_w - 100, img_h - 20),
                                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv.LINE_AA)
                            eyes_blink_frame_counter = 0

                        if showVideo == 1:
                            cv.imshow('Blink Detection', cropped_frame)
                            if cv.waitKey(1) & 0xFF == ord('q'):
                                break

            cap.release()
            cv.destroyAllWindows()

            return blink_threshold
        else:
            min_detection_confidence -= CONFIDENCE_STEP

    return None

# Example usage:
# video_path = "path/to/your/video.avi"
# face_crop_coords = [xmin, ymin, xmax, ymax]  # Output from crop_video_based_on_face_detection function
# blink_threshold = calibrate_blink_threshold(video_path, face_crop_coords, showVideo=1)
# print(f"Calculated blink threshold: {blink_threshold}")
