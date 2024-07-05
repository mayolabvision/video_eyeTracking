import cv2 as cv
import mediapipe as mp
import numpy as np
import random
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

def evaluate_confidences(confidences, video_path, face_crop_coords, duration=30, repeats=5):
    detection_confidence, tracking_confidence = confidences

    # Define the number of segments based on repeats
    cap = cv.VideoCapture(video_path)
    duration_frames = int(int(cap.get(cv.CAP_PROP_FPS)) * duration)
    segments = max(1, repeats)  # Ensure there is at least one segment
    possible_starts = [i * (int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - duration_frames) // (segments - 1) for i in range(segments)]
    cap.release()
    cv.destroyAllWindows()

    # Loop through each snippet of the video 
    detection_rates = []
    for start_frame in possible_starts:
        # Load the video
        cap = cv.VideoCapture(video_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        duration_frames = int(fps * duration)
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        xmin, ymin, xmax, ymax = face_crop_coords
        detected_frames = 0

        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        for _ in range(duration_frames):
            ret, frame = cap.read()
            if not ret:
                break

            cropped_frame = frame[ymin:ymax, xmin:xmax]
            frame_rgb = cv.cvtColor(cropped_frame, cv.COLOR_BGR2RGB)
            results = mp_face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                detected_frames += 1

        detection_rate = detected_frames / duration_frames
        detection_rates.append(detection_rate)

        cap.release()
        cv.destroyAllWindows()

    mean_detection_rate = np.mean(detection_rates)
    # We want to maximize detection_confidence and tracking_confidence, but ensure detection rate is >= 0.95
    if mean_detection_rate >= 0.95:
        return -mean_detection_rate  # Higher detection rate should result in a lower (better) objective value
    else:
        return 1.0  # A high value indicating this combination is not acceptable

def find_optimal_confidences(video_path, face_crop_coords, duration=30, repeats=5, min_confidence_threshold=0.75, showVideo=0):
    # Define the search space for Bayesian Optimization
    space = [Real(min_confidence_threshold, 1.0, name='detection_confidence'), Real(min_confidence_threshold, 1.0, name='tracking_confidence')]

    @use_named_args(space)
    def objective(**params):
        return evaluate_confidences([params['detection_confidence'], params['tracking_confidence']], video_path, face_crop_coords, duration, repeats)

    # Perform Bayesian optimization
    res = gp_minimize(objective, space, n_calls=10, n_initial_points=5, acq_func='gp_hedge', random_state=0)

    optimal_detection_confidence, optimal_tracking_confidence = res.x

    # Early stopping check
    if len(res.func_vals) > 1:
        if all(abs(res.func_vals[-i] - res.func_vals[-i-1]) < 0.005 for i in range(1, min(5, len(res.func_vals)))):
            print("Early stopping: Optimization has converged.")

    if showVideo == 1:
        cap = cv.VideoCapture(video_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        duration_frames = int(fps * duration)
        possible_starts = [0, (num_frames-duration_frames)//4, (num_frames-duration_frames)//2, 3*(num_frames-duration_frames)//4, max(0,num_frames-duration_frames)]
        start_frame = random.choice(possible_starts)
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        xmin, ymin, xmax, ymax = face_crop_coords

        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=optimal_detection_confidence,
            min_tracking_confidence=optimal_tracking_confidence
        )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
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
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * img_w)
                        y = int(landmark.y * img_h)
                        cv.circle(cropped_frame, (x, y), 1, (255, 0, 0), -1)

                cv.putText(cropped_frame, f'Detection Confidence: {optimal_detection_confidence:.2f}', (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(cropped_frame, f'Tracking Confidence: {optimal_tracking_confidence:.2f}', (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

                cv.imshow('Optimal Confidence Detection', cropped_frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv.destroyAllWindows()

    return round(optimal_detection_confidence, 3), round(optimal_tracking_confidence, 3)

# Example usage:
# video_path = "path/to/your/video.avi"
# face_crop_coords = [xmin, ymin, xmax, ymax]  # Output from crop_video_based_on_face_detection function
# optimal_detection_confidence, optimal_tracking_confidence = find_optimal_confidences(video_path, face_crop_coords, duration=20, repeats=3, showVideo=1)
# print(f"Optimal Detection Confidence: {optimal_detection_confidence}")
# print(f"Optimal Tracking Confidence: {optimal_tracking_confidence}")
