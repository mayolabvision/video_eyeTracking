import cv2 as cv
import os
import mediapipe as mp
import numpy as np
import random
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

def evaluate_confidences(video_path, detection_confidence, tracking_confidence, duration, repeats):
    # Define the number of segments based on repeats
    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    duration_frames = int(fps * duration)
    segments = max(1, repeats)  # Ensure there is at least one segment
    possible_starts = [i * (num_frames - duration_frames) // (segments - 1) for i in range(segments)]

    # Loop through each snippet of the video 
    detection_rates = []
    for start_frame in possible_starts:
        # Load the video
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        detected_frames = 0
        total_frames = 0
        for _ in range(duration_frames):
            ret, frame = cap.read()
            if not ret:
                break
    
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = mp_face_mesh.process(frame_rgb)

            total_frames += 1
            if results.multi_face_landmarks:
                detected_frames += 1

        detection_rate = detected_frames / total_frames
        detection_rates.append(detection_rate)

    cap.release()
    cv.destroyAllWindows()

    return np.mean(detection_rates)

def find_optimal_confidences(video_path, duration=30, repeats=5, min_confidence_threshold=0.75, min_tracking_confidence=0.95, output_path=None):
    
    optimal_detection_confidence = None
    for dc in np.flip(np.arange(min_confidence_threshold, 1, 0.01)): # coarser param sweep
        detection_rate = evaluate_confidences(video_path, dc, min_tracking_confidence, duration, repeats)
        #print(f'dc{dc} = {detection_rate}')
        if detection_rate >= 0.5: # start fine-tuning
            for fine_dc in np.flip(np.arange(min_confidence_threshold, dc, 0.001)):
                fine_detection_rate = evaluate_confidences(video_path, fine_dc, min_tracking_confidence, duration, repeats)
                #print(f'fine_dc{fine_dc} = {fine_detection_rate}')
                if fine_detection_rate >= 0.95:
                    optimal_detection_confidence = round(fine_dc, 3)
                    break
            if optimal_detection_confidence is not None:
                break

    if output_path:
        cap = cv.VideoCapture(video_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        duration_frames = int(fps * duration)
        possible_starts = [0, (num_frames - duration_frames) // 4, (num_frames - duration_frames) // 2, 3 * (num_frames - duration_frames) // 4, max(0, num_frames - duration_frames)]
        start_frame = random.choice(possible_starts)
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    
        # Define the codec and create a VideoWriter object if saving the video
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(output_path, 'optimal_detection_thresholds.avi')
        out = cv.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=optimal_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        for _ in range(duration_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = mp_face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                img_h, img_w, _ = frame.shape
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * img_w)
                        y = int(landmark.y * img_h)
                        cv.circle(frame, (x, y), 1, (255, 0, 0), -1)


                height, width = frame.shape[:2]
                cv.putText(frame, f'Detection Confidence: {optimal_detection_confidence:.3f}', (10, height - 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                cv.putText(frame, f'Tracking Confidence: {min_tracking_confidence:.3f}', (10, height - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

                out.write(frame)

        cap.release()
        out.release()
        cv.destroyAllWindows()

    return round(optimal_detection_confidence, 3), round(min_tracking_confidence, 3)

# Example usage:
# video_path = "path/to/your/video.avi"
# optimal_detection_confidence, optimal_tracking_confidence = find_optimal_confidences(video_path, duration=20, repeats=3, output_path='path/to/output')
# print(f"Optimal Detection Confidence: {optimal_detection_confidence}")
# print(f"Optimal Tracking Confidence: {optimal_tracking_confidence}")
