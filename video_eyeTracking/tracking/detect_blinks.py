import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import random

RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

UPPER_LOWER_LANDMARKS = {
    'right': ([37, 38, 43, 44], [41, 40, 46, 47]),
    'left': ([443, 444, 450, 451], [447, 446, 453, 454])
}

EYE_AR_CONSEC_FRAMES = 2
BLINK_DETECTION_WINDOW = 30

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(landmarks, side):
    upper_points, lower_points = UPPER_LOWER_LANDMARKS[side]
    upper_mean = np.mean([landmarks[pt] for pt in upper_points], axis=0)
    lower_mean = np.mean([landmarks[pt] for pt in lower_points], axis=0)
    return euclidean_distance(upper_mean, lower_mean)

def detect_blinks(video_path, min_detection_confidence=0.75, min_tracking_confidence=0.75, output_path=None):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    img_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    img_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    if output_path:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(output_path, 'detected_blinks.avi')
        out = cv.VideoWriter(output_video_path, fourcc, fps, (img_w, img_h))

    blink_counter = 0
    frame_counter = 0
    blink_thresholds = []
    ignore_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = mp_face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mesh_points_3D = np.array([[p.x * img_w, p.y * img_h, p.z * img_w] for p in face_landmarks.landmark])

                right_eye_ratio = eye_aspect_ratio(mesh_points_3D, 'right')
                left_eye_ratio = eye_aspect_ratio(mesh_points_3D, 'left')
                eye_ratio = (right_eye_ratio + left_eye_ratio) / 2

                if ignore_frames == 0:
                    blink_thresholds.append(eye_ratio)
                    if len(blink_thresholds) > BLINK_DETECTION_WINDOW:
                        blink_thresholds.pop(0)

                baseline = np.mean(blink_thresholds)
                deviation = baseline * 0.8  # Adjust this factor as needed

                if eye_ratio < deviation:
                    frame_counter += 1
                else:
                    if frame_counter >= EYE_AR_CONSEC_FRAMES:
                        blink_counter += 1
                        cv.putText(frame, "BLINK", (10, img_h - 20),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv.LINE_AA)
                        ignore_frames = EYE_AR_CONSEC_FRAMES + 2  # Skip blink frames and a few after
                    frame_counter = 0

                if ignore_frames > 0:
                    ignore_frames -= 1

        cv.putText(frame, f'Detection Confidence: {min_detection_confidence:.2f}', (10, img_h - 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, f'Tracking Confidence: {min_tracking_confidence:.2f}', (10, img_h - 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)

        if output_path:
            out.write(frame)

        cv.imshow('Blink Detection', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv.destroyAllWindows()

    print(f'Total blinks detected: {blink_counter}')

# Example usage:
# video_path = "path/to/your/video.avi"
# detect_blinks(video_path, output_path='path/to/output')
