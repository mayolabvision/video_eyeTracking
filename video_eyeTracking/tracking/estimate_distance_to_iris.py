import cv2 as cv
import numpy as np
import mediapipe as mp

def relative(landmark, shape):
    return (int(landmark.x * shape[1]), int(landmark.y * shape[0]))

def estimate_distance_to_iris(video_path):
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv.VideoCapture(video_path)

    iris_distances = []

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # Ensure to get iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process the frame and get the landmarks
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract left iris landmarks
                    left_iris = [
                        face_landmarks.landmark[474],
                        face_landmarks.landmark[475],
                        face_landmarks.landmark[476],
                        face_landmarks.landmark[477]
                    ]

                    # Calculate the average iris diameter in pixels
                    iris_diameter_pixels = np.mean([
                        np.linalg.norm(np.array([relative(left_iris[0], frame.shape)]) - np.array([relative(left_iris[1], frame.shape)])),
                        np.linalg.norm(np.array([relative(left_iris[1], frame.shape)]) - np.array([relative(left_iris[2], frame.shape)])),
                        np.linalg.norm(np.array([relative(left_iris[2], frame.shape)]) - np.array([relative(left_iris[3], frame.shape)])),
                        np.linalg.norm(np.array([relative(left_iris[3], frame.shape)]) - np.array([relative(left_iris[0], frame.shape)]))
                    ])

                    # Assuming an average iris diameter of 12 mm
                    iris_diameter_mm = 12.0

                    # Estimate the distance to the face (in mm) based on the iris size in pixels
                    focal_length = frame.shape[1]  # Assuming the focal length is the width of the frame in pixels
                    distance_to_face = (focal_length * iris_diameter_mm) / iris_diameter_pixels

                    iris_distances.append(distance_to_face)

    cap.release()

    # Return the average distance
    if iris_distances:
        return np.mean(iris_distances)
    else:
        raise ValueError("No face detected in the video.")
