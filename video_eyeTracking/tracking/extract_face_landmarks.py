import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions.face_mesh_connections import *
import numpy as np
import os
import csv
import time
from tqdm import tqdm  # Importing tqdm for the progress bar
from .landmarks import LEFT_EYE_IRIS, RIGHT_EYE_IRIS, LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER, _indices_pose
from .head_pose import vector_position, estimate_head_pose
from .gaze import gaze

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_face_landmarks(video_path, min_detection_confidence=0.9, min_tracking_confidence=0.9, output_path=None):
    print('------------ EXTRACTING FACE LANDMARKS ------------')
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)

    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if output_path:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(output_path, 'STEP2_face_landmarks_tracking.avi')
        out = cv.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    land_data, gaze_data = [],[]
    column_names = ["Frame Number","Time Aligned to Video Start (ms)"]

    # Columns for all landmarks
    land_columns = column_names + [f"Landmark_{i}_X" for i in range(478)] + [f"Landmark_{i}_Y" for i in range(478)]

    # Columns for gaze-related data
    gaze_columns = [
        "Frame Number",                             # Frame index, no units.
        "Time Aligned to Video Start (ms)",                           # Time in milliseconds.

        "Horizontal Gaze (-100,100)",               # Horizontal direction of gaze in the room, normalized.
        "Vertical Gaze (-100,100)",                 # Vertical direction of gaze in the room, normalized.

        "Horizontal Deviation (Left Eye)",          # Horizontal deviation of left eye from head direction, normalized.
        "Vertical Deviation (Left Eye)",            # Vertical deviation of left eye from head direction, normalized.

        "Horizontal Deviation (Right Eye)",         # Horizontal deviation of right eye from head direction, normalized.
        "Vertical Deviation (Right Eye)",           # Vertical deviation of right eye from head direction, normalized.

        "Gaze Distance (0,100)",                   # Estimated distance of gaze target, 0 (close) to 100 (far).
    ]

    initial_pitch, initial_yaw, initial_roll = 0, 0, 0
    with tqdm(total=num_frames, desc="Processing Frames") as pbar:
        for thisFrame in range(num_frames):
            ret, frame = cap.read()
            timestamp = (thisFrame+1) * (1/fps)
            if not ret:
                break

            pbar.update(1)

            frame.flags.writeable = False
            frame = cv.cvtColor(cv.flip(frame,1), cv.COLOR_BGR2RGB)
            results = mp_face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            img_h, img_w = frame.shape[:2]

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                # Convert landmark x and y to pixel coordinates
                mesh_points = np.array([
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark])
                mesh_points_3D = np.array(
                    [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark])

                # Gaze estimation 
                gaze_x, gaze_y, pos_x_left, pos_y_left, pos_x_right, pos_y_right, proximity = gaze(frame, results.multi_face_landmarks[0]) # gaze estimation

                # entry for landmark data
                land_log_entry = [(thisFrame+1),timestamp]
                land_log_entry.extend([p for point in mesh_points for p in point])
                land_data.append(land_log_entry)

                gaze_log_entry = [
                    (thisFrame + 1), timestamp,
                    gaze_x, gaze_y, pos_x_left, pos_y_left, pos_x_right, pos_y_right, proximity
                ]
                gaze_data.append(gaze_log_entry)

            else:
                # If no face is detected, append empty values for the landmark and gaze data
                land_log_entry = [(thisFrame+1), timestamp] + ['' for _ in range(len(land_columns) - 2)]
                land_data.append(land_log_entry)

                gaze_log_entry = [(thisFrame + 1), timestamp] + ['' for _ in range(len(gaze_columns) - 2)]
                gaze_data.append(gaze_log_entry)

            if output_path:
                out.write(cv.flip(frame, 1))
                # Display the min_detection_confidence and min_tracking_confidence as the window title
                window_title = f"Detection Conf: {min_detection_confidence}, Tracking Conf: {min_tracking_confidence}"
                cv.imshow(window_title, cv.flip(frame, 1))
            else:
                cv.imshow('MediaPipe Face Mesh', cv.flip(frame, 1))

            if cv.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if output_path:
        out.release()
    cv.destroyAllWindows()

    # Writing data to CSV file
    print("Writing data to CSV...")
    csv_file_name = os.path.join(output_path, f"FACE_LANDMARKS_LOGS.csv")
    with open(csv_file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(land_columns)  # Writing column names
        writer.writerows(land_data)  # Writing data rows

    gaze_file_name = os.path.join(output_path, f"EYE_GAZE_LOGS.csv")
    with open(gaze_file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(gaze_columns)  # Writing intuitive column names
        writer.writerows(gaze_data)  # Writing data rows

    return min_detection_confidence, min_tracking_confidence
