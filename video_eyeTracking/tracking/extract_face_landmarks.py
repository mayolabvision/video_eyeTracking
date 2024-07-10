import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions.face_mesh_connections import *
import numpy as np
import os
import csv
import time

## Head Pose Estimation Landmark Indices
# These indices correspond to the specific facial landmarks used for head pose estimation.
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291

USER_FACE_WIDTH = 140  # [mm]

# Face Selected points indices for Head Pose Estimation
_indices_pose = [1, 33, 61, 199, 263, 291]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to calculate vector position
def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1

def estimate_head_pose(landmarks, image_size):
    # Scale factor based on user's face width (assumes model face width is 150mm)
    scale_factor = USER_FACE_WIDTH / 150.0
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0 * scale_factor, -65.0 * scale_factor),        # Chin
        (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),     # Left eye left corner
        (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),      # Right eye right corner
        (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),    # Left Mouth corner
        (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)      # Right mouth corner
    ])
    
    # Camera internals
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4,1))

    # 2D image points from landmarks, using defined indices
    image_points = np.array([
        landmarks[NOSE_TIP_INDEX],            # Nose tip
        landmarks[CHIN_INDEX],                # Chin
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
        landmarks[LEFT_MOUTH_CORNER_INDEX],      # Left mouth corner
        landmarks[RIGHT_MOUTH_CORNER_INDEX]      # Right mouth corner
    ], dtype="double")


        # Solve for pose
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    # Combine rotation matrix and translation vector to form a 3x4 projection matrix
    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

    # Decompose the projection matrix to extract Euler angles
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()[:3]


     # Normalize the pitch angle
    pitch = normalize_pitch(pitch)

    return pitch, yaw, roll



def normalize_pitch(pitch):
    """
    Normalize the pitch angle to be within the range of [-90, 90].

    Args:
        pitch (float): The raw pitch angle in degrees.

    Returns:
        float: The normalized pitch angle.
    """
    # Map the pitch angle to the range [-180, 180]
    if pitch > 180:
        pitch -= 360

    # Invert the pitch angle for intuitive up/down movement
    pitch = -pitch

    # Ensure that the pitch is within the range of [-90, 90]
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
        
    pitch = -pitch

    return pitch

def extract_face_landmarks(video_path, min_detection_confidence=0.75, min_tracking_confidence=0.75, logged_landmarks=None, output_path=None):
    print('------------ EXTRACTING FACE LANDMARKS ------------')
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0.25)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)

    time.sleep(1)
    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(output_path, 'EYE_TRACKING.avi')
        out = cv.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    csv_data = []
    # Column names for CSV file
    column_names = [
        "Frame Number",
        "Timestamp (ms)",
        "Iris Center (Left,X)",
        "Iris Center (Left,Y)",
        "Iris Center (Right,X)",
        "Iris Center (Right,Y)",
        "Rel Iris Center (Left,dX)",
        "Rel Iris Center (Left,dY)",
        "Rel Iris Center (Right,dX)",
        "Rel Iris Center (Right,dY)",
        "Head Pose (Pitch)",
        "Head Pose (Yaw)",
        "Head Pose (Roll)"
    ]

    # Column names for CSV file
    if logged_landmarks:
        column_names.extend(
            [f"Landmark_{i}_X" for i in logged_landmarks]
            + [f"Landmark_{i}_Y" for i in logged_landmarks]
            + [f"Landmark_{i}_Z" for i in logged_landmarks])
    else:
        column_names.extend(
        [f"Landmark_{i}_X" for i in range(478)]
        + [f"Landmark_{i}_Y" for i in range(478)]
        + [f"Landmark_{i}_Z" for i in range(478)])

    initial_pitch, initial_yaw, initial_roll = None, None, None 
    for thisFrame in range(num_frames):
        ret, frame = cap.read()
        timestamp = thisFrame * (1/fps)
        if not ret:
            break

        frame.flags.writeable = False
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
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
                '''
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=FACEMESH_IRISES,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                '''
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark])
            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark])

            # getting the head pose estimation 3d points
            head_pose_points_3D = np.multiply(
                mesh_points_3D[_indices_pose], [img_w, img_h, 1])
            head_pose_points_2D = mesh_points[_indices_pose]

            # collect nose three dimension and two dimension points
            nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
            nose_2D_point = head_pose_points_2D[0]

            # create the camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
            head_pose_points_3D = head_pose_points_3D.astype(np.float64)
            head_pose_points_2D = head_pose_points_2D.astype(np.float64)
            # Solve PnP
            success, rot_vec, trans_vec = cv.solvePnP(
                head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
            )
            # Get rotational matrix
            rotation_matrix, jac = cv.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

            # Get the y rotation degree
            angle_x = angles[0] * 360
            angle_y = angles[1] * 360
            z = angles[2] * 360

            nose_3d_projection, jacobian = cv.projectPoints(
            nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = nose_2D_point
            p2 = (
                int(nose_2D_point[0] + angle_y * 10),
                int(nose_2D_point[1] - angle_x * 10),)

            cv.line(frame, p1, p2, (255, 0, 255), 3)

            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Highlighting the irises and corners of the eyes
            cv.circle(
                frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Left iris
            cv.circle(
                frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Right iris
            
            # Calculating relative positions
            l_dx, l_dy = vector_position(mesh_points[LEFT_EYE_OUTER_CORNER], center_left)
            r_dx, r_dy = vector_position(mesh_points[RIGHT_EYE_OUTER_CORNER], center_right)


            pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
            #angle_buffer.add([pitch, yaw, roll])
            #pitch, yaw, roll = angle_buffer.get_average()

            if initial_pitch is None:
                initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
            pitch -= initial_pitch
            yaw -= initial_yaw
            roll -= initial_roll

            log_entry = [(thisFrame+1),timestamp,l_cx,l_cy,r_cx,r_cy,l_dx,l_dy,r_dx,r_dy]
            log_entry.extend([pitch, yaw, roll])
            if logged_landmarks:
                log_entry.extend([p for point in mesh_points_3D[logged_landmarks] for p in point])
            else:
                log_entry.extend([p for point in mesh_points_3D for p in point])
            
            csv_data.append(log_entry)
        
        if output_path:
            out.write(frame)

        cv.imshow('MediaPipe Face Mesh', frame)
        if cv.waitKey(5) & 0xFF == 27:
            break
        
    cap.release()
    if output_path:
        out.release()
    cv.destroyAllWindows()

    # Writing data to CSV file
    print("Writing data to CSV...")
    csv_file_name = os.path.join(
        output_path, f"EYE_LANDMARKS_LOGS.csv"
    )
    with open(csv_file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)  # Writing column names
        writer.writerows(csv_data)  # Writing data rows

    return img_w, img_h

