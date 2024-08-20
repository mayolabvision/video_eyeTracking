import cv2 as cv
import numpy as np

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def normalize_vector(vec):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0: 
       return vec
    return vec / norm

def gaze(frame, points):
    """ 
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function calculates gaze, position (for each eye), and proximity within a range of 0 to 100.
    """

    # 2D image points.
    image_points = np.array([
        relative(points.landmark[4], frame.shape),  # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),  # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    # 2D image points for 3D transformation.
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),  # Nose tip
        relativeT(points.landmark[152], frame.shape),  # Chin
        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
        relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
        relativeT(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])  

    # 3D model eye points for both eyes
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

    # Camera matrix estimation
    frame_center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )   

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    # 2D pupil locations for both eyes
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)
    nose_tip = relative(points.landmark[4], frame.shape)
    
    # Transformation between image point to world point
    _, transformation, _ = cv.estimateAffine3D(image_points1, model_points)  # image to world transformation

    if transformation is not None:  # if estimateAffine3D succeeded
        def calculate_gaze(pupil, Eye_ball_center, nose_tip):
            # Project pupil image point into 3D world coordinates
            pupil_world_cord = transformation @ np.array([[pupil[0], pupil[1], 0, 1]]).T

            # 3D gaze point based on direction vector
            gaze_direction_3D = (pupil_world_cord - Eye_ball_center)

            # Normalize the gaze direction vector
            normalized_gaze_direction = normalize_vector(gaze_direction_3D)

            # Project the normalized 3D gaze direction onto the image plane
            S = Eye_ball_center + normalized_gaze_direction * 200  # Scale to a consistent length
            eye_pupil2D, _ = cv.projectPoints(S.T, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # Project 3D head pose into the image plane
            head_pose, _ = cv.projectPoints((pupil_world_cord.T[:, :3]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # Calculate the gaze direction vector starting from the nose tip
            gaze = nose_tip + (eye_pupil2D[0][0] - pupil) - (head_pose[0][0] - pupil)

            # Draw the gaze vector
            cv.line(frame, (int(pupil[0]), int(pupil[1])), (int(gaze[0] ), int(gaze[1])), (255, 0, 255), 2)

            return gaze

        def calculate_head_pose(nose_tip):
            # Project nose tip image point into 3D world coordinates
            nose_tip_world_cord = transformation @ np.array([[nose_tip[0], nose_tip[1], 0, 1]]).T

            # Head direction vector
            head_direction_3D = np.array([[0], [0], [1]])  # This represents the direction the head is facing in 3D space

            # Normalize the head direction vector
            normalized_head_direction = normalize_vector(head_direction_3D)

            # Project the normalized 3D head direction onto the image plane
            S = nose_tip_world_cord + normalized_head_direction * 200  # Scale to a consistent length
            nose_direction_2D, _ = cv.projectPoints(S.T, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # Project 3D nose tip position (head pose) into the image plane
            head_pose, _ = cv.projectPoints((nose_tip_world_cord.T[:, :3]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # Calculate the head direction vector on the 2D image plane
            head_direction = nose_tip + (nose_direction_2D[0][0] - nose_tip) - (head_pose[0][0] - nose_tip)

            # Draw the vector on the image frame
            cv.line(frame, (int(nose_tip[0]), int(nose_tip[1])), (int(head_direction[0]), int(head_direction[1])), (0, 0, 0), 2)

            return head_direction

        absGaze_leftEye = calculate_gaze(left_pupil, Eye_ball_center_left, nose_tip)
        absGaze_rightEye = calculate_gaze(right_pupil, Eye_ball_center_right, nose_tip)
        head_pose = calculate_head_pose(nose_tip)

        # Center coordinates so [0,0] is the middle of the frame
        absGaze_leftEye = absGaze_leftEye - frame_center
        absGaze_rightEye = absGaze_rightEye - frame_center
        head_pose = head_pose - frame_center

        # Inverse y-direction, so positive = up
        absGaze_leftEye[1] = -absGaze_leftEye[1]
        absGaze_rightEye[1] = -absGaze_rightEye[1]
        head_pose[1] = -head_pose[1]
 
        # Calculate relative gaze by subtracting head pose from absolute gaze
        relGaze_leftEye = absGaze_leftEye - head_pose
        relGaze_rightEye = absGaze_rightEye - head_pose

        # Calculate Point of Gaze (PoG) as the average of the left and right absolute gaze positions
        PoG = (absGaze_leftEye + absGaze_rightEye) / 2

        # Calculate Vergence as the Euclidean distance between the left and right absolute gaze positions
        vergence = np.linalg.norm(absGaze_leftEye - absGaze_rightEye)

    else:
        absGaze_rightEye, absGaze_leftEye, PoG, head_pose, relGaze_rightEye, relGaze_leftEye = (None, None), (None, None), (None, None), (None, None), (None, None), (None, None)
        vergence = None

    return absGaze_rightEye, absGaze_leftEye, PoG, vergence, head_pose, relGaze_rightEye, relGaze_leftEye

