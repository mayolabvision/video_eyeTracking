import cv2
import numpy as np

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def normalize_to_range(value, min_val, max_val, new_min, new_max):
    """Normalize a value to a new range."""
    return new_min + (value - min_val) * (new_max - new_min) / (max_val - min_val)

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
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )   

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 2D pupil locations for both eyes
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    # Transformation between image point to world point
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

    if transformation is not None:  # if estimateAffine3D succeeded
        def calculate_gaze(pupil, Eye_ball_center):
            # Project pupil image point into 3D world point
            pupil_world_cord = transformation @ np.array([[pupil[0], pupil[1], 0, 1]]).T

            # 3D gaze point (10 is an arbitrary value denoting gaze distance)
            S = Eye_ball_center + (pupil_world_cord - Eye_ball_center) * 10

            # Project a 3D gaze direction onto the image plane.
            eye_pupil2D, _ = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)

            # Project 3D head pose into the image plane
            head_pose, _ = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                             rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # Correct gaze for head rotation
            gaze = pupil + (eye_pupil2D[0][0] - pupil) - (head_pose[0][0] - pupil)

            return gaze, eye_pupil2D, head_pose

        # Calculate gaze for left eye
        left_gaze, left_eye_pupil2D, left_head_pose = calculate_gaze(left_pupil, Eye_ball_center_left)

        # Calculate gaze for right eye
        right_gaze, right_eye_pupil2D, right_head_pose = calculate_gaze(right_pupil, Eye_ball_center_right)

        # Compute average gaze direction (gaze_x, gaze_y)
        avg_gaze_x = (left_gaze[0] + right_gaze[0]) / 2
        avg_gaze_y = (left_gaze[1] + right_gaze[1]) / 2

        # Normalize avg_gaze_x and avg_gaze_y to the range -100 to 100
        gaze_x = normalize_to_range(avg_gaze_x, 0, frame.shape[1], -100, 100)
        gaze_y = normalize_to_range(avg_gaze_y, 0, frame.shape[0], -100, 100)

        # Compute the deviation of the eyes from where the head is pointing separately for each eye
        pos_x_left = normalize_to_range(left_gaze[0] - center[0], -center[0], center[0], -100, 100)
        pos_y_left = normalize_to_range(left_gaze[1] - center[1], -center[1], center[1], -100, 100)

        pos_x_right = normalize_to_range(right_gaze[0] - center[0], -center[0], center[0], -100, 100)
        pos_y_right = normalize_to_range(right_gaze[1] - center[1], -center[1], center[1], -100, 100)

        # Calculate proximity based on the difference in gaze vectors
        gaze_difference = np.linalg.norm(np.array(right_gaze) - np.array(left_gaze))
        proximity = normalize_to_range(gaze_difference, 0, frame.shape[1], 100, 0)  # Reverse the range

        # Draw gaze lines into the screen for both eyes
        cv2.line(frame, (int(left_pupil[0]), int(left_pupil[1])), (int(left_gaze[0]), int(left_gaze[1])), (255, 0, 255), 2)
        cv2.line(frame, (int(right_pupil[0]), int(right_pupil[1])), (int(right_gaze[0]), int(right_gaze[1])), (255, 0, 255), 2)

        '''
        print("Gaze X, Gaze Y:", gaze_x, gaze_y)
        print("Position X (Left), Position Y (Left):", pos_x_left, pos_y_left)
        print("Position X (Right), Position Y (Right):", pos_x_right, pos_y_right)
        print("Proximity:", proximity)
        '''

    else:
        gaze_x, gaze_y, pos_x_left, pos_y_left, pos_x_right, pos_y_right, proximity = None, None, None, None, None, None, None

    return gaze_x, gaze_y, pos_x_left, pos_y_left, pos_x_right, pos_y_right, proximity
