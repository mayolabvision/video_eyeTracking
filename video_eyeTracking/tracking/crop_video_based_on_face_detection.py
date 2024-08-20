import cv2 as cv
import mediapipe as mp
import numpy as np
from tqdm import tqdm  # Importing tqdm for the progress bar
import os

os.environ["OPENCV_FFMPEG_DEBUG"] = "0"

def crop_video_based_on_face_detection(video_path, min_detection_confidence=0.95, percent_padding=[0.2, 0.2], crop_shift=[0, 0], duration=10, repeats=10, output_path=None):
    print('------------ CROPPING VIDEO TO FACE ------------')

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration_frames = int(fps * duration)
    segments = max(1, repeats)
    possible_starts = [i * (num_frames - duration_frames) // (segments - 1) for i in range(segments)]

    best_start_frame = None
    best_face_detection_ratio = 0
    all_bboxes = []

    cv.startWindowThread()
    for start_frame in possible_starts:
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        detected_faces = 0
        total_frames = 0

        with mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence) as face_detection:
            for _ in range(duration_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                total_frames += 1
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)
                if results.detections:
                    detected_faces += 1
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x1 = int(bboxC.xmin * w)
                        y1 = int(bboxC.ymin * h)
                        x2 = x1 + int(bboxC.width * w)
                        y2 = y1 + int(bboxC.height * h)

                        padding_x = int(bboxC.width * w * percent_padding[0])
                        padding_y = int(bboxC.height * h * percent_padding[1])

                        if percent_padding[0] != 0:  # Handle padding_x only if it's not zero
                            if padding_x > 0:
                                x1 = max(0, x1 - padding_x)
                                x2 = min(w, x2 + padding_x)
                            elif padding_x < 0:
                                x1 = max(0, min(w, x1 - padding_x))  # Ensure within bounds
                                x2 = min(w, max(0, x2 + padding_x))  # Ensure within bounds

                        if percent_padding[1] != 0:  # Handle padding_y only if it's not zero
                            if padding_y > 0:
                                y1 = max(0, y1 - padding_y)
                                y2 = min(h, y2 + padding_y)
                            elif padding_y < 0:
                                y1 = max(0, min(h, y1 - padding_y))  # Ensure within bounds
                                y2 = min(h, max(0, y2 + padding_y))  # Ensure within bounds

                        all_bboxes.append([x1, y1, x2, y2])
        cv.waitKey(1)

        # Calculate the face detection ratio for this start frame
        if total_frames > 0:
            face_detection_ratio = detected_faces / total_frames
            if face_detection_ratio > best_face_detection_ratio:
                best_face_detection_ratio = face_detection_ratio
                best_start_frame = start_frame

    cap.release()
    cv.destroyAllWindows()

    if not all_bboxes:
        raise ValueError("Could not detect a face in the video.")

    all_bboxes = np.array(all_bboxes)
    percentiles = np.percentile(all_bboxes, [2.5, 97.5], axis=0)

    filtered_bboxes = all_bboxes[
        (all_bboxes[:, 0] >= percentiles[0, 0]) & (all_bboxes[:, 0] <= percentiles[1, 0]) &
        (all_bboxes[:, 1] >= percentiles[0, 1]) & (all_bboxes[:, 1] <= percentiles[1, 1]) &
        (all_bboxes[:, 2] >= percentiles[0, 2]) & (all_bboxes[:, 2] <= percentiles[1, 2]) &
        (all_bboxes[:, 3] >= percentiles[0, 3]) & (all_bboxes[:, 3] <= percentiles[1, 3]) 
    ]

    if filtered_bboxes.size == 0:
        filtered_bboxes = all_bboxes

    final_bbox = [ 
        max(0, int(np.min(filtered_bboxes[:, 0]) + crop_shift[0])),
        max(0, int(np.min(filtered_bboxes[:, 1]) - crop_shift[1])),
        min(w, int(np.max(filtered_bboxes[:, 2]) + crop_shift[0])),
        min(h, int(np.max(filtered_bboxes[:, 3]) - crop_shift[1]))
    ]

    if output_path and best_start_frame is not None:
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, best_start_frame)

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(output_path, 'STEP1_crop_video_to_face.avi')
        out = cv.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

        cv.startWindowThread()
        for _ in range(duration_frames):
            ret, frame = cap.read()
            if not ret:
                break

            cv.rectangle(frame, (final_bbox[0], final_bbox[1]), (final_bbox[2], final_bbox[3]), (0, 255, 0), 2)
            out.write(frame)
            cv.imshow(frame)
            cv.waitKey(1)

        cv.waitKey(1)
        cap.release()
        out.release()
        cv.destroyAllWindows()

        ##########################
        cap_full = cv.VideoCapture(video_path)
        num_frames = int(cap_full.get(cv.CAP_PROP_FRAME_COUNT))
        out_full = cv.VideoWriter(os.path.join(output_path, 'cropped_fullVideo.avi'), fourcc, fps, (final_bbox[2] - final_bbox[0], final_bbox[3] - final_bbox[1]))
        cv.startWindowThread()
        with tqdm(total=num_frames, desc="Cropping full video") as pbar:
            for _ in range(num_frames):
                ret, frame = cap_full.read()
                if not ret:
                    break
            
                pbar.update(1)
                cropped_frame = frame[final_bbox[1]:final_bbox[3], final_bbox[0]:final_bbox[2]]
                out_full.write(cropped_frame)

        cv.waitKey(1)
        cap_full.release()
        out_full.release()
        cv.destroyAllWindows()

    return min_detection_confidence, percent_padding, crop_shift
