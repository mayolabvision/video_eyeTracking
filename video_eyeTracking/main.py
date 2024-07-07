import os
import json
import argparse
import time
import pickle
from video_eyeTracking.calibration import crop_video_based_on_face_detection, find_optimal_confidences, calibrate_blink_threshold
from video_eyeTracking.utils import make_output_directory, extract_video_info, save_params_to_pickle, load_params_from_pickle
from video_eyeTracking.output import export_video

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a specific video for gaze tracking.")
    parser.add_argument("video_relative_path", type=str, help="Relative path to the specific video file within the data directory.")
    parser.add_argument("--config", type=str, default=None, help="Path to an alternative config file.")
    parser.add_argument("--direct_video_path", type=str, default=None, help="Direct path to a video file, bypassing config data_path.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = load_config(args.config)

    if args.direct_video_path:
        video_path = args.direct_video_path
    else:
        data_path = config["data_path"]
        raw_video_path = os.path.join(data_path, args.video_relative_path)
    
    output_path = make_output_directory(raw_video_path)

    # Check if params.pickle exists and load it if it does
    params_pickle_path = os.path.join(output_path, 'params.pickle')
    if os.path.exists(params_pickle_path):
        params = load_params_from_pickle(params_pickle_path)
        print("Loaded parameters from params.pickle")
    else:
        params = {}

    # Extract video parameters if not already done
    print('------------ EXTRACTING PATIENT INFO ------------')
    if "extract_video_info" in config.get("helpers_to_run", []) and not all(k in params for k in ("patient_id", "seizure_num", "video_num")):
        patient_id, seizure_num, video_num = extract_video_info(args.video_relative_path)
        params.update({
            "patient_id": patient_id,
            "seizure_num": seizure_num,
            "video_num": video_num
        })
        print(f"Patient ID: {patient_id}, Seizure Number: {seizure_num}, Video Number: {video_num}")

    # Crop video to face coordinates if not already done
    print('------------ CROPPING VIDEO TO FACE ------------')
    # Initialize face_crop_coords based on config and params
    if config.get("crop_video_based_on_face_detection", False):
        if "face_crop_coords" not in params:
            crop_video_based_on_face_detection(raw_video_path, output_path=output_path)
            params["face_crop_coords"] = 1
            print(f"Cropped video to face successfully.")
        video_path = os.path.join(output_path, 'cropped_fullVideo.avi')
    else:
        video_path = raw_video_path 

    # Optimize confidence thresholds if not already done
    print('------------ OPTIMIZING CONFIDENCE THRESHOLDS ------------')
    if "min_detection_confidence" not in params or "min_tracking_confidence" not in params:
        min_detection_confidence, min_tracking_confidence = find_optimal_confidences(video_path, output_path=output_path)
        params.update({
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence
        })
        print(f"Detection Confidence: {min_detection_confidence}, Tracking Confidence: {min_tracking_confidence}")

    time.sleep(3)

    # Calibrate blink threshold if not already done
    print('------------ CALIBRATING BLINK THRESHOLD ------------')
    if "blink_threshold" not in params:
        blink_threshold = calibrate_blink_threshold(video_path, min_detection_confidence=params["min_detection_confidence"], min_tracking_confidence=params["min_tracking_confidence"], output_path=output_path)
        params["blink_threshold"] = blink_threshold
        print(f"Calculated blink threshold: {blink_threshold}")

    # Save parameters to params.pickle
    save_params_to_pickle(output_path, params)

if __name__ == "__main__":
    main()
