import os
import json
import argparse
import time
from face_gazeTracking.calibration import crop_video_based_on_face_detection, find_optimal_confidences, calibrate_blink_threshold
from face_gazeTracking.utils import extract_video_info
from face_gazeTracking.output import export_video

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
        video_path = os.path.join(data_path, args.video_relative_path)

    output_path = config["output_path"]
    
    # Step 1: Getting video parameters (based on specific project)
    print('------------ EXTRACTING PATIENT INFO ------------')
    if "extract_video_info" in config.get("helpers_to_run", []):
        patient_id, seizure_num, video_num = extract_video_info(args.video_relative_path)
        print(f"Patient ID: {patient_id}, Seizure Number: {seizure_num}, Video Number: {video_num}")
   
    # Step 2: Cropping video to face coordinates
    print('------------ CROPPING VIDEO TO FACE ------------')
    if config.get("crop_video_based_on_face_detection", False):
        face_crop_coords = crop_video_based_on_face_detection(video_path, showVideo=1)
        print(f"Crop coordinates: {face_crop_coords}")
    else:
        face_crop_coords = None
    time.sleep(10)


    print(asdlkjfdask)
    # Step 3: 
    print('------------ OPTIMIZING CONFIDENCE THRESHOLDS ------------')
    min_detection_confidence, min_tracking_confidence = find_optimal_confidences(video_path, face_crop_coords, showVideo=1)
    print(f"Detection Confidence: {min_detection_confidence}, Tracking Confidence: {min_tracking_confidence}")

    time.sleep(3)
    # Step 4: 
    print('------------ CALIBRATING BLINK THRESHOLD ------------')
    blink_threshold = calibrate_blink_threshold(video_path, face_crop_coords, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence, showVideo=1)
    print(f"Calculated blink threshold: {blink_threshold}")

    # Step 1: Preprocessing
    #preprocessed_data = preprocess_step1(video_path)
    #preprocessed_data = preprocess_step2(preprocessed_data)
    
    # Export the processed video
    #output_file_path = os.path.join(output_path, os.path.basename(video_path))
    #export_video(processed_data, output_file_path)

if __name__ == "__main__":
    main()
