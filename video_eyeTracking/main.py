import os
import json
import argparse
import time
import pickle
from video_eyeTracking.helpers import make_output_directory, extract_patient_info, crop_video_based_on_face_detection
from video_eyeTracking.tracking import extract_face_landmarks

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
    parser.add_argument("--min_detection_confidence", type=float, default=None, help="Proportion of frames in which should detect a face.")
    parser.add_argument("--min_tracking_confidence", type=float, default=None, help="Minimum tracking confidence value")
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = load_config(args.config)

    if args.direct_video_path:
        video_path = args.direct_video_path
    else:
        data_path = config["data_path"]
        raw_video_path = os.path.join(data_path, args.video_relative_path)
    
###################################################################################################################################
    #################### HELPER FUNCTIONS ####################
    # Here is where you can put optional "helper" functions for your specific purpose
    # I want to name my output data a particular way, so here is my code to do that
    if "extract_patient_info" in config.get("helpers_to_run", []):
        patient_id, seizure_num, video_num = extract_patient_info(args.video_relative_path)
        print(f"Patient: {patient_id}, Seizure: {seizure_num}, Clip: {video_num}")
        output_path = os.path.join(config["out_path"],f"{patient_id}_{seizure_num}_CLIP{video_num}")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            print(f"Created output directory: {output_path}")
    else:
        # Default is to make a sub-folder in the same path as input video
        output_path = make_output_directory(raw_video_path)

    ############### Optional: CROPPING VIDEO TO FACE ###############
    # Crop video to face coordinates if not already done
    if config.get("crop_video_based_on_face_detection", False):
        video_path = os.path.join(output_path, 'cropped_fullVideo.avi')
        if not os.path.exists(video_path):
            crop_video_based_on_face_detection(raw_video_path, output_path=output_path)
            print(f"Cropped video to face successfully.")
    else:
        video_path = raw_video_path 
    
###################################################################################################################################
    frame_width, frame_height = extract_face_landmarks(video_path, min_detection_confidence=params["min_detection_confidence"], min_tracking_confidence=params["min_tracking_confidence"], output_path=output_path)
    params.update({
        "frame_width": frame_width,
        "frame_height": frame_height
    })

    save_params_to_pickle(output_path, params)

if __name__ == "__main__":
    main()
