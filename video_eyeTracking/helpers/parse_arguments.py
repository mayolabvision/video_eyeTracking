# video_eyeTracking/helpers/parse_arguments.py
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a specific video for gaze tracking.")
    parser.add_argument("video_relative_path", type=str, nargs='?', default=None, help="Relative path to the specific video file within the data directory.")
    parser.add_argument("--config", type=str, default=None, help="Path to an alternative config file.")
    parser.add_argument("--step", type=int, default=None, help="Last step you wish to complete, so if you put 3 it'll run steps 1-3.")
    parser.add_argument("--direct_video_path", type=str, default=None, help="Direct path to a video file, bypassing config data_path.")
    parser.add_argument("--crop_padding", type=float, nargs=2, default=None, help="Padding proportions for cropping, specified as two floats (e.g., 0.2 0.2).")
    parser.add_argument("--crop_shift", type=float, nargs=2, default=None, help="Padding proportions for cropping, specified as two floats (e.g., 0.2 0.2).")
    parser.add_argument("--min_crop_confidence", type=float, default=None, help="Proportion of frames in which should detect a face.")
    parser.add_argument("--min_detection_confidence", type=float, default=None, help="Proportion of frames in which should detect a face.")
    parser.add_argument("--min_tracking_confidence", type=float, default=None, help="Minimum tracking confidence value")
    return parser.parse_args()
