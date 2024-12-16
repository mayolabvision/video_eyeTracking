import os
import csv
import json
import cv2 as cv
import argparse
import glob
import time
import threading
import signal
from video_eyeTracking.helpers import *
#from video_eyeTracking.tracking import *
#from video_eyeTracking.plotting import *

def main(seizure_ID):
    args = parse_arguments()
    config = load_config(args.config)
    
    data_path = config["data_path"]
    full_path = compile_path(data_path, args.patient_ID, args.seizure_ID)
    output_path = os.path.join(full_path, "tracking")
    video_path = os.path.join(output_path, "close_clips.avi")

    print(video_path)

    params = load_params(output_path)
    clip_details = get_clip_details(args.patient_ID, args.seizure_ID)
    params['clip_details'] = clip_details

    # Write the updated parameters back to params.json
    #with open(params_file, 'w') as f:
    #    json.dump(params, f, indent=4)

    concat_videos(full_path, clip_details)
    
    print(params)
    print(asdlkfjadslk)
    
    gaze_file_path = os.path.join(output_path, 'EYE_GAZE_LOGS.csv')
    
    min_detection_confidence, min_tracking_confidence, frame_width, frame_height = extract_face_landmarks(video_path, min_detection_confidence=args.min_detection_confidence, min_tracking_confidence=args.min_tracking_confidence, output_path=output_path)
            
    params['min_detection_confidence'] = min_detection_confidence
    params['min_tracking_confidence'] = min_tracking_confidence
    params['frame_width'] = frame_width
    params['frame_height'] = frame_height
    save_params(params, output_path)

    gaze_data = load_gaze_data_from_csv(gaze_file_path)

    if 'frame_width' not in params:
        cap = cv.VideoCapture(video_path)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        params['frame_width'] = frame_width
        params['frame_height'] = frame_height
        cap.release()
        cv.waitKey(1)

    # Check if STEP3_plot_gaze.avi exists and prompt to overwrite
    plot_file_path = os.path.join(output_path, "STEP3_plot_gaze.avi")
    if redo_plot_gaze or not os.path.exists(plot_file_path):
        plot_gaze(gaze_data, params['frame_width'], params['frame_height'], output_path=output_path)
        merge_videos_side_by_side(output_path)
        print("STEP 3 COMPLETE. Gaze and eye position plotted.")
    else:
        overwrite = prompt_user_to_overwrite("STEP3_plot_gaze.avi already exists. Do you want to overwrite it? (y/n): ", default=True)
        
        if overwrite:
            plot_gaze(gaze_data, params['frame_width'], params['frame_height'], output_path=output_path)
            merge_videos_side_by_side(output_path)
            print("STEP 3 COMPLETE. Gaze and eye position plotted.")
        else:
            print("STEP 3 SKIPPED. Gaze and eye position already plotted.")
            pass

#################################################################################################################################
    print("COMPLETED ALL STEPS.")
    print('\n=========================================================================================')
    print('=========================================================================================')

    # Clean up any files ending with '-Kendra’s MacBook.avi'
    for file in os.listdir(output_path):
        if "MacBook" in file:                                                                                   
            file_path = os.path.join(output_path, file)
            try:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Failed to remove file: {file_path}. Reason: {e}")
#################################################################################################################################
if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)

    main(args.seizure_ID)  # Process a single video
