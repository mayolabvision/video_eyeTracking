import os
import csv
import json
import argparse
import glob
import time
import threading
from video_eyeTracking.helpers import *
from video_eyeTracking.tracking import *
from video_eyeTracking.plotting import *

def process_all_videos_in_data_path(config, args):
    data_path = config["data_path"]
    if args.video_relative_path:
        # Find all .avi files in subfolders of data_path
        video_files = glob.glob(os.path.join(data_path, args.video_relative_path, '**', '*.avi'), recursive=True)
    else:
        video_files = glob.glob(os.path.join(data_path, '**', '*.avi'), recursive=True)

    # Iterate through all found video files
    for video_file in video_files:
        # Convert the absolute video path to a relative path
        relative_video_path = os.path.relpath(video_file, data_path)

        print(f"Processing video: {video_file}")

        # Call the main processing function with the specific video file
        main(video_relative_path=relative_video_path)

def main(video_relative_path=None):
    print('\n\n=========================================================================================')
    print('=========================================================================================\n')
    args = parse_arguments()
    config = load_config(args.config)
    params = {}

    if args.direct_video_path:
        raw_video_path = args.direct_video_path
    elif video_relative_path:
        data_path = config["data_path"]
        # Check if video_relative_path ends with .avi
        if video_relative_path.endswith('.avi'):
            raw_video_path = os.path.join(data_path, video_relative_path)
        else:
            process_all_videos_in_data_path(config, args)
            return
    else:
        data_path = config["data_path"]
        process_all_videos_in_data_path(data_path, args)
        return

    if config["out_path"] is not None:
        output_path = os.path.join(config["out_path"], video_relative_path)
        output_path = os.path.splitext(output_path)[0]
    else:
        output_path = os.path.dirname(os.path.splitext(raw_video_path)[0])

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Created output directory: {output_path}")

#################################################################################################################################
    #################### HELPER FUNCTIONS ####################
    # Here is where you can put optional "helper" functions for your specific purpose
    # I want to name my output data a particular way, so here is my code to do that
    if "extract_patient_info" in config.get("helpers_to_run", []):
        patient_id, seizure_num, video_num = extract_patient_info(video_relative_path)
        print(f"Patient: {patient_id}, Seizure: {seizure_num}, Clip: {video_num}")

    ############### Optional: CROPPING VIDEO TO FACE ###############
    # Crop video to face coordinates if not already done
    if config.get("crop_video_based_on_face_detection", False):
        video_path = os.path.join(output_path, 'cropped_fullVideo.avi')
        if os.path.exists(video_path):
            overwrite = [False]

            def user_prompt():
                overwrite[0] = prompt_user_to_overwrite("cropped_fullVideo.avi already exists. Do you want to overwrite it? (y/n): ") 
            # Start a thread for user input
            input_thread = threading.Thread(target=user_prompt)
            input_thread.start()

            # Wait for the user input with a timeout
            input_thread.join(timeout=5)

            # If the thread is still alive (i.e., no input), assume 'y'
            if input_thread.is_alive():
                overwrite[0] = False

            if overwrite[0]:
                if args.min_crop_confidence and args.crop_padding and args.crop_shift:
                    crop_confidence, crop_padding_scale, crop_shift = crop_video_based_on_face_detection(raw_video_path, min_detection_confidence=args.min_crop_confidence, percent_padding=args.crop_padding, crop_shift=args.crop_shift, output_path=output_path)
                else:
                    crop_confidence, crop_padding_scale, crop_shift = crop_video_based_on_face_detection(raw_video_path, output_path=output_path)
                params['crop_confidence'] = crop_confidence
                params['crop_padding_scale'] = crop_padding_scale
                params['crop_shift'] = crop_shift
                save_params(params, output_path)
                print(f"STEP 1 COMPLETE. Cropped video to face successfully.")
            else:
                print(f"STEP 1 SKIPPED. Not overwritting cropped video.")            
        else:
            if args.min_crop_confidence and args.crop_padding and args.crop_shift:
                crop_confidence, crop_padding_scale, crop_shift = crop_video_based_on_face_detection(raw_video_path, min_detection_confidence=args.min_crop_confidence, percent_padding=args.crop_padding, crop_shift=args.crop_shift, output_path=output_path)
            else:
                crop_confidence, crop_padding_scale, crop_shift = crop_video_based_on_face_detection(raw_video_path, output_path=output_path)
            params['crop_confidence'] = crop_confidence
            params['crop_padding_scale'] = crop_padding_scale
            params['crop_shift'] = crop_shift
            save_params(params, output_path)
            print(f"STEP 1 COMPLETE. Cropped video to face successfully.")
    else:
        video_path = raw_video_path # return original video as video to process in next steps
    
    if args.step is not None:
        if args.step == 1:
            print("STEP 1 completed. Exiting as per the --step argument.")
            print('=========================================================================================')
            return  # Exit the main() function
        elif args.step == 0 or args.step > 3:
            raise ValueError("Invalid step value. Please use a step value between 1 and 3.")

#################################################################################################################################
    gaze_file_path = os.path.join(output_path, 'EYE_GAZE_LOGS.csv')
    if os.path.exists(gaze_file_path):
        overwrite = [False]

        def user_prompt():
            overwrite[0] = prompt_user_to_overwrite("EYE_GAZE_LOGS.csv already exists. Do you want to overwrite it? (y/n): ") 

        # Start a thread for user input
        input_thread = threading.Thread(target=user_prompt)
        input_thread.start()

        # Wait for the user input with a timeout
        input_thread.join(timeout=5)

        # If the thread is still alive (i.e., no input), assume 'y'
        if input_thread.is_alive():
            overwrite[0] = True

        if overwrite[0]:
            if args.min_detection_confidence and args.min_tracking_confidence:
                min_detection_confidence, min_tracking_confidence = extract_face_landmarks(video_path, min_detection_confidence=args.min_detection_confidence, min_tracking_confidence=args.min_tracking_confidence, output_path=output_path)
            else:
                min_detection_confidence, min_tracking_confidence = extract_face_landmarks(video_path, output_path=output_path)
            params['min_detection_confidence'] = min_detection_confidence
            params['min_tracking_confidence'] = min_tracking_confidence
            save_params(params, output_path)
            redo_plot_gaze = True
            print("STEP 2 COMPLETE. Gaze and face landmarks extracted from video.")
        else:
            print("STEP 3 SKIPPED. Not overwriting gaze and face landmarks.")
            redo_plot_gaze = False
    else:
        if args.min_detection_confidence and args.min_tracking_confidence:
            min_detection_confidence, min_tracking_confidence = extract_face_landmarks(video_path, min_detection_confidence=args.min_detection_confidence, min_tracking_confidence=args.min_tracking_confidence, output_path=output_path)
        else:
            min_detection_confidence, min_tracking_confidence = extract_face_landmarks(video_path, output_path=output_path)
        params['min_detection_confidence'] = min_detection_confidence
        params['min_tracking_confidence'] = min_tracking_confidence
        save_params(params, output_path)
        redo_plot_gaze = True
        print("STEP 2 COMPLETE. Gaze and face landmarks extracted from video.")

    if args.step is not None and args.step ==2:
        print("Step 2 completed. Exiting as per the --step argument.")
        print('=========================================================================================')
        return  # Exit the main() function

#################################################################################################################################
    # Now that gaze_data is loaded, proceed with further processing, e.g., plotting
    gaze_data = load_gaze_data_from_csv(gaze_file_path)
    
    # Check if STEP3_plot_gaze.avi exists and prompt to overwrite
    plot_file_path = os.path.join(output_path, "STEP3_plot_gaze.avi")
    if redo_plot_gaze or not os.path.exists(plot_file_path):
        plot_gaze(gaze_data, output_path=output_path)
        print("STEP 3 COMPLETE. Gaze and eye position plotted.")
    else:
        overwrite_plot = [False] 
        def plot_user_prompt():
            overwrite_plot[0] = prompt_user_to_overwrite("STEP3_plot_gaze.avi already exists. Do you want to overwrite it? (y/n): ")

        # Start a thread for user input
        input_thread = threading.Thread(target=plot_user_prompt)
        input_thread.start()

        # Wait for the user input with a timeout
        input_thread.join(timeout=5)

        # If the thread is still alive (i.e., no input), assume 'y'
        if input_thread.is_alive():
            overwrite_plot[0] = True

        if overwrite_plot[0]:
            plot_gaze(gaze_data, output_path=output_path)
            print("STEP 3 COMPLETE. Gaze and eye position plotted.")
        else:
            print("STEP 3 SKIPPED. Gaze and eye position already plotted.")
    
#################################################################################################################################
    merge_videos_side_by_side(output_path)

    print("COMPLETED ALL STEPS.")
    print('\n=========================================================================================')
    print('=========================================================================================')
#################################################################################################################################
if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)

    if args.video_relative_path or args.direct_video_path:
        main(args.video_relative_path)  # Process a single video
    else:
        process_all_videos_in_data_path(config, args)  # Process all videos in data_path
