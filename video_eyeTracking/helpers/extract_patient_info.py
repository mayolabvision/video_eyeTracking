import re
import os

def extract_patient_info(video_relative_path):
    print('------------ EXTRACTING PATIENT INFO ------------')
    # Split the path and extract PATIENT_ID and SEIZURE_NUM
    parts = video_relative_path.split('/')
    patient_id = parts[0]
    seizure_num = parts[1]
    
    # Use regex to extract VIDEO_NUM
    match = re.search(r'\((\d+)\)\.avi$', parts[-1])
    if match:
        video_num = int(match.group(1))
    else:
        video_num = None
    
    return patient_id, seizure_num, video_num 
