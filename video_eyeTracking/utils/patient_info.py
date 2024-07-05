import re

def extract_video_info(video_relative_path):
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

# Example usage:
# video_path = "DF/SZ1/26d70c16-5ad8-4e4d-847c-be4cfc5b2ea1/26d70c16-5ad8-4e4d-847c-be4cfc5b2ea1_0002 (1).avi"
# patient_id, seizure_num, video_num = extract_video_info(video_path)
# print(patient_id, seizure_num, video_num)  # Outputs: DF SZ1 1
