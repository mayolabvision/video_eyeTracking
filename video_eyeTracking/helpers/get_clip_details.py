import pandas as pd
import ast

# Load the CSV file once to avoid reloading it each time the function is called
clip_details_path = '/Users/kendranoneman/OneDrive/EyeMovement/clip_times.csv'
df = pd.read_csv(clip_details_path)

def get_clip_details(patient_ID, seizure_ID):
    """ 
    Finds and returns clip details for the specified patient and seizure as a list.

    Parameters:
        patient_ID (str or int): ID of the patient.
        seizure_ID (str or int): ID of the seizure.

    Returns:
        list: Clip details.
    """

    # Filter the DataFrame for the specific patient and seizure
    df_row = df.loc[(df['patient_ID'] == patient_ID) & (df['seizure_ID'] == seizure_ID)].copy()

    # Ensure seizure_ID is converted to an integer
    df_row['seizure_ID'] = df_row['seizure_ID'].astype(int)

    # Ensure closeCam_order and wideCam_order columns are converted to lists
    df_row['closeCam_order'] = df_row['closeCam_order'].apply(ast.literal_eval)
    df_row['wideCam_order'] = df_row['wideCam_order'].apply(ast.literal_eval)

    if not df_row.empty:
        # Convert the row to a list directly from the specified columns
        details = df_row[['patient_ID', 'seizure_ID', 'recording_date', 'clip_start', 
                          'seizure_start', 'seizure_end', 'seizure_duration', 
                          'clip_end', 'closeCam_order', 'wideCam_order']].values.flatten().tolist()
        
        return details
    else:
        return None
