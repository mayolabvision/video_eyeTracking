import csv

def load_gaze_data_from_csv(csv_file_path):
    """
    Loads gaze_data from the specified CSV file, removing the headers.
    """
    gaze_data = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            # Convert numeric values and handle empty strings
            formatted_row = [
                int(row[0]),  # Frame Number
                float(row[1])  # Timestamp
            ]
            formatted_row.extend(
                [float(value) if value != '' else '' for value in row[2:]]
            )
            gaze_data.append(formatted_row)
    return gaze_data
