# video_eyeTracking/helpers/prompt_user_to_overwrite.py
def prompt_user_to_overwrite(message="File already exists. Do you want to overwrite it? (y/n): "):
    user_input = input(message)
    return user_input.strip().lower() == 'y'
