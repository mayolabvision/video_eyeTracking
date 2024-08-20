import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import os
from tqdm import tqdm
import matplotlib


# Use a non-interactive backend for macOS to avoid threading issues
matplotlib.use('Agg')

def plot_gaze(gaze_data, frame_width, frame_height, output_path=None, fade_duration=1.0, fps=30, dpi=200):
    """ 
    Creates a video with a Cartesian plot of absolute gaze (left and right) and head pose.

    Parameters:
    - gaze_data: List of lists containing gaze information.
    - output_path: Path to save the output video.
    - fade_duration: Duration in seconds for the gaze trace to fade out.
    - fps: Frames per second for the output video.
    - dpi: Dots per inch for the output video resolution.
    """
    print('------------ PLOTTING GAZE ------------')
    # Extract gaze x, gaze y, eye positions, and timestamp data
    abs_gaze_left_x, abs_gaze_left_y = [], []
    abs_gaze_right_x, abs_gaze_right_y = [], []
    pog_x, pog_y, verg = [], [], []
    head_pose_x, head_pose_y = [], []
    timestamps = []

    for row in gaze_data:
        timestamps.append(float(row[1]))  # Always record the timestamp
        if row[2] != '' and row[3] != '':  # Ensure gaze_x and gaze_y are not empty
            abs_gaze_right_x.append(float(row[2]))
            abs_gaze_right_y.append(float(row[3]))
            abs_gaze_left_x.append(float(row[4]))
            abs_gaze_left_y.append(float(row[5]))
            pog_x.append(float(row[6]))
            pog_y.append(float(row[7]))
            verg.append(float(row[8]))
            head_pose_x.append(float(row[9]))
            head_pose_y.append(float(row[10]))
        else:
            abs_gaze_right_x.append(None)
            abs_gaze_right_y.append(None)
            abs_gaze_left_x.append(None)
            abs_gaze_left_y.append(None)
            pog_x.append(None)
            pog_y.append(None)
            verg.append(None)
            head_pose_x.append(None)
            head_pose_y.append(None)

    # Setup the figure
    fig, ax = plt.subplots(figsize=(frame_width / 100, frame_height / 100), dpi=dpi)
    ax.set_aspect(aspect='auto')

    x_min = int(np.nanmin(np.array((abs_gaze_right_x + abs_gaze_left_x + head_pose_x),dtype=np.float64)))
    y_min = int(np.nanmin(np.array((abs_gaze_right_y + abs_gaze_left_y + head_pose_y),dtype=np.float64)))
    x_max = int(np.nanmax(np.array((abs_gaze_right_x + abs_gaze_left_x + head_pose_x),dtype=np.float64)))
    y_max = int(np.nanmax(np.array((abs_gaze_right_y + abs_gaze_left_y + head_pose_y),dtype=np.float64)))

    # Plot for gaze position and head pose
    line_gaze_left, = ax.plot([], [], '-', color='magenta', markersize=5, alpha=0.3, label='Absolute Gaze (each eye)')
    line_gaze_right, = ax.plot([], [], '-', color='magenta', markersize=5, alpha=0.3)
    line_head_pose, = ax.plot([], [], '-', color='black', markersize=5, alpha=0.5, label='Head Pose')

    filtered_verg = [v for v in verg if v is not None]

    # Adjust the layout to make room for the legend
    ax.legend(loc='lower center', bbox_to_anchor=(1, 0.5))
    
    # Create the progress bar
    with tqdm(total=len(abs_gaze_left_x), desc="Creating Animation") as pbar:
        # Update function for the animation
        # Update function for the animation
        def update(num, abs_gaze_left_x, abs_gaze_left_y, abs_gaze_right_x, abs_gaze_right_y, head_pose_x, head_pose_y, timestamps):
            ax.clear()
            ax.set_aspect(aspect='auto')
            ax.set_xlim(x_min-1, x_max+1)
            ax.set_ylim(y_min-1, y_max+1)
            ax.set_xlabel('Horizontal Position (pix)')
            ax.set_ylabel('Vertical Position (pix)')

            # Determine how many frames to include in the fading effect
            fade_frames = int(fps * fade_duration)
            start_frame = max(0, num - fade_frames)

            # Gaze data (Left Eye)
            x_data_gaze_left = abs_gaze_left_x[start_frame:num+1]
            y_data_gaze_left = abs_gaze_left_y[start_frame:num+1]

            # Gaze data (Right Eye)
            x_data_gaze_right = abs_gaze_right_x[start_frame:num+1]
            y_data_gaze_right = abs_gaze_right_y[start_frame:num+1]

            # Head Pose
            x_data_head_pose = head_pose_x[start_frame:num+1]
            y_data_head_pose = head_pose_y[start_frame:num+1]

            # Point of Gaze (POG)
            if all(v is not None for v in [pog_x[num], pog_y[num], verg[num]]):
                min_verg = min(filtered_verg)
                max_verg = max(filtered_verg)
                marker_size = 10 - 9 * (verg[num] - min_verg) / (max_verg - min_verg)
                ax.scatter(pog_x[num], pog_y[num], color='magenta', s=marker_size**2, label='Point of Gaze (size = vergence)')

            # Apply the fading effect by setting the alpha values
            if all(v is not None for v in [x_data_gaze_left[-1], y_data_gaze_left[-1]]):
                line_gaze_left = ax.plot(x_data_gaze_left, y_data_gaze_left, 'magenta', alpha=0.3)
                text_gaze_left = [ax.text(x_data_gaze_left[-1], y_data_gaze_left[-1], 'L', color='magenta', fontsize=12, ha='center', va='center')]
            else:
                line_gaze_left = []
                text_gaze_left = []

            if all(v is not None for v in [x_data_gaze_right[-1], y_data_gaze_right[-1]]):
                line_gaze_right = ax.plot(x_data_gaze_right, y_data_gaze_right, 'magenta', alpha=0.3)
                text_gaze_right = [ax.text(x_data_gaze_right[-1], y_data_gaze_right[-1], 'R', color='magenta', fontsize=12, ha='center', va='center')]
            else:
                line_gaze_right = []
                text_gaze_right = []

            if all(v is not None for v in [x_data_head_pose[-1], y_data_head_pose[-1]]):
                line_head_pose = ax.plot(x_data_head_pose, y_data_head_pose, 'black', alpha=0.3)
                text_head_pose = [ax.text(x_data_head_pose[-1], y_data_head_pose[-1], 'h', color='black', fontsize=12, ha='center', va='center')]
            else:
                line_head_pose = []
                text_head_pose = []

            ax.set_xlim(x_min-1, x_max+1)
            ax.set_ylim(y_min-1, y_max+1)
            ax.axhline(0, color='lightgray', linestyle='--')
            ax.axvline(0, color='lightgray', linestyle='--')
            ax.set_xlabel('Horizontal Position (pix)')
            ax.set_ylabel('Vertical Position (pix)')

            # Adjust the layout to make room for the legend
            plt.subplots_adjust(right=0.75)
            
            # Update the common title with the timestamp
            fig.suptitle(f'Time: {round(timestamps[num], 1)} s', fontsize=16)

            # Update the progress bar
            pbar.update(1)

            return line_gaze_left + line_gaze_right + line_head_pose + text_gaze_left + text_gaze_right + text_head_pose

        # Create the animation
        interval = 1000 / fps  # Calculate interval based on fps for real-time effect
        ani = animation.FuncAnimation(fig, update, frames=len(abs_gaze_left_x), fargs=[abs_gaze_left_x, abs_gaze_left_y, abs_gaze_right_x, abs_gaze_right_y, head_pose_x, head_pose_y, timestamps], interval=interval, blit=True)

        # Save the animation as a video file
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if output_path:
            video_file_path = os.path.join(output_path, 'STEP3_plot_gaze.avi')
            ani.save(video_file_path, writer='ffmpeg', fps=fps, dpi=dpi)

    plt.close(fig)

    print(f"Plot video saved at: {video_file_path}")  
