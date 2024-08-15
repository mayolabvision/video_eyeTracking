import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from tqdm import tqdm
import matplotlib

# Use a non-interactive backend for macOS to avoid threading issues
matplotlib.use('Agg')

def plot_gaze(gaze_data, output_path=None, fade_duration=1.0, fps=30, dpi=100):
    """ 
    Creates a video with a side-by-side view of gaze position on polar coordinates 
    and eye positions on Cartesian coordinates.

    Parameters:
    - gaze_data: List of lists containing gaze information.
    - output_path: Path to save the output video.
    - fade_duration: Duration in seconds for the gaze trace to fade out.
    - fps: Frames per second for the output video.
    - dpi: Dots per inch for the output video resolution.
    """
    print('------------ PLOTTING GAZE ------------')
    # Extract gaze x, gaze y, eye positions, and timestamp data
    gaze_x, gaze_y = [], []
    pos_x_left, pos_y_left = [], []
    pos_x_right, pos_y_right = [], []
    gaze_distance = []  # Added to store the gaze distance for marker size
    timestamps = []

    for row in gaze_data:
        timestamps.append(float(row[1]))  # Always record the timestamp
        if row[2] != '' and row[3] != '':  # Ensure gaze_x and gaze_y are not empty
            gaze_x.append(float(row[2]))
            gaze_y.append(float(row[3]))
            pos_x_left.append(float(row[4]))
            pos_y_left.append(float(row[5]))
            pos_x_right.append(float(row[6]))
            pos_y_right.append(float(row[7]))
            gaze_distance.append(float(row[8]))  # Get the gaze distance for marker size
        else:
            gaze_x.append(None)
            gaze_y.append(None)
            pos_x_left.append(None)
            pos_y_left.append(None)
            pos_x_right.append(None)
            pos_y_right.append(None)
            gaze_distance.append(None)

    # Convert to polar coordinates
    theta_gaze = np.array([np.arctan2(y, x) if x is not None and y is not None else None for x, y in zip(gaze_x, gaze_y)])
    r_gaze = np.array([np.sqrt(x**2 + y**2) if x is not None and y is not None else None for x, y in zip(gaze_x, gaze_y)])

    # Setup the figure with one polar and one Cartesian subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
    ax1 = plt.subplot(121, projection='polar')
    ax2 = plt.subplot(122)

    ax1.set_ylim(0, 100)
    ax2.set_xlim(-100, 100)
    ax2.set_ylim(-100, 100)
    ax2.axhline(0, color='lightgray', linestyle='--')
    ax2.axvline(0, color='lightgray', linestyle='--')
    ax2.set_xlabel('Horizontal Eye Position')
    ax2.set_ylabel('Vertical Eye Position')

    # Plot for gaze position (polar)
    line_gaze, = ax1.plot([], [], 'm-', label='Gaze Position', alpha=0.5)
    scatter_gaze = ax1.scatter([], [], s=[], c='m', label='(Marker Size = Distance)')

    # Plot for eye positions (Cartesian)
    line_left, = ax2.plot([], [], 'b-', markersize=5, label='Left Eye Position', alpha=0.5)
    line_right, = ax2.plot([], [], 'r-', markersize=5, label='Right Eye Position', alpha=0.5)

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    # Create the progress bar
    with tqdm(total=len(theta_gaze), desc="Creating Animation") as pbar:
        # Update function for the animation
        def update(num, theta_gaze, r_gaze, pos_x_left, pos_y_left, pos_x_right, pos_y_right, gaze_distance, timestamps):
            # Determine how many frames to include in the fading effect
            fade_frames = int(fps * fade_duration)
            start_frame = max(0, num - fade_frames)

            # Gaze data
            x_data_gaze = theta_gaze[start_frame:num+1]
            y_data_gaze = r_gaze[start_frame:num+1]
            size_data_gaze = np.array([d * 0.1 if d is not None else 0 for d in gaze_distance[start_frame:num+1]])  # Reduce marker size
            alphas_gaze = np.linspace(0, 1, len(x_data_gaze))

            # Eye positions
            x_data_left = pos_x_left[start_frame:num+1]
            y_data_left = pos_y_left[start_frame:num+1]
            alphas_left = np.linspace(0, 1, len(x_data_left))

            x_data_right = pos_x_right[start_frame:num+1]
            y_data_right = pos_y_right[start_frame:num+1]
            alphas_right = np.linspace(0, 1, len(x_data_right))

            # Apply the fading effect by setting the alpha values
            line_gaze.set_data(x_data_gaze, y_data_gaze)
            line_gaze.set_alpha(alphas_gaze[-1])
            scatter_gaze.set_offsets(np.c_[x_data_gaze, y_data_gaze])
            scatter_gaze.set_sizes(size_data_gaze)
            scatter_gaze.set_alpha(alphas_gaze[-1])

            line_left.set_data(x_data_left, y_data_left)
            line_left.set_alpha(alphas_left[-1])

            line_right.set_data(x_data_right, y_data_right)
            line_right.set_alpha(alphas_right[-1])

            # Update the common title with the timestamp
            fig.suptitle(f'Time: {round(timestamps[num], 1)} s', fontsize=16)

            # Update the progress bar
            pbar.update(1)

            return line_gaze, scatter_gaze, line_left, line_right

        # Create the animation
        interval = 1000 / fps  # Calculate interval based on fps for real-time effect
        ani = animation.FuncAnimation(fig, update, frames=len(theta_gaze), fargs=[theta_gaze, r_gaze, pos_x_left, pos_y_left, pos_x_right, pos_y_right, gaze_distance, timestamps], interval=interval, blit=True)

        # Save the animation as a video file
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if output_path:
            video_file_path = os.path.join(output_path, 'STEP3_plot_gaze.avi')
            ani.save(video_file_path, writer='ffmpeg', fps=fps, dpi=dpi)

    plt.close(fig)

    print(f"Plot video saved at: {video_file_path}")    
