[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

[![bg][banner]][website]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/mayolabvision/video_eyeTracking">
    <img src="images/logo.png" alt="Logo" width="420" height="400">
  </a>

<h3 align="center">Offline Gaze Tracking from Video: Analyzing Eye Movements and Head Pose</h3>

  <p align="center">
    This project focuses on offline gaze tracking from video data, where the input is a video of a subject’s face. The code processes the video to analyze eye movements and head pose, generating output videos that visually demonstrate each processing step. Additionally, two CSV files are produced: one containing the pixel coordinates of facial landmarks for each frame, and another with the calculated gaze and head pose vectors. In this version, the gaze vectors are in pixel units because the necessary calibration information—such as camera focal length, subject distance, and face width—is not yet known, preventing the calculation of gaze in degrees of visual angle.
    <br />
    <a href="https://github.com/mayolabvision/video_eyeTracking">View Demo</a>
    ·
    <a href="https://github.com/mayolabvision/video_eyeTracking/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/mayolabvision/video_eyeTracking/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ABOUT THE PROJECT -->
## About The Project

### Built With

* [![Python][Python.com]][Python-url]
* [![MediaPipe][MediaPipe.com]][MediaPipe-url]
* [![OpenCV][OpenCV.com]][OpenCV-url]

There are numerous repositories available that track gaze in real-time using webcams, but I needed a solution that could perform this analysis on multiple videos offline. This repository is designed to analyze pre-recorded videos by determining where the face of the subject is located, cropping the video to focus solely on that face, and then extracting the facial landmarks from each frame. These landmarks are saved in a file called FACE_LANDMARKS_LOGS.csv and are crucial for estimating gaze direction.

The repository calculates the absolute and relative gaze for each eye, the head pose, Point of Gaze (PoG), and vergence based on the relevant facial landmarks. Since we do not have access to the camera’s focal length, the width of the subject’s head, or the distance between the camera and the subject, all calculations are done in units of pixels.

The output file EYE_GAZE_LOGS.csv includes the following columns:
* *Frame Number:* The index of the frame within the video, used to track the frame order.
* *Time Aligned to Video Start (ms):* The timestamp of each frame in milliseconds, aligned to the start of the video. This helps in synchronizing gaze data with the video timeline.
* *Absolute Gaze (right, x):* The x-coordinate of the right eye’s gaze direction in pixels. This value represents where the right eye is looking on the screen. If the value is 0, the gaze is directed towards the center of the frame; positive values indicate a gaze directed to the right of center, negative values to the left, and if the |value| is greater than half the frame width, the subject is looking “out of frame".
* *Absolute Gaze (right, y):* The y-coordinate of the right eye’s gaze direction in pixels. If the value is 0, the gaze is directed towards the center of the frame; positive values indicate a gaze directed to the upwards of center, negative values downwards, and if the |value| is greater than half the frame width, the subject is looking “out of frame".
* *Absolute Gaze (left, x):* The x-coordinate of the left eye’s gaze direction in pixels. See above.
* *Absolute Gaze (left, y):* The y-coordinate of the left eye’s gaze direction in pixels. See above.
* *Point of Gaze (x):* The x-coordinate of the average absolute gaze direction between the two eyes, in pixels. 
* *Point of Gaze (y):* The y-coordinate of the average absolute gaze direction between the two eyes, in pixels. 
* *Vergence Distance:* The distance between the gaze directions of the left and right eyes, indicating depth perception. A smaller distance indicates that the eyes are converging, suggesting that the subject is focusing on a closer object. A larger distance indicates a more parallel gaze, suggesting that the subject is focusing on a farther object.
* *Head Pose (x):* The x-coordinate of the head’s orientation in pixels, representing the horizontal position of the nose or face center. If the value is 0, the head is oriented towards the center of the frame; positive values indicate the head is oriented to the right of center, negative values to the left, and if the |value| is greater than half the frame width, the subject's head is oriented “out of frame".
* *Head Pose (y):* The y-coordinate of the head’s orientation in pixels, representing the horizontal position of the nose or face center. If the value is 0, the head is oriented towards the center of the frame; positive values indicate the head is oriented to the right of center, negative values to the left, and if the |value| is greater than half the frame width, the subject's head is oriented “out of frame".
* *Relative Gaze (right, x):* The difference between the right eye’s absolute gaze and the head pose in pixels, representing how far the right eye’s gaze direction deviates from the head’s orientation on the horizontal axis. A positive value indicates that the gaze is directed to the right relative to the head, and a negative value indicates a leftward direction.
* *Relative Gaze (right, y):* The difference between the right eye’s absolute gaze and the head pose in pixels, representing how far the right eye’s gaze direction deviates from the head’s orientation on the vertical axis. A positive value indicates that the gaze is directed downward relative to the head, and a negative value indicates an upward direction.
* *Relative Gaze (left, x):* The difference between the left eye’s absolute gaze and the head pose, in pixels. See above.
* *Relative Gaze (left, y):* The y-coordinate difference between the left eye’s absolute gaze and the head pose, in pixels. See above.

At the end of the process, a video called SUMMARY_VID.avi is generated. This video shows the subject with the estimated absolute gaze position for each eye and head pose represented as vectors (in the left subpanel), and these values are plotted on Cartesian axes.

# Gaze Function Output Description

The `gaze` function calculates several outputs related to eye and head movement. Below is a detailed description of each output, including how it is calculated, the landmarks used, and the range of values.

## Outputs

### 1. Absolute Gaze (`absGaze_rightEye` and `absGaze_leftEye`)
- **Description**: 
  - Represents the 2D gaze direction of each eye (left and right) relative to the image frame.
  - Calculated independently for each eye.
- **Calculation**:
  - Uses the **pupil position**:
    - `landmark[468]` for the left eye.
    - `landmark[473]` for the right eye.
  - Projects the pupil position into 3D world coordinates using the transformation matrix.
  - Calculates the normalized 3D gaze direction from the **eyeball center**:
    - `Eye_ball_center_left` for the left eye.
    - `Eye_ball_center_right` for the right eye.
  - Projects the 3D gaze direction back into 2D image space.
- **Range**:
  - Values are 2D coordinates relative to the **frame center**:
    - Positive X values indicate movement to the right.
    - Positive Y values indicate upward movement (Y-axis is inverted).
  - The exact range depends on the frame resolution.
- **Visualization**:
  - A line is drawn on the frame from the pupil position to the calculated gaze point.

---

### 2. Relative Gaze (`relGaze_rightEye` and `relGaze_leftEye`)
- **Description**:
  - Represents the gaze direction of each eye relative to the **head pose**. 
  - Removes the effect of head movement from the gaze calculation.
- **Calculation**:
  - Subtract the **head pose vector** from the **absolute gaze** for each eye.
- **Range**:
  - Same as **absolute gaze**, but adjusted to account for head movement.
- **Visualization**:
  - No direct visualization in the code, but these values represent the gaze adjusted for head pose.

---

### 3. Point of Gaze (PoG)
- **Description**:
  - Represents the average gaze position of both eyes in 2D space.
  - Indicates where both eyes are collectively looking on the image plane.
- **Calculation**:
  - Average of the **absolute gaze** vectors for the left and right eyes:
    - `PoG = (absGaze_leftEye + absGaze_rightEye) / 2`
- **Range**:
  - Same as **absolute gaze**, since it is the midpoint of both gaze vectors.
- **Visualization**:
  - No direct visualization, but this is the collective focus point of both eyes.

---

### 4. Vergence
- **Description**:
  - Represents the angular difference between the gaze directions of the left and right eyes.
  - Indicates how much the eyes are converging or diverging.
- **Calculation**:
  - Euclidean distance between the **absolute gaze** positions of both eyes:
    - `vergence = np.linalg.norm(absGaze_leftEye - absGaze_rightEye)`
- **Range**:
  - A smaller value indicates more convergence (e.g., focusing on a nearby object).
  - A larger value indicates divergence (e.g., focusing on a distant object).

---

### 5. Head Pose
- **Description**:
  - Represents the 2D direction of the head relative to the image frame.
  - Calculated based on the **nose tip landmark** (`landmark[4]`).
- **Calculation**:
  - Projects the **nose tip** into 3D world coordinates using the transformation matrix.
  - Normalizes the head direction vector and projects it back into 2D image space.
- **Range**:
  - Values are in 2D coordinates relative to the **frame center**.
  - Positive X values indicate the head is pointing to the right.
  - Positive Y values indicate the head is pointing up (Y-axis is inverted).
- **Visualization**:
  - A line is drawn from the **nose tip** to the calculated head direction on the frame.

---

## Landmarks Used

- **Nose tip**: `landmark[4]`
- **Left eye**: `landmark[468]` (pupil position), `landmark[263]` (corner).
- **Right eye**: `landmark[473]` (pupil position), `landmark[33]` (corner).
- **Chin**: `landmark[152]`
- **Mouth corners**: `landmark[287]` (left), `landmark[57]` (right).

---

## Notes

1. The **camera matrix** is estimated based on the frame resolution.
2. **Transformation Matrix**: Used to convert 2D image coordinates to 3D world coordinates and back.
3. Scale factors are applied to ensure all vectors are proportional.

--- 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Make a new Python environment with set version and additional packages
```buildoutcfg
conda create -n "eyedetect" python=3.11.4 ffmpeg tqdm dlib
```
Activate new environment called "eyedetect"
```buildoutcfg
conda activate eyedetect
```

2. Install a couple other packages with pip
```buildoutcfg
pip install opencv-python
```
```buildoutcfg
pip install --upgrade mediapipe
```
```buildoutcfg
brew install wget
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

https://github.com/user-attachments/assets/c57a8ac7-6bcc-4a56-b231-f53c57c04c48


<p align="right">(<a href="#readme-top">back to top</a>)</p>

![mediapipe_face_landmark_fullsize](https://github.com/user-attachments/assets/4726dee0-295c-445a-b5f4-324ddb526a23)


<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/mayolabvision/video_eyeTracking/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/mayolabvision/video_eyeTracking/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mayolabvision/video_eyeTracking" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@kendranoneman](https://twitter.com/kendranoneman) - knoneman@andrew.cmu.edu

Project Link: [https://github.com/mayolabvision/video_eyeTracking](https://github.com/mayolabivision/video_eyeTracking)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

https://link.springer.com/chapter/10.1007/978-3-030-82681-9_4
https://dl.acm.org/doi/pdf/10.1145/2578153.2578185

* [Python-Gaze-Face-Tracker (alireza787b)](https://github.com/alireza787b/Python-Gaze-Face-Tracker)
* [iris-Segmentation-mediapipe-python (Asadullah-Dal17)](https://github.com/Asadullah-Dal17/iris-Segmentation-mediapipe-python)
* [Gaze_estimation (amitt1236)](https://github.com/amitt1236/Gaze_estimation?source=post_page-----570d4683fe23--------------------------------)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/mayolabvision/video_eyeTracking.svg?style=for-the-badge
[contributors-url]: https://github.com/mayolabvision/video_eyeTracking/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/mayolabvision/video_eyeTracking.svg?style=for-the-badge
[issues-url]: https://github.com/mayolabvision/video_eyeTracking/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/mayolabvision
[banner]: https://github.com/mayolabvision/.github/blob/main/mayolab-logo.png
[website]: https://www.mayolab.net/research
[twitter]: https://twitter.com/mayo_lab
[Python.com]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org
[MediaPipe.com]: https://img.shields.io/badge/MediaPipe-00C853?style=for-the-badge&logo=mediapipe&logoColor=white
[MediaPipe-url]: https://mediapipe.dev
[OpenCV.com]: https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org
