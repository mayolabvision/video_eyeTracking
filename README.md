# video_eyeTracking_ECoG

1. Make a new Python environment
```buildoutcfg
# Sets python version, adds necessary packages
conda create -n "eyedetect" python=3.11.4 ffmpeg tqdm dlib
```
```buildoutcfg
# Activate new environment called "eyedetect"
conda activate eyedetect
```

2. Install a couple other packages with pip
```buildoutcfg
pip install opencv-python
```
```buildoutcfg
pip install --upgrade mediapipe
```
