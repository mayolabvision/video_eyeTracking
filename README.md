# video_eyeTracking_ECoG

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

