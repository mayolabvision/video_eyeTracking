import os
import subprocess
from tqdm import tqdm

def get_landmarksFile():
    landmarksFile =  "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(landmarksFile):
        print('installing landmarks file')
        subprocess.Popen('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', shell=True)
        subprocess.Popen('bunzip2 /content/shape_predictor_68_face_landmarks.dat.bz2', shell=True)
    else:
        print('landmarks file already installed')

def convert_mov_to_flv(file):
    print(file)
