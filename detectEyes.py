import cv2 as cv
import numpy as np
from operator import rshift
import mediapipe as mp
import dlib

import os
import subprocess
from tqdm import tqdm

from helpers import convert_mov_to_flv,get_landmarksFile

get_landmarksFile()

landmarksFile =  "/content/shape_predictor_68_face_landmarks.dat"
