import cv2
import csv
import time
import mediapipe as mp
from pathlib import Path
import math

mp_hands = mp.solution.hands

#standardizing hand datasets bc they may vary in size and distance-------------------------------------------------------

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y) #the standard wrist to middlefinger distance for each hand

def extract_features(landmarks):
    feature = []
    scale = _dist(landmarks[0], landmarks[9])

    for i in landmarks:
        feature.append((i.x-landmarks[0].x)/scale)
        feature.append((i.x-landmarks[0])/scale)

    return feature


#snapshotting process---------------------------------------------------------------------------------------------------

def main():
    
