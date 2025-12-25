import cv2
import mediapipe as mp

mp_hands = mp.solution.hands
mp.drawing=mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)
while webcam.isOpened():
    success, img = webcam.read()

    img = cv2.cvtColor(img, cvt.COLOR_BGR2RGB)
    result = mp_hands.Hands().process(img)


