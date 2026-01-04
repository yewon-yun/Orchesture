import cv2
import csv
import time
import mediapipe as mp
from pathlib import Path
import math

mp_hands = mp.solutions.hands

#standardizing hand datasets bc they may vary in size and distance-------------------------------------------------------

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y) #the standard wrist to middlefinger distance for each hand

def extract_features(landmarks):
    feature = []
    scale = _dist(landmarks[0], landmarks[9]) + 1e-6

    for i in landmarks:
        feature.append((i.x-landmarks[0].x)/scale)
        feature.append((i.y-landmarks[0].y)/scale)

    return feature


def main():
    #setting up csv---------------------------------------------------------------------------------------------------------
    
    Path("data").mkdir(exist_ok=True)
    out_path = Path("data/gesture_samples.csv")

    header = ["label", "hand"] + [f"f{i}" for i in range(42)]
    if not out_path.exists():
        with out_path.open("w", newline="") as f:
            csv.writer(f).writerow(header)

    #snapshotting process----------------------------------------------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    current_label = "neutral"
    recording = False

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #same thing we did in the earlier file
        res = hands.process(rgb)

        cv2.putText(frame, f"label={current_label} rec={recording}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if res.multi_hand_landmarks and res.multi_handedness:
            lm = res.multi_hand_landmarks[0].landmark
            hand_label = res.multi_handedness[0].classification[0].label #gonna be left or right

            if recording:
                feats = extract_features(lm) #use the helper function to figure out values
                row = [current_label, hand_label] + feats
                with out_path.open("a", newline="") as f: #actually write it in csv
                    csv.writer(f).writerow(row)

        cv2.imshow("Collect Gestures", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
        elif key == ord("0"):
            current_label = "neutral"
        elif key == ord("1"):
            current_label = "thumb"
        elif key == ord("2"):
            current_label = "pinky"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()   
