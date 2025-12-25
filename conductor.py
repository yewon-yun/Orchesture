import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)
while webcam.isOpened():
    success, img = webcam.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #difference in opencv and mediapipes' color model
    result = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3).process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #change image color back
    if result.multi_hand_landmarks: #any hand detected
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

    
    cv2.imshow('Conductor',img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows







