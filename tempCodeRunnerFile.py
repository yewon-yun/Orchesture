import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)
while webcam.isOpened():
    success, img = webcam.read()

    if not success:
        print("ignoring empty web cam")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #difference in opencv and mediapipe's color model
    result = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3).process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #change image color back
    if result.multi_hand_landmarks: #any hand detected
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

    #lines for note regions

    h = int(img.shape[0])
    w = int(img.shape[1])

    o = int(h/8)

    cv2.line(img, (0,o),(w,o),(217,221,220), 1)
    cv2.line(img, (0,2*o),(w,2*o),(217,221,220), 1)
    cv2.line(img, (0,3*o),(w,3*o),(217,221,220), 1)
    cv2.line(img, (0,4*o),(w,4*o),(217,221,220), 1)
    cv2.line(img, (0,5*o),(w,5*o),(217,221,220), 1)
    cv2.line(img, (0,6*o),(w,6*o),(217,221,220), 1)
    cv2.line(img, (0,7*o),(w,7*o),(217,221,220), 1)
    
    cv2.imshow('Conductor',cv2.flip(img, 1)) #mirroring the image
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows







