import cv2
import mediapipe as mp

notes = ["B", "A", "G", "F", "E", "D", "C"]

#region class to determine what pitch its supposed to be at---------------------------------------------------------
class RightPitch:
    def __init__(self, region=None, highoct=False, sharp=False):
        self.region = region
        self.highoct = highoct
        self.sharp = sharp

class LeftChord:
    def __init__(self, region=None, minor=False, seven=False):
        self.region = region
        self.minor = False
        self.seven = False


right = RightPitch()
left = LeftChord()

mp_hands = mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils


#webcam track-------------------------------------------------------------------------------------------------------
webcam = cv2.VideoCapture(0) #capturing the webcam

while webcam.isOpened():
    success, img = webcam.read()

    if not success:
        print("ignoring empty web cam")
        continue

    #for easier calculation-----------------------------------------------------------------------------------------
    h = int(img.shape[0])
    w = int(img.shape[1])
    o = int(h/7)


    #using mediapipe------------------------------------------------------------------------------------------------

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #difference in opencv and mediapipe's color model
    result = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3).process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #change image color back
    if result.multi_hand_landmarks: #any hand detected
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

    if result.multi_hand_landmarks and result.multi_handedness:
        for i in range(len(result.multi_hand_landmarks)):
            hand_landmarks = result.multi_hand_landmarks[i] #which hand
            whichhand = result.multi_handedness[i].classification[0].label #left or right

            mp_drawing.draw_landmarks(img, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

            pos = hand_landmarks.landmark[12]
            pos_pixel = pos.y * h #because pos.y is a decimal between 0.0 and 1.0

            region = int(pos_pixel/o) #using the 7 divide we used earlier to determine the position

            if whichhand == "Right":
                left.region = region
            elif whichhand == "Left":
                right.region = region

            


    #lines for note regions-------------------------------------------------------------------------------------------
    cv2.line(img, (0,o),(w,o),(217,221,220), 1)
    cv2.line(img, (0,2*o),(w,2*o),(217,221,220), 1)
    cv2.line(img, (0,3*o),(w,3*o),(217,221,220), 1)
    cv2.line(img, (0,4*o),(w,4*o),(217,221,220), 1)
    cv2.line(img, (0,5*o),(w,5*o),(217,221,220), 1)
    cv2.line(img, (0,6*o),(w,6*o),(217,221,220), 1)
    

    #show image-------------------------------------------------------------------------------------------------------
    img = cv2.flip(img,1)
    
    if right.region is not None:
        cv2.putText(img, f"Right region: {notes[right.region]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if left.region is not None:
        cv2.putText(img, f"Left region: {notes[left.region]}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Conductor',img) #mirroring the image

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows

