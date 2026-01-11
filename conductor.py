import cv2
import mediapipe as mp
import joblib
import math

notes = ["B", "A", "G", "F", "E", "D", "C"]

#the functions from the training model-----------------------------------------------------------------------------
model = joblib.load("models/knn_model.joblib") #for the two trained gestures we got

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y) #the standard wrist to middlefinger distance for each hand

def extract_features(landmarks):
    feature = []
    scale = _dist(landmarks[0], landmarks[9]) + 1e-6

    for i in landmarks:
        feature.append((i.x-landmarks[0].x)/scale)
        feature.append((i.y-landmarks[0].y)/scale)

    return feature

#region class to determine what pitch its supposed to be at---------------------------------------------------------
class RightPitch:
    def __init__(self, region=None, highoct=False, sharp=False, stop=False):
        self.region = region
        self.highoct = highoct
        self.sharp = sharp
        self.stop = stop


class LeftChord:
    def __init__(self, region=None, minor=False, seven=False, stop=False):
        self.region = region
        self.minor = minor
        self.seven = seven
        self.stop = stop


right = RightPitch()
left = LeftChord()

mp_hands = mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils


#webcam track-------------------------------------------------------------------------------------------------------
webcam = cv2.VideoCapture(0) #capturing the webcam

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #change image color back

    if result.multi_hand_landmarks and result.multi_handedness: #let's say the hand exists
        #for each hand
        for i in range(len(result.multi_hand_landmarks)):
            hand_landmarks = result.multi_hand_landmarks[i] #which hand
            whichhand = result.multi_handedness[i].classification[0].label #left or right

            features = extract_features(hand_landmarks.landmark)
            gesture = model.predict([features])[0] #we use the model to actually track the hand gestures

            mp_drawing.draw_landmarks(img, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

            pos = hand_landmarks.landmark[12]
            pos_pixel = pos.y * h #because pos.y is a decimal between 0.0 and 1.0

            region = max(0, min(6, int(pos_pixel / o))) #using the 7 divide we used earlier to determine the position, bound between 0 and 6

            if whichhand == "Right":
                left.region = region
                if gesture == "pinky":
                    left.seven = True
                else:
                    left.seven = False

                if gesture == "thumb":
                    left.minor = True
                else:
                    left.minor = False
                
                if gesture == "stop":
                    left.stop = True
                else:
                    left.stop = False
                
                if gesture == "both":
                    left.seven = True
                    left.minor = True
                else:
                    left.seven = False
                    left.minor = False

            elif whichhand == "Left":
                right.region = region
                if gesture == "pinky":
                    right.highoct = True
                else:
                    right.highoct = False

                if gesture == "thumb":
                    right.sharp = True
                else:
                    right.sharp = False
                if gesture == "stop":
                    right.stop = True
                else:
                    right.stop = False
                if gesture == "both":
                    right.highoct = True
                    right.sharp = True
                else:
                    right.highoct = False
                    right.sharp = False



    #lines for note regions-------------------------------------------------------------------------------------------
    cv2.line(img, (0,o),(w,o),(217,221,220), 1)
    cv2.line(img, (0,2*o),(w,2*o),(217,221,220), 1)
    cv2.line(img, (0,3*o),(w,3*o),(217,221,220), 1)
    cv2.line(img, (0,4*o),(w,4*o),(217,221,220), 1)
    cv2.line(img, (0,5*o),(w,5*o),(217,221,220), 1)
    cv2.line(img, (0,6*o),(w,6*o),(217,221,220), 1)
    

    #show image-------------------------------------------------------------------------------------------------------
    img = cv2.flip(img,1)

    octav = "+1" if right.highoct==True else "0"
    shar = "#" if right.sharp==True else " "
    
    mino = "m" if left.minor==True else "M"
    sev = "7" if left.seven==True else " "



    
    if right.region is not None and right.stop == False:
        cv2.putText(img, f"Right region: {notes[right.region]}{shar}{octav}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        cv2.putText(img, f"Right region: break", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    if left.region is not None and left.stop == False:
        cv2.putText(img, f"Left region: {notes[left.region]}{mino}{sev}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        cv2.putText(img, f"Left region: break", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow('Conductor',img) #mirroring the image

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

