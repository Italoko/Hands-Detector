import cv2
import numpy as np

log_console = True #Show exceptions in terminal

#For calc coordinates
def Manhattan(p1,p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) #|x1 – x2| + |y1 – y2|

#Returns hands detected
def hands_detector(mp_hands_instance, frame):
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return mp_hands_instance.process(frame)
            
# Draw the hand annotations on the image.
def draw_hand_connections(frame, result_mp, mp_hands, mp_draw):
    if result_mp.multi_hand_landmarks:
        frame.flags.writeable = True
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for hand_landmarks in result_mp.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    elif log_console: 
        print("hand not detected")
    return frame

#Returns specific hand detected
def get_hand(result_mp, hand):
    if result_mp.multi_hand_landmarks and len(result_mp.multi_hand_landmarks)-1 >= hand: 
        return result_mp.multi_hand_landmarks[hand]
    elif log_console:
        print("index out of range or hand not detected")
    return None

#Returns label of an specific hand detected
def get_hand_label(result_mp, hand):
    if result_mp.multi_hand_landmarks and len(result_mp.multi_handedness)-1 >= hand: 
        for idx, hand_handedness in enumerate(result_mp.multi_handedness):
            return hand_handedness.classification[hand].label
    elif log_console:
        print("index out of range or hand not detected")
    return None

#Returns hand side relative to screen
def get_hand_side(hand,w, h):
    wrist = get_landmark(hand,0)
    coord_wrist = tuple(np.multiply(np.array((wrist[0],wrist[1])), [w,h]).astype(int))

    dist_wrist_to_minborder = Manhattan(coord_wrist, (0, coord_wrist[1]))
    dist_wrist_to_maxborder = Manhattan(coord_wrist, (w, coord_wrist[1]))

    if dist_wrist_to_minborder < dist_wrist_to_maxborder:
        return "Left"
    elif dist_wrist_to_minborder > dist_wrist_to_maxborder:
        return "Right"
    else:
        return "Center"

#Returns the coordinates of a specific hand/finger point
def get_landmark(hand, key):
    if(hand is not None):
        return  hand.landmark[key].x , hand.landmark[key].y, hand.landmark[key].z
    elif log_console:
        print("hand not detected")
    return 0,0,0

#Returns if the dots are connected
def linked_finger(hand, key_a, key_b):
    x = False
    y = False 

    if(hand is not None):
        if(abs(hand.landmark[key_a].x - hand.landmark[key_b].x) * 10 <= 0.5):
            x = True
        if(abs(hand.landmark[key_a].y - hand.landmark[key_b].y) * 10 <= 0.5): 
            y = True
    elif log_console:
        print("hand not detected")
    
    return x,y
