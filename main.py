import cv2
import mediapipe as mp
import time 
import hands_detector as detector

#Settings mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    max_num_hands = 1)

def print_fps(image,current_time, prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(image,f'FPS: {str(int(fps))}',(1,15),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),2)
    return image, current_time

def main ():
    cap = cv2.VideoCapture(0)
    prev_time = time.time() #For FPS    

    with hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Error capturing camera image")
                break

            #Return Collection of detected/tracked hands
            result = detector.hands_detector(hands,image)
            
            # If detected hands:
            if result.multi_hand_landmarks:  
                image = detector.draw_hand_connections(image,result,mp_hands,mp_drawing) #Draw hand

                #Returns specific hand detected    
                print("Hand:")
                hand_detected = detector.get_hand(result,0)
                #print(hand_detected)
                
                #Returns hand label (left or right) *consider selfie-view display
                print(f"Label: {detector.get_hand_label(result,0)}")

                #Returns the coordinates (x,y,z) of a specific point
                # // 8 ~ INDEX_FINGER_TIP
                print(f"Coordinates INDEX_FINGER_TIP:{detector.get_landmark(hand_detected,8)}")

                #Returns (bool,bool) if the dots are connected
                # // (x,y)
                # // 8 ~ INDEX_FINGER_TIP and 4 ~ THUMB_TIP
                print(detector.linked_finger(hand_detected,8,4))

            cv2.flip(image, 1) # Flip the image horizontally for a selfie-view display.
            image, prev_time = print_fps(image,time.time(),prev_time) # Print FPS Screen
            cv2.imshow('Capture',image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == "__main__":
    main()