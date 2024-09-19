import cv2 
import mediapipe as mp
cap = cv2.VideoCapture(0)
import math

mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

while True:
    success, frame = cap.read() #BGR
    if success:
        converted_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        result = hand.process(converted_frame)
        if result.multi_hand_landmarks:
            hand_detected = result.multi_hand_landmarks[0]
            thumb = hand_detected.landmark[4]
            index = hand_detected.landmark[8]
            thumb_x = int(thumb.x * frame.shape[1])
            thumb_y = int(thumb.y * frame.shape[0])
            index_x = int(index.x * frame.shape[1])
            index_y = int(index.y * frame.shape[0])
            distance = math.sqrt((thumb.x-index.x)**2+(thumb.y-index.y)**2)
            #print(distance)
            frame= cv2.line(frame,(thumb_x,thumb_y),(index_x,index_y), (0, 255, 0),2)
            for hand_landmarks in result.multi_hand_landmarks:
                #print(hand_landmarks)
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                

            width = ((distance-0.026)/(0.26-0.026))*10.5
            print(width)

        cv2.imshow("Video Flow",frame)
        
        if cv2.waitKey(1)==ord('q'):
            break
cv2.destroyAllWindows()