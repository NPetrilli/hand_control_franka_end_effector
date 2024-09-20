import cv2 
import mediapipe as mp
cap = cv2.VideoCapture(0)
import math
import numpy as np
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils


# Function to classify different hands in frame
def get_label(result,index,hand):
    output = None
    for idx,obj in enumerate(result.multi_handedness):
        if obj.classification[0].index==index:
         label=obj.classification[0].label
         score=obj.classification[0].score
         text='{} {}'.format(label,round(score,2))
         coords=tuple(np.multiply(np.array((hand.landmark[0].x,hand.landmark[0].y)),[640,480]).astype(int))

         output=label,coords,text
    return output


def distanceFunc(hand_detected,frame):
   thumb = hand_detected.landmark[4]
   index = hand_detected.landmark[8]
   thumb_x = int(thumb.x * frame.shape[1])
   thumb_y = int(thumb.y * frame.shape[0])
   index_x = int(index.x * frame.shape[1])
   index_y = int(index.y * frame.shape[0])
   distance = math.sqrt((thumb.x-index.x)**2+(thumb.y-index.y)**2)
   frame= cv2.line(frame,(thumb_x,thumb_y),(index_x,index_y), (0, 255, 0),2)
   width = ((distance-0.026)/(0.26-0.026))*10.5
   if width > 10.5:
      width = 10.5
   if width<0:
      width = 0
   output=frame,width
   return output

def findpos(hand_detected):
    up=[8,12,16,20]
    down=[5,9,13,17]
    landmarks=[]
    for id, l in enumerate(hand_detected.landmark):
     landmarks.append([id,int(l.x*640),int(l.y*480)])
    c=0
    for i in range(0,4):
     if landmarks[up[i]][2]<landmarks[down[i]][2]:
        c+=1
    if c==4:
       return "Open"
    else:
       return "Closed"

while True:
    success, frame = cap.read() #BGR
    if success:

        converted_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #Process next frame
        result = hand.process(converted_frame)

        #If find some hands
        if result.multi_hand_landmarks:

            for num,hand_detected in enumerate(result.multi_hand_landmarks):
             mp_drawing.draw_landmarks(frame,hand_detected,mp_hands.HAND_CONNECTIONS)
              
             if get_label(result,num,hand_detected):
              class_result=get_label(result,num,hand_detected)
              cv2.putText(frame,class_result[2],class_result[1],cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
              print(class_result[0])
              if class_result[0]=="Right":
                 frame,distance = distanceFunc(hand_detected,frame)
                 print(distance)
              if class_result[0]=="Left":
                 print(findpos(hand_detected))
              

        cv2.imshow("Video Flow",frame)
        
        if cv2.waitKey(1)==ord('q'):
            break
cv2.destroyAllWindows()


    
               
                
               


