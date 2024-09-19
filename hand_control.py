import cv2 

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if success:
        cv2.imshow("Video Flow",frame)
        
        if cv2.waitKey(1)==ord('q'):
            break
cv2.destroyAllWindows()