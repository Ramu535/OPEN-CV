import cv2

cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()  # Read a frame
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cv2.line(frame, (0,219), (364,0), (0,0,255), 5) 
cap.release()  # Release the camera
cv2.destroyAllWindows()