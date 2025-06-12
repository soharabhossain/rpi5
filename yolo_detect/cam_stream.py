import cv2

cap = cv2.VideoCapture('http://192.168.1.37:8080/video')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run YOLO inference here
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
