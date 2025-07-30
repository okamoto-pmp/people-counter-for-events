import cv2

cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"NV12"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ok, frame = cap.read()
    if not ok:
        print("grab failed"); break
    cv2.imshow("cam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
