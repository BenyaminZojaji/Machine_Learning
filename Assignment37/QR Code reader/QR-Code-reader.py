import cv2
import webbrowser

cap = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    data, bbox, rectifiedImage = detector.detectAndDecode(frame)
    if data:
        x, y, w, h = cv2.boundingRect(bbox[0])
        cv2.putText(frame, data, (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (28, 28, 183), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        if cv2.waitKey(1) == ord("e"):
            webbrowser.open(str(data))

    cv2.imshow("QR-Code-scanner", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
