import cv2
import numpy as np

face_net = cv2.dnn.readNetFromCaffe("data/DNN/deploy.prototxt", "data/DNN/res10_300x300_ssd_iter_140000.caffemodel")


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('tracking face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    (w, h) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0)) #Перетворює зображення на формат, яке може прочитати днн(нейронка)
    face_net.setInput(blob)
    detection = face_net.forward() #forward виконує перебір зображ через мережу
    #detection - масив зі знайденими обличчями
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.5:
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            x, y = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(frame, (x, y), (x2 + w, y2 + h), (255, 0, 0), 2)
cap.release()
cv2.destroyAllWindows()