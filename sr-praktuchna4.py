import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#2 - формуємо набори даних для навчання нейронки

#X = []список ознак
#y = [] список міток
#ознаки: тип фігури, колір
#мітки - список правильних відповідей
X = []
y = []
colors = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255)
}
shapes = ["square", "triangle", "oval"]
for color_name, bgr in colors.items():
    for i in range(30):
        noise = np.random.randint(-5, 15, 3)
        sample = np.clip(np.array(bgr) + noise, 0, 255)
        X.append(sample)
        y.append(color_name)

#3 розділяємо дані за пропорцієб 70 на 30: 70% для навчання, 30% на перевірку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=42)

#4 навчаємо модель: дивиться на найблидчі приклади зправа і зліва
model = KNeighborsClassifier(n_neighbors = 5) #БАЖАНО НЕПАРНІ ЧИСЛА, якщо зображ складе, то більше число
model.fit(X_train, y_train)#запамёятовуємо тренуваьні приклади


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (20, 50, 50), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            mask_cnt = np.zeros(frame.shape[:2], "uint8")
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_bgr = cv2.mean(frame, mask_cnt)[:3]
            enterbgr = np.array(mean_bgr).reshape(1, -1)
            pred_color = model.predict(enterbgr)[0]
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y + h, x:x + w]
            perimeter = cv2.arcLength(cnt, True)





            aspect_ratio = round(w / h, 2)
            # міра округлості обёєкта


            # до якого типу відноситься фігура
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            if len(approx) == 3:
                shape = "triangle"
            elif len(approx) == 4:
                shape = "square"
            elif len(approx) > 8:
                shape = "oval"
            else:
                shape = "another"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, f"{shape}, {pred_color}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("color", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()