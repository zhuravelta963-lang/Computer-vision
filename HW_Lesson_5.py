import cv2
import numpy as np
img = cv2.imread('photo/photo.jpg')
img_copy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([119, 17, 36]) #мінімальний поріг, створ.' матриц.
upper = np.array([172, 220, 165]) #максимальний поріг
mask = cv2.inRange(img, lower, upper) #в послідовності
img = cv2.bitwise_and(img, img, mask=mask) #накладаємо маску на початкове зображ

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt) #момент контуру

        #центр мас

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])


        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2) #допомагає відрізняти співвідношення сторін (чи це прямокутник чи це квадрат)
        #міра округлості обёєкта
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)

        #до якого типу відноситься фігура
        approx = cv2.approxPolyDP(cnt, 0.005 * perimeter, True)
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "square"
        elif len(approx) > 6 and 0.7 < compactness < 0.97:
            shape = "oval"
        elif len(approx) >= 5:
            shape = "zirka"
        else:
            shape = "another"

        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)
        cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(img_copy, f"shape:{shape}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img_copy, f"area: {int(area)}, p: {int(perimeter)}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img_copy, f"AR:{aspect_ratio}, C: {compactness}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
cv2.imshow('mask',img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()