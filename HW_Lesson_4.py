import cv2
import numpy as np

img = cv2.imread("images/girls.jpg")

img_copy = img.copy()
img_copy_color = img_copy
img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 8)

#піdсилення контрасту
img_copy = cv2.equalizeHist(img_copy)
img_copy = cv2.Canny(img_copy, 100, 150)

contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #external знаходить лише крайні зовнішні контури, chain_aprox - ставимо точки контуру
x_list = [cv2.boundingRect(cnt)[0] for cnt in contours if 50 < cv2.contourArea(cnt) < 200]
min_x = min(x_list)
max_x = max(x_list)

#малювання контурів прямокутників та тест
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 70 < area < 200:  #фільтр шуму
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 5 if y - 5 > 10 else y + 15
        if x == min_x:
            person = "Nadia"
        elif x == 897:
            person = "Olecia"
        elif x == 522:
            person = "Tetianka"
        else:
            person = "none"
        text = f"x:{x}, y:{y}, person:{person}, s: {area}"
        cv2.putText(img_copy_color, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# cv2.imshow("img", img)
# cv2.imshow("img_copy", img_copy)

cv2.imshow("copy", img_copy_color)
cv2.waitKey(0)
cv2.destroyAllWindows()