import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#1 - створюємо функцію длля генерації простих фігур
def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        point = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [point], 0, color, -1)
    return img

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
}
shapes = ["circle", "square", "triangle"]

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3] #mean повертає значення (b, g, r, alpha)
            features = [mean_color[0], mean_color[1], mean_color[2]]
            X.append(features)
            y.append(f"{color_name}_{shape}")

#3 розділяємо дані за пропорцієб 70 на 30: 70% для навчання, 30% на перевірку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=42)

#4 навчаємо модель: дивиться на найблидчі приклади зправа і зліва
model = KNeighborsClassifier(n_neighbors = 3) #БАЖАНО НЕПАРНІ ЧИСЛА, якщо зображ складе, то більше число
model.fit(X_train, y_train)#запамёятовуємо тренуваьні приклади

#5 перевіряємо точність
accuracy = model.score(X_test, y_test)
print(f"точність моделі:{round(accuracy * 100, 2)}%")
test_image =  generate_image((0, 255, 0), "circle")
mean_color = cv2.mean(test_image[:3])
prediction = model.predict(mean_color)
print(f"передбачення: {prediction[0]}")
cv2.imshow("img", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


