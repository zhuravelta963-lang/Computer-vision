import cv2

#1 крок завантажуємо моделі

net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt", "data/MobileNet/mobilenet.caffemodel")
#2 крок зчитуємо список назв у класі

classes = []
with open("data/MobileNet/synset.txt", "r", encoding = "utf-8") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

#3 готуємо зображення для мережі
image = cv2.imread("images/MobileNet/dog.jfif")
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)

#4 класдемо зображення в мережу та запускаємо
net.setInput(blob)
preds = net.forward() #вектор імовірності для класів

#5 знаходимо індеск класу з найбільшою імовірністю

idx = preds[0].argmax()

#6 знаходимо назву класа та впевненість у відсотках
label = classes[idx] if idx < len(classes) else "unknown"
conf = float(preds[0][idx]) * 100

#7 виводимо  результати в консоль
print("Class: ", label)
print("Confid: ", conf)

#виводимо на екран
text = f'{label}: {int(conf)}%'
cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()