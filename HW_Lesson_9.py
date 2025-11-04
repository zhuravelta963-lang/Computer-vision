import cv2

net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt", 'data/MobileNet/mobileNet.caffemodel')

classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

Files = [          "plyashka.jpg",
                    "dog.jpg",
                    "shpruz.jpg",
                    "vanna.jpg"]

Clasess={}

for name in Files:
    Image = f"images/MobileNet/{name}"
    image = cv2.imread(Image)

    if image is None:
        print("None")
        continue




    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (224, 224)),
        1.0 / 127.5,
        (224, 224),
        (127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    preds = net.forward()

    idx = preds[0].argmax()

    label = classes[idx] if idx < len(classes) else "unknown"
    conf = float(preds[0][idx]) * 100


    print(f"Файл: {name}")
    print(f"Клас: {label}")
    print(f"Впевненість: {round(conf, 2)}%")



    if label in Clasess:
        Clasess[label] += 1
    else:
        Clasess[label] = 1


    text = f'{label}: {int(conf)}%'
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow(f"Image: {name}", image)
    cv2.waitKey(0)



cv2.destroyAllWindows()
for label, count in Clasess.items():
    print(f"{label:<26}, {count} раз")