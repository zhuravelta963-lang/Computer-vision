import cv2
import os
import math
import pandas as pd
from ultralytics import YOLO
import yt_dlp

# -------------------------
# Налаштування шляхів
# -------------------------
PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, 'cars_data.csv')

YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"

MODEL_PATH = "yolov8l.pt"
CONF_THRESH = 0.4
TRACKER = "bytetrack.yaml"

# -------------------------
# Отримання прямого посилання на відео
# -------------------------
ydl_opts = {
    'format': 'best[ext=mp4]',
    'quiet': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(YOUTUBE_URL, download=False)
    video_url = info['url']

cap = cv2.VideoCapture(video_url)

# -------------------------
# Завантаження моделі
# -------------------------
model = YOLO(MODEL_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 30

# Масштаб (потрібно калібрувати!)
PIXEL_TO_METER = 0.05

# -------------------------
# Змінні
# -------------------------
previous_positions = {}
car_counter = 0
id_map = {}

car_speed_history = {}
car_average_speed = {}

# -------------------------
# Основний цикл
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    results = model.track(
        frame,
        conf=CONF_THRESH,
        tracker=TRACKER,
        persist=True,
        stream=True,
        verbose=False
    )

    r = next(results)

    if r.boxes is None:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    ids = boxes.id.cpu().numpy() if boxes.id is not None else None

    height, width, _ = frame.shape
    roi_top = int(height * 0.45)

    for i in range(len(xyxy)):

        class_id = int(cls[i])
        class_name = model.names[class_id]

        if class_name != "car":
            continue

        if ids is None:
            continue

        track_id = int(ids[i])

        if track_id not in id_map:
            car_counter += 1
            id_map[track_id] = car_counter

        car_number = id_map[track_id]

        x1, y1, x2, y2 = xyxy[i].astype(int)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # ROI-фільтр
        if center_y < roi_top:
            continue

        speed_kmh = 0

        if track_id in previous_positions:
            prev_x, prev_y = previous_positions[track_id]

            pixel_distance = math.sqrt(
                (center_x - prev_x) ** 2 +
                (center_y - prev_y) ** 2
            )

            distance_meters = pixel_distance * PIXEL_TO_METER
            speed_mps = distance_meters * fps
            raw_speed = speed_mps * 3.6

            # фільтрація шуму
            if raw_speed > 5:

                if car_number not in car_speed_history:
                    car_speed_history[car_number] = []

                car_speed_history[car_number].append(raw_speed)

                # беремо середнє останніх 10 значень
                last_speeds = car_speed_history[car_number][-10:]
                speed_kmh = sum(last_speeds) / len(last_speeds)

                car_average_speed[car_number] = speed_kmh

        previous_positions[track_id] = (center_x, center_y)

        label = f"Car {car_number} | {speed_kmh:.1f} km/h"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Лінії на дорозі
    cv2.line(
        frame,
        (int(width * 0.4), int(height * 0.4)),
        (int(width * 0.8), int(height * 0.50)),
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (int(width * 0.20), int(height * 0.54)),
        (int(width * 0.90), int(height * 0.75)),
        (255, 255, 255),
        3
    )

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------
# Збереження CSV
# -------------------------
final_data = []

for car_number, avg_speed in car_average_speed.items():
    final_data.append({
        "Car": f"Car {car_number}",
        "Average Speed (km/h)": round(avg_speed, 2)
    })

df = pd.DataFrame(final_data)
df.to_csv(CSV_PATH, index=False)

print("Дані збережено у:", CSV_PATH)
