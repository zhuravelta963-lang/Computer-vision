import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

TIME_LIMIT = 1
MOVE_DIST = 10
TARGET_CLASSES = {"backpack", "handbag", "suitcase"}

frame_count = 0
last_pos = {}
still_frames = {}
suspicious = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.track(frame, persist=True, verbose=False)[0]

    if results.boxes is None:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    ids = results.boxes.id

    if ids is None:
        ids = list(range(len(boxes)))
    else:
        ids = ids.cpu().numpy()

    for i in range(len(boxes)):
        box = boxes[i]
        cls = int(classes[i])
        tid = int(ids[i])

        class_name = model.names[cls]
        if class_name.lower() not in {c.lower() for c in TARGET_CLASSES}:
            continue

        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if tid in last_pos:
            px, py = last_pos[tid]
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

            if dist < MOVE_DIST:
                if tid not in still_frames:
                    still_frames[tid] = frame_count
                elif frame_count - still_frames[tid] > fps * TIME_LIMIT:
                    suspicious.add(tid)
            else:
                still_frames.pop(tid, None)
        else:
            still_frames[tid] = frame_count

        last_pos[tid] = (cx, cy)

        if tid in suspicious:
            color = (0, 0, 255)
            label = "Suspicious"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()