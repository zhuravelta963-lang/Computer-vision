import cv2
import numpy as np


cap = cv2.VideoCapture(0)
lover_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lover_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# points = []
while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lover_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lover_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        cv2.rectangle(frame, (0, 0), (0, 0, 255), 2)
        if area > 1000:
            obj = True
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(frame, (int(cx), int(cy)), 7, (255, 0, 0), -1)
            if obj:
                

    #             points.append((cx, cy))
    #         #print(points)
    # for i in range(1, len(points)):
    #     if points[i - 1] is None or points[i] is None:
    #         continue
    #     cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)

    cv2.imshow("video", frame)
    cv2.imshow("video2", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()