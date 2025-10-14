import cv2

#обробка відео

#fps - частота кадрів

cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
grey1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
grey1 = cv2.convertScaleAbs(grey1, alpha=1, beta=0) #яскравість і тмяність


while True:
    ret, frame2 = cap.read()  #frame - кадр якиий перебирається, ret - чи відкритий кадр
    if not ret:
        print("відео скінчилося")
        break
    grey2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(grey2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(grey1, grey2) #різниця
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video", frame2)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() #коли буде закрито, то закрий цей поток, камера закривається
cv2.destroyAllWindows()



