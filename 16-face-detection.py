# https://docs.opencv.org/master/d1/de5/classcv_1_1CascadeClassifier.html

import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
cap = cv2.VideoCapture('imagens/rosto.webm')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=.6, fy=.6)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.5, 5)  # (imagem, fator de escala, minimo de vizinhos)
    eyes = eye_cascade.detectMultiScale(gray, 1.5, 10)


    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(frame, (x,y-25),(x+w,y),(0,255,0),-1)
        cv2.putText(frame, "Rosto Detectado", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1,cv2.LINE_AA)


    for (x,y,w,h) in eyes:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,255),2)


    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        cv2.destroyAllWindows()
        exit()

