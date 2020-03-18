# Foreground Detection: https://en.wikipedia.org/wiki/Foreground_detection
# Background Subtraction: https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html


import numpy as np
import cv2


fgbg = cv2.createBackgroundSubtractorKNN(history=1, detectShadows=False)  #  Cria segmentador de fundo
# fgbg = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=150, detectShadows=False)  #  Cria segmentador de fundo

cap = cv2.VideoCapture('imagens/carros.mp4')  #  Habilita captura de Webcam/Video


while True:
    ret, image = cap.read()  #  Captura um frame

    if not ret:
        exit()
    black = np.zeros_like(image)  #  Cria imagem preta nas mesmas dimensões do frame.

    ROI = black.copy()  # Copia a imagem preta

    ROI[80:,80:] = image[80:,80:]  # Recorta image a partir de 140 x 140 e cola na cópia da imagem preta ( Segmenta ROI  - Region of Interest )

    gray = cv2.cvtColor(ROI, cv2.COLOR_RGB2GRAY)  # converte ROI para tons de cinza

    gray = cv2.equalizeHist(gray) # Equaliza o histograma da imagem Gray

    fgmask = fgbg.apply(gray)  #  Aplica segmentador de fundo no frame e cria uma máscara binária

    mask_blur = cv2.blur(fgmask, (3,3))  # Aplica Blur à Máscara

    cv2.imshow('mascara',cv2.hconcat([gray,fgmask]))   #  Concatena gray / blur

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)  #  Detecta contornos na máscara

    if len(contours) > 0:

        for (i,contour) in enumerate (contours):

            cv2.drawContours(black, contour, 0, (255,255,255), 5) #  Desenha os contornos na imagem preta

            (x,y,w,h) = cv2.boundingRect(contour)  # detecta grupos de contornos e retorna as posições e dimensões do retângulo da caixa

            contour_valid = (w >= 50) and (h >= 50)  # Considera válido apenas os contornos com dimensão superior a 40 pixels por 40 pixels

            if contour_valid:  # Verifica se o contorno é válido
                cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2, cv2.LINE_AA)  # plota no frame o retângulo ao redor dos contornos válidos

    final = cv2.hconcat([ROI,image])  # concatena a imagem do frame e a imagem dos contornos
    cv2.imshow('Camera',final)  #  Exibe o resultado final

    k = cv2.waitKey(15)
    if k == ord('q'):
        break