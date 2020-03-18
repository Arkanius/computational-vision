# Referência: https://docs.opencv.org/trunk/d4/dee/tutorial_optical_flow.html
import os
import numpy as np
import cv2
import time
import math



grava = True
cam = cv2.VideoCapture('imagens/flow.avi')
fgbg = cv2.createBackgroundSubtractorKNN(history=10, detectShadows=False)  #  Cria segmentador de fundo

def draw(flow):
    M = 10
    (h, w) = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1]) # Converte a detecção de fluxo em ângulo
    hsv = np.zeros((h, w, 3), np.uint8) # Cria uma imagem HSV preta na dimensão do frame - flow
    hsv[..., 0] = ang * (180 / np.pi / 2)  # Conversão do ângulo em cor
    hsv[..., 1] = 0xFF
    hsv[..., 2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)  # Normaliza o canal 


    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Converte a imagem hsv em BGR


    return flow_img


if __name__ == '__main__':

    Threshold = 5
    scale = .6
    x1, x2 = int(120 * scale),int(480 * scale)
    y1,y2 = int(120 * scale),int(480 * scale)

    ret, prev = cam.read() # Cria um frame prévio para o cálculo de fluxo ótico

    prev = cv2.resize(prev, None, fx=scale, fy=scale)

    ROI = np.zeros_like(prev)

    ROI[y1:y2,x1:x2] = prev[y1:y2,x1:x2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Cria o objeto para gravar vídeo

    out = cv2.VideoWriter('out.mp4', fourcc, 20.0, (prev.shape[1]*2, prev.shape[0]))  # Determina o nome do arquivo de saída, sua taxa de FPS e sua resolução.

    prevgray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)  # Converte o frame prévio em tons de cinza para o cálculo ótico

    prevgray = cv2.equalizeHist(prevgray)
    prevgray = cv2.blur(prevgray, (3,3))
    prevgray = fgbg.apply(prevgray)  #  Aplica segmentador de fundo no frame e cria uma máscara binária


    while True:
        ret, img = cam.read()
        if not ret:
            out.release()
            exit()
        img = cv2.resize(img, None, fx=scale, fy=scale)

        copy = img.copy()
        ROI = np.zeros_like(img)
        ROI[y1:y2,x1:x2] = img[y1:y2,x1:x2]

        gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        gray = fgbg.apply(gray)  #  Aplica segmentador de fundo no frame e cria uma máscara binária
        gray = cv2.blur(gray, (3,3))

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0),6, cv2.LINE_AA)
        cv2.rectangle(img, (x1,y1), (x2,y2), (100,255,255),4, cv2.LINE_AA)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0),1, cv2.LINE_AA)

        flow = cv2.calcOpticalFlowFarneback(prevgray,gray,None,0.5,30,Threshold,1,1,1.02,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        prevgray = gray

        flow_img = draw(flow)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)  #  Detecta contornos na máscara

        if len(contours) > 0:

            for (i,contour) in enumerate (contours):


                (x,y,w,h) = cv2.boundingRect(contour)  # detecta grupos de contornos e retorna as posições e dimensões do retângulo da caixa

                contour_valid = (w >= 50) and (h >= 30)  # Considera válido apenas os contornos com dimensão superior a 40 pixels por 40 pixels

                carros1 = cv2.bitwise_and(flow_img,flow_img, mask=gray)  # Utiliza a máscara para segmentar a detecção de fluxo
                carros2 = cv2.bitwise_and(copy,copy, mask=gray)  # Utiliza a máscara para segmentar a detecção de fluxo

                carros = cv2.addWeighted(carros1,1,carros2,1,0)



                if contour_valid:  # Verifica se o contorno é válido

                    img[y:y+h,x:x+w] =  cv2.addWeighted(flow_img[y:y+h,x:x+w],1,img[y:y+h,x:x+w],1,0) #  combina as áreas da detecção de fluxo na imagem do frame
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255),1, cv2.LINE_AA)

        final = cv2.hconcat([img,carros])
        if grava == True:
            out.write(final)

        final = cv2.resize(final, None, fx=1*(1/scale), fy=1*(1/scale))
        cv2.imshow('Final', final)



        ch = cv2.waitKey(1)

        if ch == ord('q'):
            out.release()
            cv2.destroyAllWindows()
            exit()



out.release()