#Optical Flow - https://docs.opencv.org/trunk/d4/dee/tutorial_optical_flow.html

import os
import numpy as np
import cv2
import time
import math

grava = True
cam = cv2.VideoCapture('imagens/flow.avi')

def draw(flow):

    M = 10
    (h, w) = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((h, w, 3), np.uint8) # Cria uma imagem HSV preta na dimensão do frame - flow
    hsv[..., 0] = ang * (180 / np.pi / 2)  # Parte da Conversão do ângulo em cor
    hsv[..., 1] = 0xFF
    hsv[..., 2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)  # Parte da Conversão do ângulo em cor

    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Converte a imagem hsv em BGR

    return flow_img

if __name__ == '__main__':

    Threshold = 5
    ret, prev = cam.read() # Cria um frame prévio para o cálculo de fluxo ótico
    prev = cv2.resize(prev, None, fx=.4, fy=.4)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Cria o objeto para gravar vídeo
    out = cv2.VideoWriter('out.mp4', fourcc, 20.0, (prev.shape[1] * 2, prev.shape[0]))  # Determina o nome do arquivo de saída, sua taxa de FPS e sua resolução.
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)  # Converte o frame prévio em tons de cinza para o cálculo ótico

    scala = .4
    while True:
        (ret, img) = cam.read()
        img = cv2.resize(img, None, fx=scala, fy=scala)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray,gray,None,0.5,30,Threshold,1,1,1.02,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        prevgray = gray
        flow_img = draw(flow)

        final = cv2.hconcat([img,flow_img])

        if grava == True:
            out.write(final)

        final = cv2.resize(final, None, fx=1*(1/scala), fy=1*(1/scala))
        cv2.imshow('Final', final)

        ch = cv2.waitKey(15)

        if ch == ord('q'):
            out.release()
            cv2.destroyAllWindows()
            exit()



out.release()