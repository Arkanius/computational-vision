# ORB = Oriented FAST and Rotated BRIEF
# BRIEF = Binary Robust Independent Elementary Features
# ORB detection -> https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html


import numpy as np
import cv2

img = cv2.imread('imagens/vasco.jpg')  # Carrega Imagens
orb = cv2.ORB_create()  # Inicializa ORB

kp = orb.detect(img,None)  #  Encontra pontos-chave com ORB.
kp, des = orb.compute(img, kp) #  Computa os descritores com ORB

cv2.drawKeypoints(img,kp,img,color=None, flags=0)  # Traça pontos-chave em suas localizações.
cv2.imshow('Resultado', img)
cv2.waitKey(0)