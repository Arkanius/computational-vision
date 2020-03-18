# https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=setmousecallback
import cv2
import numpy as np


cap = cv2.VideoCapture('imagens/carros.mp4')
def rotate(image, angle, scale = 1.):
    (h, w) = image.shape[:2]

    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated
def get_position(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:   # Se o evento do mouse detectado for o botão esquerdo
        print('x {}, y {} '.format(x,y))



cv2.namedWindow("Distorcida") # Cria uma janela no opencv com o nome 'Distorcida'
cv2.setMouseCallback("Distorcida",get_position)  # Determina que cada ação do mouse na janela 'Distorcida' chame a função 'get_position'

referencia = cap.read()[1]
cv2.imshow('Distorcida', referencia)
cv2.waitKey(0)
while True: 

    distorcida = cap.read()[1]

    pts_ref = np.array([[200,400],[200,0],[400,0],[400,400]])  # pontos das extremidades do cartão
    p1  = pts_ref.reshape((-1,1,2))  # Reshape na array para o formato compatível




    #cv2.imshow("referencia", referencia)





    pts_dis = np.array([[31,264],[480,29],[586,34],[466,365]])# coordenadas das extremidades do cartão
    p2  = pts_dis.reshape((-1,1,2))


    h, status = cv2.findHomography(pts_dis, pts_ref)  # encontra homografia entre os pontos e retorna array com os parâmetros de distorção


    resultado = cv2.warpPerspective(distorcida, h, (distorcida.shape[1],distorcida.shape[0]))  # Aplica distorção na imagem baseado na homografia 'h'    
    cv2.polylines(distorcida,[p2],True,(0,255,255))  # Traça linhas na imagem com as extremidades
    cv2.polylines(distorcida,[p1],True,(0,255,0))

    
    #resultado = rotate(resultado,90)
    cv2.imshow('resultado',resultado)
    cv2.imshow("Distorcida", distorcida)    

    k = cv2.waitKey(1)
    if k == ord('q'):
        exit()

 
