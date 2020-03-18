# https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=setmousecallback
import cv2
import numpy as np

def get_position(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:   # Se o evento do mouse detectado for o botão esquerdo
        print('x {}, y {} '.format(x,y))

cv2.namedWindow("Distorcida") # Cria uma janela no opencv com o nome 'Distorcida'
cv2.setMouseCallback("Distorcida",get_position)  # Determina que cada ação do mouse na janela 'Distorcida' chame a função 'get_position'

if __name__ == '__main__' : 

    referencia = cv2.imread('imagens/cartao1.jpg')  # Abre a imagem referência de posição

    pts_ref = np.array([[737,59],[327,59],[318,718],[739,718]])  # pontos das extremidades do cartão
    p1  = pts_ref.reshape((-1,1,2))  # Reshape na array para o formato compatível
    print(referencia.shape)

    distorcida = cv2.imread('imagens/cartao2.jpg')
    cv2.imshow("Distorcida", distorcida)
    cv2.waitKey(0)

    cv2.imshow("Distorcida", distorcida)    
    pts_dis = np.array([[85, 474],[298, 112],[726, 426],[615, 733]]) # coordenadas das extremidades do cartão
    p2  = pts_dis.reshape((-1,1,2))

    h, status = cv2.findHomography(pts_dis, pts_ref)  # encontra homografia entre os pontos e retorna array com os parâmetros de distorção

    resultado = cv2.warpPerspective(distorcida, h, (1032,774))  # Aplica distorção na imagem baseado na homografia 'h'    
    cv2.polylines(referencia,[p1],True,(0,255,255))  # Traça linhas na imagem com as extremidades
    cv2.polylines(distorcida,[p2],True,(0,255,255))
    resultado = cv2.flip(resultado,1)
    final = cv2.hconcat([referencia,distorcida,resultado])
    final = cv2.resize(final, None, fx=.5,fy=.5)
    cv2.imshow("Homografia", final)
    cv2.waitKey(0)

    cv2.imshow("Homografia", final)