# ORB detection -> https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
# Feature Matching + Homography to find Objects - > https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
# Flann Matching -> https://docs.opencv.org/master/d5/d6f/tutorial_feature_flann_matcher.html
# Knn algorithm -> https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
import cv2
import numpy as np


def main():


    detector = cv2.ORB_create()

    imagem = cv2.imread('imagens/vasco.jpg')
    imagem = cv2.resize(imagem, None,fx=.7, fy=.7)

    cv2.imshow('main', imagem)

    compara = cv2.imread("imagens/vasco.jpg")
    compara = cv2.resize(compara, None,fx=.8, fy=.8)
    compara = cv2.flip(compara,0)

    # Find the keypoints and descriptors with ORB.

    kp1, des1 = detector.detectAndCompute(compara, None)
    kp2, des2 = detector.detectAndCompute(imagem, None)

    matcher = cv2.FlannBasedMatcher()

    des1 = np.array(des1, np.float32)
    des2 = np.array(des2, np.float32)

    matches = matcher.knnMatch(des1, des2, k=2)  # Detecta similaridade entre os descritores  e cria duas classes.

    aprovadas = []
    thresh  = .7
    for m, n in matches:
        if m.distance < thresh * n.distance:  # compara a distância entre as duas classes do KNN
            aprovadas.append(m)  # se a distância for menor que a aceitável, adiciona à lista 'aprovadas'


    src_pts = np.float32([kp1[m.queryIdx].pt for m in aprovadas]).reshape(-1, 1, 2)  # Cria array com o valor de index dos pontos que combinam em kp1
    src2_pts = np.float32([kp2[m.trainIdx].pt for m in aprovadas]).reshape(-1, 1, 2)  # Cria array com o valor de index dos pontos que combinam em kp2

    H, mask = cv2.findHomography(src_pts, src2_pts, cv2.RANSAC) # Cria máscara a partir dos pontos-chave que combinam segundo o algoritmo KNN e valores da distorção da homografia em H

    if mask is not None:

        combinando = mask.ravel().tolist()  #  Converte a array máscara para list

        h, w, c = compara.shape  #  Converte a array [h,w,c] em inteiros.

        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, H)

        params = dict(matchColor=None,  # None = Cores aleatórias ou no formato bgr [0,255,0] (verde)
                            singlePointColor=None,  #  None = cores aleatórias, True = cor preta
                            matchesMask=combinando,  # matchesMask=None -> Determina que Plote todos os pontos |  matchesMask=combinando -> Apenas a máscara gerada por KNN
                            flags=0)   # flags = 2 -> plota círculos apenas nos pontos que combinam, flags = 0 -> Plota círculos nos pontos que não combinam, mas relaciona apenas os que combinam segundo a máscara KNN.

        resultado = cv2.drawMatches(imagem, kp2, compara, kp1, aprovadas, None, **params) #  plota as combinações nas imagens e salva na array 'result'
 
    cv2.imshow('resultado', resultado)   #  Exibe a imagem resultado

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()