# warp affine - https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html

import numpy as np
import cv2

def rotate(image, angle, scale = 1.):
    (h, w) = image.shape[:2]

    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

img = cv2.imread("imagens/cartao1.jpg")
for i in range(0,360):
	rotacionada1 = rotate(img, i,1.)
	cv2.imshow("Rotacionada2", rotacionada1)
	cv2.waitKey(1)