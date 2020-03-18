# Histograma : https://pt.wikipedia.org/wiki/Histograma

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("imagens/mandril_foto.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converte a imagem de BRG para RGB

img2 = cv2.imread("imagens/mandril_foto.png")
bgr_mandril = cv2.split(img2)  # Divide uma imagem de 3 canais em 3 imagens de um canal

fig, ax = plt.subplots(5)  # Cria uma figura com 4 subplots

r = bgr_mandril[2].reshape(img2.shape[0],img2.shape[1]) #  muda o shape da array para imagem


g = bgr_mandril[1].reshape(img2.shape[0],img2.shape[1]) #  muda o shape da array para imagem
b = bgr_mandril[2].reshape(img2.shape[0],img2.shape[1]) #  muda o shape da array para imagem


ax[0].imshow(img)  #  Exibe a imagem
ax[1].imshow(img2)
ax[2].imshow(r, cmap="Reds")  #  Exibe o canal vermelho da imagem
ax[3].imshow(g, cmap="Greens")  # Exibe o canal Verde da imagem
ax[4].imshow(b, cmap="Blues")  # Exibe o canal Azul da imagem

plt.show()

