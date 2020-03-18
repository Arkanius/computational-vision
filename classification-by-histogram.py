import cv2
import numpy as np
import matplotlib.pyplot as plt

#Função para gerar os histogramas
def get_histograma(src):

    bgr_planes = cv2.split(src)  # Divide uma imagem de 3 canais em 3 imagens de um canal
    histSize = 1024
    histRange = (0, histSize) # the upper boundary is exclusive
  
    #Calculo histograma
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange)  # Calcula Histograma do canal Blue
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange)  # Calcula Histograma do canal Green
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange)  # Calcula Histograma do canal Red

    #Normalização devido a distancia da foto na imagem. Exemplo; O calculo do histograma da cabeça para o corpo pode mudar devido a distancia da foto tirada
    b = b_hist / sum(b_hist)  # Normaliza canal Blue
    r = r_hist / sum(r_hist)  # Normaliza canal Red
    g = g_hist / sum(g_hist)  # Normaliza canal Green

    #Histograma
    histograma = np.array([b,g,r]).reshape(-1,1)  # Concatena as 3 arrays em uma única array

    return histograma, b,g,r
    
    
# Cria plot com 2 linhas e 3 colunas 
fig, ax = plt.subplots(2,3, figsize = (10,15))

#Histograma das imagens dos times originais
vasco = cv2.imread('imagens/vasco.jpg')  # Carrega Imagem
h1,b1,g1,r1 = get_histograma(vasco)  # recebe o histograma da imagem 
ax[0][0].set_title("Vasco")
ax[0][0].plot(h1)  # faz a plotagem do histograma
###ax[0][0].plot(b1, c='b')  # faz a plotagem do histograma canal azul
###ax[0][0].plot(g1, c='g')  # faz a plotagem do histograma canal verde
###ax[0][0].plot(r1, c='r')  # faz a plotagem do histograma canal vermelho



flamengo = cv2.imread('imagens/flamengo.jpg')   # Carrega Imagem
h2,b2,g2,r2 = get_histograma(flamengo)   # recebe o histograma da imagem 
ax[0][1].set_title("Flamengo")    #  Coloca o título no subplot
ax[0][1].plot(h2)   # faz a plotagem do histograma
###ax[0][1].plot(b2, c='b')  # faz a plotagem do histograma canal azul
###ax[0][1].plot(g2, c='g')  # faz a plotagem do histograma canal verde
###ax[0][1].plot(r2, c='r')  # faz a plotagem do histograma canal vermelho



palmeiras = cv2.imread('imagens/palmeiras.jpg')   # Carrega Imagem
h3,b3,g3,r3 = get_histograma(palmeiras)  # recebe o histograma da imagem 
ax[0][2].set_title("Palmeiras")   #  Coloca o título no subplot
ax[0][2].plot(h3)   # faz a plotagem do histograma
###ax[0][2].plot(b3, c='b')  # faz a plotagem do histograma canal azul
###ax[0][2].plot(g3, c='g')  # faz a plotagem do histograma canal verde
###ax[0][2].plot(r3, c='r')  # faz a plotagem do histograma canal vermelho



#Histograma das imagens dos times genericas
vasco_teste = cv2.imread('imagens/vasco_teste.jpg')  # Carrega Imagem
h4,b4,g4,r4 = get_histograma(vasco_teste)  # recebe o histograma da imagem 
ax[1][0].set_title("Vasco Teste")   #  Coloca o título no subplot
ax[1][0].plot(h4)   # faz a plotagem do histograma
###ax[1][0].plot(b4, c='b')  # faz a plotagem do histograma canal azul
###ax[1][0].plot(g4, c='g')  # faz a plotagem do histograma canal verde
###ax[1][0].plot(r4, c='r')  # faz a plotagem do histograma canal vermelho



flamengo_teste = cv2.imread('imagens/flamengo_teste.jpg')  # Carrega Imagem
h5,b5,g5,r5 = get_histograma(flamengo_teste)   # recebe o histograma da imagem 
ax[1][1].set_title("Flamengo Teste")  #  Coloca o título no subplot
ax[1][1].plot(h5)   # faz a plotagem do histograma
###ax[1][1].plot(b5, c='b')  # faz a plotagem do histograma canal azul
###ax[1][1].plot(g5, c='g')  # faz a plotagem do histograma canal verde
###ax[1][1].plot(r5, c='r')  # faz a plotagem do histograma canal vermelho



palmeiras_teste = cv2.imread('imagens/palmeiras_teste.jpg')   # Carrega Imagem
h6,b6,g6,r6 = get_histograma(palmeiras_teste) # recebe o histograma da imagem 
ax[1][2].set_title("Palmeiras Teste")  #  Coloca o título no subplot
ax[1][2].plot(h6)  # faz a plotagem do histograma
###ax[1][2].plot(b6, c='b')  # faz a plotagem do histograma canal azul
###ax[1][2].plot(g6, c='g')  # faz a plotagem do histograma canal verde
###ax[1][2].plot(r6, c='r')  # faz a plotagem do histograma canal vermelho



#Classificação
#Usando o metodo de correlação = 0
r = cv2.compareHist(h1,h1,0)  # Compara o histograma do vasco h1 com ele mesmo
print("Comparação Vasco -> Vasco (mesma imagem) = {} ".format(round(r,2)))


res = cv2.compareHist(h1, h4, 0)  # Compara o histograma do vasco h1 com o do vasco generico h4
print("Comparação Vasco  -> Vasco generico = {}  ".format(round(res,2)))



res2 = cv2.compareHist(h1, h2, 0)  # Compara o histograma do vasco h1 com o do flamengo h2
print("Comparação Vasco  -> Flamengo =  {} ".format(round(res2,2)))


res3 = cv2.compareHist(h1, h3, 0)   # Compara o histograma do vasco h1 com o do palmeiras h3
print("Comparação Vasco  -> Palmeiras = {} ".format(round(res3,2)))

plt.show()