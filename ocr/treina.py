from PIL import ImageFont, ImageDraw, Image
import cv2
import time
import numpy as np
import skimage.measure
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os


#Cria o modelo em regressão logística com o solver SAG

clf = LogisticRegression(random_state=0, solver='sag',multi_class='multinomial', max_iter=100000, verbose=1, tol=0.00001)


X = []
Y = [] 

#Adiciona uma imagem vazia em X e atribui a classe "fundo" em Y para esta imagem
img = np.zeros((50,30,3),np.uint8)
X.append(img)
Y.append("fundo")

#carrega o diretório com as fontes
fonts = os.listdir('fonts/')

#carrega o diretório onde serão gravadas as imagens das fontes
im_folder = os.listdir('images/')

#cria uma pasta vazia para cada letra e número
if len(im_folder) < 1:
    for c in range(48,91):
        if c< 58 or c > 64:
            os.system("mkdir images/"+str(chr(c)))
im_folder = os.listdir('images/')


print(sorted(im_folder))


#Cria uma imagem com cada caracter e número
for c in range(48,91):  
	count = 0
	#ignora os caracteres de 58 a 64 na tabela ASCII
	if c < 58 or c > 64: 
		#plota as letras nas fontes tamanho 40 a 42
		for i in range(40,42):
			for fontpath in fonts: 
				#cria a imagem vazia para plotar 
				img = np.zeros((50,30,3),np.uint8)
				#cor da fonte com alpha transparente
				b,g,r,a = 255,255,255,0
				font = ImageFont.truetype("fonts/"+fontpath, i)
				#converte a imagem para o formato PIL
				img_pil = Image.fromarray(img)
				#prepara a imagem para plotagem
				draw = ImageDraw.Draw(img_pil)
				#plota o texto na imagem com a fonte selecionada
				draw.text((1,1), str(chr(c)), font = font, fill = (b, g, r, a))
				#converte a imagem para o formato openCV
				img = np.array(img_pil)
				#inverte as cores da imagem gerada
				img = np.invert(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
				#salva a imagem em sua respectiva pasta
				cv2.imwrite('images/' + str(chr(c)) + "/" + str(chr(c)) + str(count)+".jpg", img)  
				print("Salvando: " + 'images/' + str(chr(c)) + "/" + str(chr(c)) + str(count)+".jpg") 
				#normaliza [0,1] para a imagem
				img = img.astype(float) / 255.0
				#adiciona imagem em X
				X.append(img)
				#atribui em Y a classe da imagem com ASCII
				Y.append(str(chr(c)))
				count +=1
				#exibe a imagem do caracter
				cv2.imshow("res", img)
				k = cv2.waitKey(60)
				if k ==ord('q'):
					exit()


#Converte X em Array e aplica Reshape em X para o formato correto para treinamento
X = np.array(X).reshape(len(Y),-1)

#converte Y em Array
Y = np.array(Y)
Y = Y.reshape(-1)

#cria o objeto para escala dos vetores
ss = StandardScaler()
print("treinando modelo")

#Normaliza [-1,1] as imagens em X - utilizado por conta da ativação da função sigmoid que vai de - 1 a 1
X = X - (X/127.5)

#Treina o modelo
clf.fit(X, Y)

#aplica escala nos vetores
ss.fit(X)
print("salvando modelo")

#Salva modelo
joblib.dump((clf, ss), "caracteres.pkl", compress=3)
print("modelo salvo")
