from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np

imagem = cv2.imread('imagens/mandril_foto.png')
#imagem = Image.open('imagens/mandril_foto.png').convert('RGB')

img_pil = Image.fromarray(imagem) # Converte para o formato de imagem PIL

draw = ImageDraw.Draw(img_pil)  # Determina que a plotagem será na imagem img_pil

font = ImageFont.truetype("Roboto/Roboto-Regular.ttf", 40)  # Seleciona a fonte através do endereço

draw.text((10, 25), "Mandril", font=font, fill=(0,255,0))   #  Imprime o texto com a cor Verde

imagem_final = np.array(img_pil)   #  Converte para o formato array (opencv)

cv2.imshow("Resultado", imagem_final)   #  imprime o resultado final.

cv2.waitKey(0)  # Aguarda a tecla ser pressionada