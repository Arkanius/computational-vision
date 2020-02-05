import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX 

if __name__=='__main__':

    cap = cv2.VideoCapture('imagens/pacman.webm')  # Captura Vídeo
    
    while True:
        ret,frame = cap.read() # Captura um frame do video
        if not ret:  # Verifica status do vídeo
            exit()

        lower = np.array([0,220, 220], dtype=np.uint8)  # (Amarelo escuro) Determina o limite inferior [daqui para cima] 
        upper = np.array([50, 255, 255], dtype=np.uint8)   # (Amarelo Claro) Determina o limite superior [daqui para baixo]

        mask = cv2.inRange(frame, lower, upper) # Cria máscara a partir dos limites LOWER-UPPER
        # Documentação sobre InRange: https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html

        # cv2.imshow('mascara',mask)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Encontra os contornos na máscara
        # Documentação sobre Contours: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html

        
        for cnts in contours:
            (x,y,w,h) = cv2.boundingRect(cnts) # Cria retângulos com os limites dos contornos
            if w > 25 and h > 25:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4, cv2.LINE_AA) # imprime o retângulo no Frame

                cv2.putText(frame, 'x: '+str(x)+' - y: '+str(y), (x+5,y-5), font,.8, [255,255,255], 1, cv2.LINE_AA) # Imprime o texto das coordenadas
                imagem_cortada = frame[y:y+h,x:x+w]

                cv2.imshow('imagem cortada', imagem_cortada) # Segmenta e exibe a área da detecção em uma imagem menor

        cv2.imshow('frame', frame) # Exibe o resultado

        c = cv2.waitKey(15)    # Aguarda tecla ser pressionada por determinad tempo
        # documentação sobre WaitKey: https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=waitkey

        if c == ord('q'):
            break

    cv2.destroyAllWindows()