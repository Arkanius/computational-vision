import cv2
import numpy as np


cap = cv2.VideoCapture('imagens/estabiliza.mp4')


ret,old = cap.read()
old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)

fourcc = cv2.VideoWriter_fourcc(*'XVID') # Cria o objeto para gravar vídeo

out = cv2.VideoWriter('video_estabilizado.mp4', fourcc, 20.0, (old.shape[1] , old.shape[0]))  # Determina o nome do arquivo de saída, sua taxa de FPS e sua resolução.
transforms = [np.identity(3)]


height, width = old.shape[0], old.shape[1]

count = 0

while ret:
    count += 1
    ret,frame = cap.read()
    atual = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pontos_chave_old = cv2.goodFeaturesToTrack(old_gray, 100, 0.0001, 10)

    pontos_chave_old_b = np.int0(pontos_chave_old)
    copy = frame.copy()
    for i in pontos_chave_old_b:
        x,y = i.ravel()
        cv2.circle(copy,(x,y),3,(0,255,255),-1)

    pontos_chave_atual, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, atual, pontos_chave_old, np.array([]))

    pontos_chave_old, pontos_chave_atual = map(lambda pontos_chave_old_b: pontos_chave_old_b[status.ravel().astype(bool)], [pontos_chave_old, pontos_chave_atual])


    transform,_ = cv2.estimateAffinePartial2D(pontos_chave_old, pontos_chave_atual, True)



    height, width = frame.shape[0], frame.shape[1]
    
    last_transform = np.identity(3)

    transform = transform.dot(last_transform)

    transformado = cv2.warpAffine(frame, transform, (width, height))


    inverse_transform = cv2.invertAffineTransform(transform[:2])

    estabilizado = cv2.warpAffine(frame, inverse_transform, (width, height))
    out.write(estabilizado)
    last_transform = transform

    final1 = cv2.hconcat([frame,transformado])
    final2 = cv2.hconcat([copy, estabilizado])
    final = cv2.vconcat([final1,final2])
    cv2.imshow('resultado',final)
    k = cv2.waitKey(1)

    if k == ord('q'):
        exit()
        out.release()
cap.release()
out.release()


