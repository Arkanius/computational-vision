import cv2

def get_tracker():

  #tracker = cv2.TrackerBoosting_create()
  #tracker = cv2.TrackerMIL_create()
  #tracker = cv2.TrackerKCF_create()
  #tracker = cv2.TrackerTLD_create()
  #tracker = cv2.TrackerMedianFlow_create()
  #tracker = cv2.TrackerGOTURN_create()
  #tracker = cv2.TrackerMOSSE_create()
  tracker = cv2.TrackerCSRT_create()

  return tracker



if __name__ == '__main__':

  cap = cv2.VideoCapture("imagens/aviao.webm") # Criamos o objeto de leitura de vídeo
  _, frame = cap.read()  # Capturamos o primeiro frame
  bb = []  # Criamos uma lista vazia que receberá as coordenadas dos boxes

  while True:

    roi = cv2.selectROI('Frame', frame)  # Função para seleção de ROI
    bb.append(roi)  #  Coordenadas do box da ROI adicionadas à lista


    k = cv2.waitKey(0)
    if k == ord('q'): 
      break


  multiTracker = cv2.MultiTracker_create()  # Cria o objeto Tracker


  for bbox in bb:
    multiTracker.add(get_tracker(), frame, bbox)  # Inicializa o objeto Tracker para cada ROI selecionada



  while True:

    _, frame = cap.read()  # Captura novo frame

    _, bxs = multiTracker.update(frame) # Atualiza o objeto Tracker para a nova posição de cada ROI selecionada

    for ID, box in enumerate(bxs):

      p1 = (int(box[0]), int(box[1]))  # coordenadas do box das detecções
      p2 = (int(box[0] + box[2]), int(box[1] + box[3]))

      cv2.rectangle(frame, p1, p2, (0,255,0), 2, cv2.LINE_AA)  # Retângulo nas áreas detectadas
      cv2.putText(frame, str(ID), (int(box[0]-5),int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)  # Texto com o ID de cada objeto

    cv2.imshow('Frame', frame)
    

    k = cv2.waitKey(1)
    if  k == ord('q'):
      exit()



