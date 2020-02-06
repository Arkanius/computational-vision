import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX 

if __name__=='__main__':

    cap = cv2.VideoCapture('imagens/pacman.webm')
    
    while True:
        ret,frame = cap.read()
        if not ret:
            exit()

        lower = np.array([0,220, 220], dtype=np.uint8)  # Amarelo escuro
        upper = np.array([50, 255, 255], dtype=np.uint8)   # Amarelo Claro

        mask = cv2.inRange(frame, lower, upper) # create the mask - InRange Docs: https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find the countours - Countors doc - https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html

        for cnts in contours:
            (x,y,w,h) = cv2.boundingRect(cnts)
            if w > 25 and h > 25:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, 'x: '+str(x)+' - y: '+str(y), (x+5,y-5), font,.8, [255,255,255], 1, cv2.LINE_AA)
                image = frame[y:y+h,x:x+w]

                cv2.imshow('pacman', image)

        cv2.imshow('frame', frame)

        c = cv2.waitKey(15)
        if c == ord('q'):
            break

    cv2.destroyAllWindows()