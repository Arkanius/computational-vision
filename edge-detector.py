import cv2
import numpy as np

cap = cv2.VideoCapture('imagens/chroma1.mp4')

blur = (3,3)

convolution_matrix = np.array([[1,0,-1],
                               [1,0,-1],
                               [1,0,-1]])

while True:
    ret,frame = cap.read()

    if not ret:
        exit()

    filter_2d = cv2.filter2D(frame,-1, convolution_matrix) #https://en.wikipedia.org/wiki/Kernel_%28image_processing%29

    #cv2.imshow('convolução',filter_2d)
    
    frame_copy = frame.copy()

    img_blur = cv2.blur(frame_copy, blur) # https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html

    #cv2.imshow('blur',img_blur)

    src_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    detected_edges = cv2.Canny(src_gray, 50, 150, 3)  #https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html

    edges = cv2.cvtColor(detected_edges, cv2.COLOR_GRAY2BGR)

    #cv2.imshow('edges_canny',edges) # Exibe a imagem com efeito Canny

    #cv2.imshow("edges",edges)
    concat = cv2.vconcat([filter_2d,edges])

    final = cv2.resize(concat, None, fx=0.5,fy=0.5)

    cv2.imshow("Filter 2D", final)

    k = cv2.waitKey(1) 

    if k == ord('q'):
        exit()	