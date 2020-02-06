import cv2
import numpy as np
import time

if __name__=='__main__':

    cap = cv2.VideoCapture('imagens/chroma1.mp4')
    while cap:
        background = cv2.imread('imagens/fundo.jpg')

        ret,frame = cap.read() # read frame by frame
        if not ret:
           exit()
        
        background = cv2.resize(background, (frame.shape[1],frame.shape[0]))  # resizes the background image to the same size of video

        lower = np.array([0, 150, 0], dtype=np.uint8)
        upper = np.array([100, 255, 100], dtype=np.uint8)

        mask = cv2.inRange(frame, lower, upper)
        #cv2.imshow('mask', mask) # white mask

        background_process = cv2.bitwise_and(background, background, mask=mask)  # remove the intersection of both: mask and background

        # cv2.imshow('background_process', background_process) #changed the green background to the image background

        invert_mask = np.invert(mask) # invert the color (like invert selection of image softwares)

        # cv2.imshow('invert_mask', invert_mask) # inverted mask - black

        frame_process = cv2.bitwise_and(frame, frame, mask=invert_mask) # proccess the frame from mask, removing the commom area (white part)

        # cv2.imshow('frame_process', frame_process) # intersection of inverted mask and the original video

        final = cv2.addWeighted(background_process, 1, frame_process, 1, 0)  # add the image to the bg of video: https://docs.opencv.org/4.0.1/d5/dc4/tutorial_adding_images.html

        cv2.imshow('final', final)

        c = cv2.waitKey(5)

        if c == ord('q'):
            break

    cv2.destroyAllWindows()

