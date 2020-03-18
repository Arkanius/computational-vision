import cv2
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image
import os

import random
clf,pp = joblib.load('caracteres.pkl') 
car_cascade = cv2.CascadeClassifier('placa.xml')



bi = 5
cap = cv2.VideoCapture(0)
lista_imagens = ['carro5.png','carro4.png', 'carro3.jpg', 'carro2.jpg'] #['onix.png', 'carro2.jpg','carro3.jpg']

font = cv2.FONT_HERSHEY_SIMPLEX
ratio = .9  # resize ratio
kernel = np.array(([0,1,0],[1,1,1],[0,1,0]),dtype = np.uint8)
kernel2 = np.ones((10,10),np.uint8)

im_folder = os.listdir('images/')
im_folder2 = sorted(im_folder)

while True:
    idx = random.randint(0,len(lista_imagens)-1)
    tess = []
    img = cv2.imread('carros/'+lista_imagens[idx])
    print(lista_imagens[idx])

    black = np.zeros_like(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('gray', gray)
    cars = car_cascade.detectMultiScale(gray, 1.3,2)
    tem = False
    for (x,y,w,h) in cars:
        if h > 30:
            tem = True
    if tem == False:
        cars = car_cascade.detectMultiScale(gray, 1.2,2)
    foi = 0

    ncars  = 0
    x_letter = []
    letter = []
    lit = []
    res = ''
    for (x, y, w, h) in cars:
        if h >30:

            img2 = np.ones_like(gray)

          
            img2[y:y+h,x:x+w] = gray[y:y+h,x:x+w]
            placa = img2[y:y+h,x:x+w]
 
            if True:
                value = (11,11)
                blurred = cv2.GaussianBlur(placa, value, 10)

                thresh = cv2.threshold(blurred,100, 255,cv2.THRESH_BINARY)[1]

                erode = cv2.erode(thresh,kernel,iterations = 7)

                dilate = cv2.dilate(erode,kernel2,iterations = 1)

                contours,hierarchy = cv2.findContours(dilate,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                dilate = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)

                

            else:
                break
            for (i, contour) in enumerate(contours):
                xx,yy,ww,hh = cv2.boundingRect(contour)
                if ww > 30 and ww < 100 and hh > 90:
                    dil = dilate.copy()
                    roi = dil[yy-bi:yy+hh+bi+25,xx-bi:xx+ww+bi+4] 
                    #cv2.imshow('roi'+str(i),roi)
                    roi2 = dil[yy-bi-20:yy+hh+bi+25,xx-bi-20:xx+ww+bi+14] 
                    roi = cv2.resize(roi, (30,50)) 
                    F = np.array(roi).reshape(1,-1)
                    pre = clf.predict(F)
                    score = clf.predict_proba(F)
                    print("Letra - {}  ", pre)
                    print("Score - {}  ", np.argmax(score))


                    msg = str(pre).replace("_"," ").replace("[","").replace("]","").replace("'","")

                    x_letter.append(xx)
                    letter.append(msg) 

            x_c = []
            if len(x_letter) > 1:
                for it in sorted(x_letter):
                    x_c.append(it) 
                    ll = x_letter.index(it)
                    lit.append(ll)
                for sss in lit:
                    res = res + str(letter[sss])
                res = list(res[0:7])
       

            if len(res) > 6:
                if res[0] == "0": res[0]= "O"
                if res[0] == "1": res[0]= "I"
                if res[0] == "2": res[0] = "Z"
                if res[0] == "3": res[0] = "B"
                if res[0] == "4": res[0] = "A"
                if res[0] == "5": res[0] = "S"
                if res[0] == "6": res[0]= "C"
                if res[0] == "7": res[0]= "T"
                if res[0] == "8": res[0] = "B"
                if res[0] == "9": res[0]= "P"

                if res[1] == "0": res[1]= "O"
                if res[1] == "1": res[1]= "I"
                if res[1] == "2": res[1] = "Z"
                if res[1] == "3": res[1] = "R"
                if res[1] == "4": res[1] = "A"
                if res[1] == "5": res[1] = "S"
                if res[1] == "6": res[1]= "R"
                if res[1] == "7": res[1]= "T"
                if res[1] == "8": res[1] = "B"
                if res[1] == "9": res[1]= "P"

                if res[2] == "0": res[2]= "O"
                if res[2] == "1": res[2]= "I"
                if res[2] == "2": res[2] = "Z"
                if res[2] == "3": res[2] = "B"
                if res[2] == "4": res[2] = "A"
                if res[2] == "5": res[2] = "S"
                if res[2] == "6": res[2]= "C"
                if res[2] == "7": res[2]= "T"
                if res[2] == "8": res[2] = "B"
                if res[2] == "9": res[2]= "P"

                if res[3] == "I": res[3]= "1"
                if res[3] == "Z": res[3]= "2"
                if res[3] == "J": res[3]= "3"
                if res[3] == "A": res[3]= "4"
                if res[3] == "S": res[3]= "5"
                if res[3] == "C": res[3]= "6"
                if res[3] == "T": res[3]= "1"
                if res[3] == "B": res[3]= "8"
                if res[3] == "P": res[3]= "9"
                if res[3] == "O": res[3]= "0"
                if res[3] == "U": res[3]= "0"

                if res[4] == "0": res[4]= "O"
                if res[4] == "1": res[4]= "I"
                if res[4] == "2": res[4] = "Z"
                if res[4] == "3": res[4] = "B"
                if res[4] == "4": res[4] = "A"
                if res[4] == "5": res[4] = "S"
                if res[4] == "6": res[4]= "C"
                if res[4] == "7": res[4]= "T"
                if res[4] == "8": res[4] = "B"
                if res[4] == "9": res[4]= "P"

                if res[5] == "I": res[5]= "1"
                if res[5] == "Z": res[5]= "2"
                if res[5] == "J": res[5]= "3"
                if res[5] == "A": res[5]= "4"
                if res[5] == "S": res[5]= "5"
                if res[5] == "C": res[5]= "6"
                if res[5] == "T": res[5]= "1"
                if res[5] == "B": res[5]= "8"
                if res[5] == "P": res[5]= "9"
                if res[5] == "O": res[5]= "0"


                if res[6] == "I": res[6]= "1"
                if res[6] == "Z": res[6]= "2"
                if res[6] == "J": res[6]= "3"
                if res[6] == "A": res[6]= "4"
                if res[6] == "S": res[6]= "5"
                if res[6] == "C": res[6]= "6"
                if res[6] == "T": res[6]= "1"
                if res[6] == "B": res[6]= "8"
                if res[6] == "P": res[6]= "9"
                if res[6] == "O": res[6]= "0"


                res = str(res).replace("_","").replace("[","").replace("]","").replace("'","").replace(",","")

                res = res.replace(" ","")
                mid =int( x+ ((x+w)/2))
                mid2 =int( y+ ((y+h)/2))



                cv2.rectangle(img, (x,y), (x+w,y+h), (50,200,255), 4, cv2.LINE_AA)
                cv2.rectangle(img, (x,y-30), (x+w,y), (50,200,255), 4, cv2.LINE_AA)
                cv2.rectangle(img, (x+5,y+5), (x+w-5,y+h-5), (0,255,255), 4, cv2.LINE_AA)
                cv2.rectangle(img, (x+10,y+10), (x+w-10,y+h-10), (255,255,255), 4, cv2.LINE_AA)
                cv2.rectangle(img, (x-5,y-35), (x+w+5,y+h+5), (0,100,255), 4, cv2.LINE_AA)
                cv2.rectangle(img, (x-10,y-40), (x+w+10,y+h+10), (0,50,255), 4, cv2.LINE_AA)
                cv2.rectangle(img, (x-15,y-45), (x+w+15,y+h+15), (0,0,200), 4, cv2.LINE_AA)
                cv2.rectangle(img, (x,y-30), (x+w,y), (50,200,255), cv2.FILLED)
                cv2.putText(img,"PLACA" ,(mid-120,y-6),font, 0.8, (0,0,0),2, cv2.LINE_AA)
                cv2.putText(img, "RASTREAMENTO DE PLACA" , (50,40), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255),1, cv2.LINE_AA)

                x,y,h,w = 200,300,200,600
                cv2.rectangle(black, (x,y), (x+w,y+h), (50,200,255), 4, cv2.LINE_AA)
                cv2.rectangle(black, (x,y-30), (x+w,y), (50,200,255), 4, cv2.LINE_AA)
                cv2.rectangle(black, (x+5,y+5), (x+w-5,y+h-5), (0,255,255), 4, cv2.LINE_AA)
                cv2.rectangle(black, (x+10,y+10), (x+w-10,y+h-10), (255,255,255), 4, cv2.LINE_AA)
                cv2.rectangle(black, (x-5,y-35), (x+w+5,y+h+5), (0,100,255), 4, cv2.LINE_AA)
                cv2.rectangle(black, (x-10,y-40), (x+w+10,y+h+10), (0,50,255), 4, cv2.LINE_AA)
                cv2.rectangle(black, (x-15,y-45), (x+w+15,y+h+15), (0,0,200), 4, cv2.LINE_AA)
                cv2.rectangle(black, (x,y-30), (x+w,y), (50,200,255), cv2.FILLED)

                cv2.putText(black, "RECONHECIMENTO DE CARACTERES" , (50,40), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255),1, cv2.LINE_AA)
                print("H ->", h)
                print("w ->", w)


                b,g,r,a = 255,255,255,0
                fontq = ImageFont.truetype("fonts/FE-FONT.TTF", 100)
                img_pil = Image.fromarray(black)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x+50, y+int(h/4)), res, font = fontq, fill = (b, g, r, a))
                black = np.array(img_pil)
                
    if img.shape[1] > 1000: 
        ratio = .5
    else:
        ratio = .8

    img2 = cv2.resize(cv2.hconcat([img,black]), (900, 300))
    cv2.imshow('Resultado', img2 )
    res = ""
    x_letter = []
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()



