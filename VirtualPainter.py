import cv2
import time
import numpy as np 
import os
import handTrackingModule as htm

def pickAndSetColor(x):
    global imgNumber
    colorRange=[(300,400),(500,600),(700,800),(900,1000)]
    color=[BLUE_COLOR,GREEN_COLOR,YELLOW_COLOR,PINK_COLOR]
    
    for i in range(len(colorRange)):
        if x in range(colorRange[i][0],colorRange[i][1]):
            imgNumber=i+1
            return color[i]
        
    return None


folderPath= "VirtualPainter/Header"
myList=os.listdir(folderPath)
myList.sort()

wCam, hCam = 1200,720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
cTime = 0
imgNumber=0


overlayList=[]
for imgPath in myList:
    image= cv2.imread(f"{folderPath}/{imgPath}")
    image=cv2.resize(image,(wCam,100))
    overlayList.append(image)

detector=htm.handDetector(detectionCon=0.75)


#############
# Parameters for drawing

POINTER=8
SECOND_POINTER =12

GREEN_COLOR = (195,254,118)
PINK_COLOR = (254,106,195)
BLUE_COLOR = (95,224,229)
YELLOW_COLOR = (225,221,95)

##############

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img=cv2.resize(img,(wCam,hCam))
    
    overlayImg = overlayList[imgNumber]
    h,w,c=overlayImg.shape
    img[0:h,0:w]=overlayImg

    img=detector.findHands(img,draw=False)
    lmList=detector.findPosition(img, draw=False)
    
    if(len(lmList)!=0):
        
        if(lmList[POINTER][2]<h and lmList[SECOND_POINTER][2]<h):
            distance, cx ,cy = detector.distanceBetweenTwoLandmarks(img,POINTER,SECOND_POINTER)
            cv2.circle(img,(cx,cy),12,(0,0,0),cv2.FILLED)        
            color=pickAndSetColor(cx)       
                    
        else:
            cv2.circle(img,(lmList[POINTER][1],lmList[POINTER][2]),9,(255,0,255),cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.imshow('Image', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
