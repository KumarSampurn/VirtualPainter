import time
import cv2
import mediapipe as mp
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2,modComplex=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modComplex=modComplex
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
    
    
    def findPosition(self,img,handno=0,draw=False):
        
        self.lmList=[]
        
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handno]
            
            for id,lm in enumerate(myhand.landmark):
                    h,w,c = img.shape 
                    cx,cy=int(lm.x*w), int(lm.y*h)
                    self.lmList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),1,(255,0,255),cv2.FILLED)
                    
        return self.lmList
    
    
    def distanceBetweenTwoLandmarks(self,img,landmark1, landmark2):
        x1,y1=self.findPosition(img)[landmark1][1:]
        x2,y2=self.findPosition(img)[landmark2][1:]
        
        cx,cy=(x1+x2)//2, (y1+y2)//2
        distance= math.hypot(x2 - x1, y2 - y1)
        
        return distance,cx,cy
    
    
        
   
    
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img=cv2.flip(img,1)
        img = detector.findHands(img)
        lmList=detector.findPosition(img)
        if(len(lmList)!=0):
            print(lmList[4])
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == "__main__":
    main()