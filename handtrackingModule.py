import cv2
import mediapipe as mp
import time

#https://google.github.io/mediapipe/solutions/hands#python-solution-api
class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils
        
    def findHands(self,img,draw=True):
        imgRB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,hand,self.mpHands.HAND_CONNECTIONS)
                
            
        return img
    def findPosition(self,img,handNo=0,draw=True):
        lm_list=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                #print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lm_list.append([id,cx,cy])
                
                if draw:
                    cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
        return lm_list

def main():
    cap=cv2.VideoCapture(0)
    detector=HandDetector()
    pTime,cTime=0,0

    while True:
        success,img=cap.read()
        img=detector.findHands(img)
        lm_lists=detector.findPosition(img,draw=False)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,78),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
        
        cv2.imshow("Image",img)
        key=cv2.waitKey(1)
        #ascii for q and Q to quit
        if key==81 or key==113:
            break
    cap.release()
if __name__ == '__main__':
    main()