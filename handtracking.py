import cv2
import mediapipe as mp
import time


cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

pTime,cTime=0,0

while True:
    success,img=cap.read()
    imgRB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id,lm in enumerate(hand.landmark):
                #print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #if id == 0:
                cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                
            mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)
            
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