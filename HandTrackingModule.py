import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.max = maxHands
        self.detection = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max, self.detection, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw = True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                #print(id,lm)
                h, w, c= img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 25, (255, 0, 255), cv2.FILLED)
        return lmlist

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) !=0:
            print(lmlist[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv2.putText(img,f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow("Img", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()