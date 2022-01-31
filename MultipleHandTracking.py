import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import mediapipe

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8,maxHands=2)

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)#,flipType=False)

    #Hand = dict (lmList - bBox - center - type)
    if hands:
        # hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"] #list of 21 landmarks points
        bBox1 = hand1["bbox"] # Bounding box info x,y,w,h
        centerPoint1 = hand1["center"] # center of the hand cx, cy
        handType1 = hand1["type"]  #hand type left or right
        #print(len(lmList1),lmList1)
        #print(bBox1)
        #print(centerPoint1)
        #print(handType1)
        fingers1 = detector.fingersUp(hand1)

        #length, info, img = detector.findDistance(lmList1[8], lmList1[12], img)

        if len(hands)==2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # list of 21 landmarks points
            bBox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2["center"]  # center of the hand cx, cy
            handType2 = hand2["type"]  # hand type left or right

            fingers2 = detector.fingersUp(hand2)
            print(fingers1, fingers2)

            #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img) # distance between tip of the fore finger of two hands

            length, info, img = detector.findDistance(centerPoint1, centerPoint2, img) #distance between center of two hands

    cv2.imshow("Image", img)
    cv2.waitKey(1)