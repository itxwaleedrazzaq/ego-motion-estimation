import cv2
import numpy as np

vid1 = cv2.VideoCapture('http://192.168.10.51:8081')
vid2 = cv2.VideoCapture('http://192.168.10.51:8081')
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

def stitch(image1,image2):
    kp1,des1 = orb.detectAndCompute(image1,None)
    matches = bf.match(des1,des1)
    stitched = cv2.drawMatches(image1,kp1,image2,kp1,matches[:0],None,flags = 2)
    return stitched

while vid1.isOpened() and vid2.isOpened():
    ret1,left = vid1.read()
    ret2,right = vid2.read()
    stitched = stitch(cv2.resize(left,(640,480)),cv2.resize(right,(640,480)))
    cv2.imshow('result',stitched)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
vid1.release()
vid2.release()

