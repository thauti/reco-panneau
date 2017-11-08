import numpy as np
import cv2

img = cv2.imread("sample/clean/001.jpg")


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

rouge = np.uint8([[[0,0,255]]])
hsv_red = cv2.cvtColor(rouge, cv2.COLOR_BGR2HSV)
print(hsv_red)

rouge2 = np.uint8([[[176,34,20]]])
hsv_red2 = cv2.cvtColor(rouge2, cv2.COLOR_BGR2HSV)
print(hsv_red2)

upper_red = np.array([255,255,196])
lower_red = np.array([0,150,116])

mask = cv2.inRange(hsv, lower_red, upper_red)

circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('image',img)
cv2.imshow('image2',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
