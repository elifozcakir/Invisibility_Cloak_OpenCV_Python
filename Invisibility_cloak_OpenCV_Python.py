import cv2
import numpy as np

##reading from the webcam
cap = cv2.VideoCapture(0)
## Capture the background 
ret, backplane = cap.read()

#read every frame
while True:
    ret, img = cap.read()
    ## Convert the color space from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ##Generate masks for blue color detection
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([120,255,255])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    low_blue = np.array([10,50,50])
    up_blue = np.array([140,255,255])
    mask2 = cv2.inRange(hsv, low_blue, up_blue)

    mask1 = mask1 + mask2
    #cv2.imshow("binary", mask1)

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    ## Create an inverted mask
    mask2 = cv2.bitwise_not(mask1)

    ## Segmentation
    im1 = cv2.bitwise_and(img, img, mask=mask2)

    im2 = cv2.bitwise_and(backplane, backplane, mask=mask1)

    #Generation final image
    final_img = cv2.addWeighted(im1, 1, im2, 1, 0)
    cv2.imshow("invisibility cloak", final_img)
    #For quit, press q button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

