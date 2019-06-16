import numpy
import cv2
import matplotlib

img = cv2.imread("C:/Users/Marija/Desktop/Zavrsni_rad/Tv111.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("image", thresh)
cv2.waitKey(0)
