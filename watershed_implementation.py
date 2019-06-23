import cv2

img = cv2.imread("C:/Users/Marija/Desktop/Zavrsni_rad/Tv111.png")

# Change color space from blue, green, red to gray. Grayscale image is needed for later
# thresholding.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("bgimage", gray)

# Apply thresholding to the grayscale image. If the pixel value is greater than the threshold
# value, it is assigned one value (e.g. white), else it's assigned another value (e.g. black).
# First argument is the image source, second one is the threshold value which is used to classify
# pixel values. Third argument is maximum value which represents the value to be given if the pixel
# value is more (or less) than the threshold value. Fourth parameter is the type of thresholding.
# Since the only way to get a good threshold value is trial and error, it's usually more accurate
# to use Otsu binarization which automatically calculates the optimal threshold value and returns
# it as a first return value. The second return value is the thresholded image. When using the Otsu
# binarization, we pass zero as the threshold value (since it will be automatically calculated for
# us).
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("image", thresh)
cv2.waitKey(0)
