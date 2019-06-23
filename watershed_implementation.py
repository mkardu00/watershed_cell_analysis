import cv2
import numpy

img = cv2.imread("C:/Users/Marija/Desktop/Zavrsni_rad/Tv111.png")

# Change color space from blue, green, red to gray. Grayscale image is needed for later
# thresholding.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
cv2.imshow("thresholded", thresh)

# Returns a 3x3 matrix of ones which is later used in the morphological filterings.
kernel = numpy.ones((3, 3), numpy.uint8)

# https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
#
# ---- EROSION ----
# The basic idea of erosion is just like soil erosion, only it erodes away the boundaries of
# foreground object (Always try to keep foreground in white). So what does it do? The kernel slides
# through the image (as in 2D convolution). A pixel in the original image (either 1 or 0) will be
# considered 1 only if all the pixels under the kernel are 1, otherwise it is eroded
# (made to zero). So what happens is that all of the pixels near boundary will be discarded
# depending upon the size of kernel. So the thickness or size of the foreground object decreases or
# simply white region decreases in the image. It is useful for removing small white noises,
# detaching two connected objects etc.
#
# ---- DILATION ----
# It is just opposite of erosion. Here, a pixel element is '1' if at least one pixel under the
# kernel is '1'. So it increases the white region in the image or size of the foreground object
# increases. Normally, in cases like noise removal, erosion is followed by dilation. Because
# erosion removes white noises and also shrinks our object, we dilate it. Since noise is
# gone it won't come back, but our object area increases. It is also useful in joining broken
# parts of an object.
#
# Opening is just another name for erosion followed by dilation.
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("noise_removal", opening)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
cv2.imshow("sure_bg", sure_bg)

# Finding sure foreground area
sure_fg = cv2.erode(opening, kernel, iterations=3)
cv2.imshow("sure_fg", sure_fg)

# Finding unknown region
sure_fg = numpy.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow("unknown", unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
cv2.imshow("watershed", img)

cv2.waitKey(0)
