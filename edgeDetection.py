import cv2
import numpy as np

img = cv2.imread('6.png')
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Sobel
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_combind = cv2.magnitude(sobelx, sobely)
sobel_combind = sobel_combind.astype(np.uint8)
_, threshold_img = cv2.threshold(sobel_combind, 50, 255, cv2.THRESH_BINARY_INV)

anime = cv2.bitwise_and(img, img, mask = threshold_img)
canny = cv2.Canny(blurred, threshold1=100, threshold2=200)
_, threshold_img2 = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)

dilation = cv2.dilate(threshold_img2, kernel, iterations=1)

erosion = cv2.erode(threshold_img2, kernel, iterations=1)

opening = cv2.morphologyEx(threshold_img2, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(threshold_img2, cv2.MORPH_CLOSE, kernel)


# Find contours
contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2) 
print(len(contours))


#cv2.imshow("blurr", blurred)
#cv2.imshow("sobel x", sobelx)
#cv2.imshow("sobel y", sobely)
#cv2.imshow("sobel", sobel_combind)
#cv2.imshow("anime", anime)
cv2.imshow("edge", threshold_img2)
cv2.imshow('Canny', canny)
cv2.imshow('contours', result)


cv2.imshow('Dilation', dilation)
cv2.imshow('Erosion', erosion)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)


cv2.waitKey(0)
cv2.destroyAllWindows()

