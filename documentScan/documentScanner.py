import cv2 as cv
import numpy as np
import argparse

def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
        return rect
    
def four_point_transform(image,pts):

     rect = order_points(pts)
     (tl, tr, br, bl) = rect
     # Compute width and height
     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl [1]) ** 2))
     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
     maxWidth = max(int(widthA), int(widthB))
     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
     maxHeight = max(int(heightA), int(heightB))
     # Construct destination points
     dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1,maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
     # Apply perspective transform
     M = cv.getPerspectiveTransform(rect, dst)
     warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
     return warped

def main() -> None:
    # Construct argument parser
    parser = argparse.ArgumentParser(description='Scan a document from an image.')
    parser.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    args = parser.parse_args()

    # Load and preprocess image
    image = cv.imread(args.image)
    if image is None:
        raise ValueError("Could not read the image.")


    
    
    # Resize the image
    ratio = 1500/image.shape[0] 
    orig = image.copy()
    resize_image = cv.resize(image, (0, 0), fx=ratio, fy=ratio,
                      interpolation = cv.INTER_CUBIC)
    # Convert to grayscale and find edges
    gray = cv.cvtColor(resize_image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    cv.imshow('gray',gray)
    edged = cv.Canny(gray, 75, 200)
    cv.imshow('edged',edged)

    #Find CONTOURS
    contours, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse= True)[:5]
    # Find the document contour
    screen_cnt = None
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break
    cv.drawContours(resize_image, [screen_cnt], -1, (0, 255, 0), 2)
    cv.imshow("Outline", resize_image)

    
    warped = four_point_transform(resize_image,screen_cnt.reshape(4, 2)) 
          
  
    cv.imshow("scanned", warped)
 
    cv.waitKey(0)
    cv.destroyAllwindows()

if __name__ == '__main__':
    main()