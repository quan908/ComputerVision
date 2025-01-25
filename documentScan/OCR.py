import cv2 as cv
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog, Text
from PIL import Image, ImageTk

import pytesseract

# Set the tesseract executable path manually
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # Compute width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Construct destination points
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    # Apply perspective transform
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def perform_ocr(image):
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(image)
    return text

def scan_document(image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image.")
    
    # Resize image
    ratio = 1500/image.shape[0] 
    resize_image = cv.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_CUBIC)
    
    # Convert to grayscale and find edges
    gray = cv.cvtColor(resize_image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    
    edged = cv.Canny(gray, 75, 200)
    
    # Find contours
    contours, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]
    
    # Find the document contour
    screen_cnt = None
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break
    
    warped = four_point_transform(resize_image, screen_cnt.reshape(4, 2))
    warp_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    warp_final = cv.adaptiveThreshold(warp_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 10)

    # Perform OCR on the scanned image
    extracted_text = perform_ocr(warp_final)
    
    return warp_final, extracted_text

def open_file():
    # Open file dialog to choose an image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    
    if file_path:
        # Scan the document and extract text
        scanned_image, text = scan_document(file_path)
        
        # Convert image to displayable format (PIL image for Tkinter)
        scanned_image_pil = Image.fromarray(scanned_image)
        scanned_image_pil = scanned_image_pil.convert("RGB")
        
        img_display = ImageTk.PhotoImage(scanned_image_pil)
        
        # Update image in Tkinter window
        panel.config(image=img_display)
        panel.image = img_display
        
        # Display extracted text
        text_box.delete(1.0, tk.END)  # Clear the previous text
        text_box.insert(tk.END, text)  # Insert new extracted text

def create_gui():
    # Create the main window
    window = tk.Tk()
    window.title("Document Scanner with OCR")

    # Create and place the image display panel
    global panel
    panel = tk.Label(window)
    panel.pack()

    # Create and place the text box for displaying extracted text
    global text_box
    text_box = Text(window, height=10, width=50)
    text_box.pack()

    # Create and place the "Open File" button
    open_button = tk.Button(window, text="Open Image", command=open_file)
    open_button.pack()

    # Start the Tkinter main loop
    window.mainloop()

if __name__ == "__main__":
    create_gui()
