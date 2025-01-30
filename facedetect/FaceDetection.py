#load an image: imread
#scale an image: cv.resize
#crop an image: cv.getRectSubPix
#rotate an image: cv.getRotationMatrix2d,cv.warpAffine
#replace some part of an image: img1[0:300,0:300]=img2

#detect a human face
#located a human face
import cv2 as cv
import numpy as np
import argparse

# Load 
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# Function to rotate the image based on eye positions
def rotate_image(img, eyes):
    eye1, eye2 = eyes[0], eyes[1]
    dY = eye2[1] - eye1[1]
    dX = eye2[0] - eye1[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180  # Calculate the angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated

# Function to detect eyes and align the face
def align_face(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(eyes) >= 2:
        
        eyes = sorted(eyes, key=lambda x: x[0])
        
        left_eye, right_eye = eyes[0], eyes[1]
        left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
        right_eye_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)

        # angle
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))  # 去掉 -180

        # rotate
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

        # check
        if rotated.shape[0] > rotated.shape[1]:  
            rotated = cv.rotate(rotated, cv.ROTATE_180)

        return rotated
    else:
        return img

# Function to process the face 
def process_face(img):
    # Apply Gaussian blur 
    img = cv.GaussianBlur(img, (5, 5), 0)

    # Increase brightness and contrast
    alpha = 1.2  # Contrast control
    beta = 30    # Brightness control
    img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img

# Function to process the user's image and create the ID card
def create_id_card(user_image_path, bg_image_path='bg.png'):
    # Load images
    img = cv.imread(user_image_path)
    bg = cv.imread(bg_image_path)

    if img is None:
        print("Error: Could not load user image. Check the file path.")
        return
    if bg is None:
        print("Error: Could not load background image. Check the file path.")
        return

    # Convert user image to grayscale for face detection
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Check if no faces are detected
    if len(faces) == 0:
        print("Rejected: No face detected in the image.")
        return

    # If multiple faces are detected, choose the largest one
    if len(faces) > 1:
        print("Multiple faces detected. Selecting the largest one.")
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)  # Sort by area (w * h)

    # Get the largest face
    x, y, w, h = faces[0]

    # Crop the detected face
    cropped_face = img[y:y + h, x:x + w]

    # Align the face using eye detection
    aligned_face = align_face(cropped_face)

    # Process the face 
    processed_face = process_face(aligned_face)

    # Resize the cropped face to fit the ID card
    face_width, face_height = 280, 280  
    resized_face = cv.resize(processed_face, (face_width, face_height))

    # Define the position to place the face on the background
    start_x, start_y = 670, 215 
    end_x, end_y = start_x + face_width, start_y + face_height

    # Place the resized face onto the background
    bg[start_y:end_y, start_x:end_x] = resized_face

    # Display the results
    cv.imshow('Detected Face', img)
    cv.imshow('Processed Face', processed_face)
    cv.imshow('ID Card', bg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save the final ID card
    cv.imwrite('id_card_output.png', bg)
    print("ID card created successfully: id_card_output.png")

# Main function to handle command-line arguments
def main() -> None:
    # Construct argument parser
    parser = argparse.ArgumentParser(description='Create an ID card from a user image.')
    parser.add_argument("-i", "--image", required=True, help="Name of the user image file (e.g., user.png)")
    args = parser.parse_args()

    # Call the ID card creation function
    create_id_card(args.image)

if __name__ == "__main__":
    main()