import cv2
import numpy as np

# Read images
pokemon1 = cv2.imread('pokemon1.jpg')
pokemon2 = cv2.imread('pokemon2.jpg')

# Convert to grayscale
pokemon1_gray = cv2.cvtColor(pokemon1, cv2.COLOR_BGR2GRAY)
pokemon2_gray = cv2.cvtColor(pokemon2, cv2.COLOR_BGR2GRAY)

# Convert to binary 
_, pokemon1_binary = cv2.threshold(pokemon1_gray, 128, 255, cv2.THRESH_BINARY)
_, pokemon2_binary = cv2.threshold(pokemon2_gray, 128, 255, cv2.THRESH_BINARY)

# Create a canvas for displaying the images
height, width = pokemon1_gray.shape
canvas = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

# Fill the canvas with grayscale and binary images
canvas[0:height, 0:width] = cv2.cvtColor(pokemon1_gray, cv2.COLOR_GRAY2BGR)  # Top-left
canvas[0:height, width:width*2] = cv2.cvtColor(pokemon1_binary, cv2.COLOR_GRAY2BGR)  # Top-right
canvas[height:height*2, 0:width] = cv2.cvtColor(pokemon2_gray, cv2.COLOR_GRAY2BGR)  # Bottom-left
canvas[height:height*2, width:width*2] = cv2.cvtColor(pokemon2_binary, cv2.COLOR_GRAY2BGR)  # Bottom-right

# Draw a circle around my favorite Pokemon
favorite_pokemon_center = (int(width * 1.5), int(height * 1.5))  # Center of the circle
cv2.circle(canvas, favorite_pokemon_center, 50, (0, 0, 255), thickness=5)  # Red circle

# Display the image
cv2.imshow('Pokemon Collage', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
