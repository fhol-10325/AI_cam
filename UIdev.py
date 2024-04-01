import cv2
import numpy as np

# Callback function for the mouse events
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), 1)

# Initialize drawing flag and coordinates
drawing = False
ix, iy = -1, -1

# Create a black image
img = np.zeros((512, 1000, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

# Main loop to display the image
while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()
