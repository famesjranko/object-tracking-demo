"""
Author: Andrew J McDonald
Date: 17.07.2023
"""

import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize the global variables
x, y, w, h = 0, 0, 0, 0
drawing = False

# Define a mouse callback function to capture the mouse events
def draw_rectangle(event, x_new, y_new, flags, params):
    global x, y, w, h, drawing
    # Capture the mouse down event and initialize the rectangle drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x, y = x_new, y_new
        
    # Capture the mouse up event and finalize the rectangle drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = x_new - x, y_new - y
        
    # Capture the mouse move event and update the rectangle dimensions
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            w, h = x_new - x, y_new - y

# Create a window and set the mouse callback function
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_rectangle)

# Run an infinite loop to capture the video feed and draw the rectangle
while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()

    # Draw the rectangle on the video feed
    if drawing:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy the window
cap.release()
cv2.destroyAllWindows()
