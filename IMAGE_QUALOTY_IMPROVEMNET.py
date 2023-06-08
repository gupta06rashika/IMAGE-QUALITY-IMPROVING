import cv2
import numpy as np

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Default values for brightness, contrast, gamma correction, clip limit, and grid size
brightness = 0
contrast = 0
gamma = 0
clip_limit = 1.0
grid_size = 8
white_balance_blue = 100
white_balance_red = 100

# Callback function for brightness trackbar event
def on_brightness_trackbar(val):
    global brightness
    # Update the brightness based on the trackbar value
    brightness = val - 100

# Callback function for contrast trackbar event
def on_contrast_trackbar(val):
    global contrast
    # Update the contrast based on the trackbar value
    contrast = val / 100.0

# Callback function for gamma correction trackbar event
def on_gamma_trackbar(val):
    global gamma
    # Update the gamma correction based on the trackbar value
    gamma = val / 100.0

# Callback function for clip limit trackbar event
def on_clip_limit_trackbar(val):
    global clip_limit
    # Update the clip limit based on the trackbar value
    clip_limit = val / 10.0

# Callback function for grid size trackbar event
def on_grid_size_trackbar(val):
    global grid_size
    if val == 0:
        return
    # Update the grid size based on the trackbar value
    grid_size = val

# Callback function for white balance blue trackbar event
def on_white_balance_blue_trackbar(val):
    global white_balance_blue
    # Update the white balance blue based on the trackbar value
    white_balance_blue = val

# Callback function for white balance red trackbar event
def on_white_balance_red_trackbar(val):
    global white_balance_red
    # Update the white balance red based on the trackbar value
    white_balance_red = val

# Create a named window for trackbars
cv2.namedWindow('Controls')

# Create trackbar for adjusting brightness
cv2.createTrackbar('Brightness', 'Controls', 100, 200, on_brightness_trackbar)

# Create trackbar for adjusting contrast
cv2.createTrackbar('Contrast', 'Controls', 100, 200, on_contrast_trackbar)

# Create trackbar for adjusting gamma correction
cv2.createTrackbar('Gamma', 'Controls', 100, 200, on_gamma_trackbar)

# Create trackbar for adjusting clip limit
cv2.createTrackbar('Clip Limit', 'Controls', int(clip_limit * 10), 50, on_clip_limit_trackbar)

# Create trackbar for adjusting grid size
cv2.createTrackbar('Grid Size', 'Controls', grid_size, 16, on_grid_size_trackbar)

# Create trackbar for adjusting white balance blue
cv2.createTrackbar('White Balance Blue', 'Controls', white_balance_blue, 255, on_white_balance_blue_trackbar)

# Create trackbar for adjusting white balance red
cv2.createTrackbar('White Balance Red', 'Controls', white_balance_red, 255, on_white_balance_red_trackbar)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Adjust the brightness and contrast of the frame
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    # Apply gamma correction to the frame
    gamma_corrected_frame = np.power(adjusted_frame / 255.0, gamma) * 255.0
    gamma_corrected_frame = np.clip(gamma_corrected_frame, 0, 255).astype(np.uint8)

    # Apply white balance to the frame
    white_balanced_frame = cv2.addWeighted(gamma_corrected_frame, white_balance_blue / 255.0, np.zeros_like(gamma_corrected_frame), 0, white_balance_red / 255.0)

    # Convert the frame to LAB color space
    lab = cv2.cvtColor(white_balanced_frame, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Create CLAHE object with the current clip limit and grid size
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    # Apply CLAHE to the L channel
    l_clahe = clahe.apply(l)

    # Merge the processed L channel with the original A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convert the LAB image back to BGR color space
    result_frame = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Create a window with the WINDOW_NORMAL flag
    cv2.namedWindow('CLAHE', cv2.WINDOW_NORMAL)

    # Resize the window to the desired size
    cv2.resizeWindow('CLAHE', 800, 600)

    # Display the frame
    cv2.imshow('CLAHE', result_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
