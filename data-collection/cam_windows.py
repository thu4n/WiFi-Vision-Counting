import cv2
import time

# Open the camera (usually 0 for the default camera)
cap = cv2.VideoCapture(0)

# Set the desired frame rate (10 FPS)
desired_fps = 10
time_between_frames = 1 / desired_fps

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Delay to maintain the desired FPS
    time.sleep(time_between_frames)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
