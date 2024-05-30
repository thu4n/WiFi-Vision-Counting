#!/usr/bin/python3
import cv2
import os
import time
import sys

# Define the GStreamer pipeline for the CSI camera
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=20,  # Set framerate to 20 fps
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def capture_image():
    frame_count = 0
    start_time = time.time()
    duration = 10  # capture duration in seconds

    while True:  
        ret_val, frame = cap.read()
        if not ret_val:
            break

        # Save the frame as an image file
        frame_filename = os.path.join(output_dir, f"frame_{time.time()}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        # Display the frame (optional)
        #cv2.imshow('CSI Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if the duration has passed
        if time.time() - start_time > duration:
            print(f"Frames: {frame_count-1}")
            break

num = sys.argv[1] if len(sys.argv) > 1 else 1

output_dir = f"captured_frames_{num}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the camera
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

if cap.isOpened():
    try:
        capture_image()

    finally:
        cap.release()
        cv2.destroyAllWindows()
else:
    print("Unable to open camera")
