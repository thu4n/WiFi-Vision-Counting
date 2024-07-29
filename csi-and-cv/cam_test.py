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
    while True:  
        ret_val, frame = cap.read()
        if not ret_val:
            break

        # Display the frame (optional)
        cv2.imshow('CSI Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
