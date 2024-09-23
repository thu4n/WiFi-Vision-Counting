#!/usr/bin/python3
import cv2
import os
import time


def capture_image(time_duration, person_num):
    frame_count = 0
    camera = cv2.VideoCapture(0)
    start_time = time.time()
    duration = time_duration if time_duration else 10 # capture duration in seconds
    output_dir = f"cv_dev/{person_num}_person"
    desired_fps = 10
    time_between_frames = 1 / desired_fps
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:  
        ret_val, frame = camera.read()
        if not ret_val:
            break

        if time.time() - start_time > duration:
            print(f"Picture frames for {person_num} persons: {frame_count} frames.")
            break

        frame_filename = os.path.join(output_dir, f"frame_{time.time()}.jpg") 
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        time.sleep(time_between_frames)

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        try:
            capture_image(10, 1)

        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == "__main__":
    main()