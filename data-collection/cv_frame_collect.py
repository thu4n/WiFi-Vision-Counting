#!/usr/bin/python3
import cv2
import os
import time
from data_clean import clear_directory
from sound_alert import camera_start_sound, end_sound

def capture_image(session, time_duration, person_num):
    # Camera specs
    camera = cv2.VideoCapture(0)
    desired_fps = 10
    time_between_frames = 1 / desired_fps
    frame_count = 0
    duration = time_duration if time_duration else 10 # capture duration in seconds

    # Output directory
    output_dir = f"cv_main/session_{session}/{person_num}_person"

    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)
    else:
        clear_directory(output_dir)

    print(f"--- Start capturing picture frames for {person_num} person(s) ---")
    start_time = time.time()
    camera_start_sound()
    while True:  
        ret_val, frame = camera.read()
        if not ret_val:
            break

        if time.time() - start_time > duration:
            break

        frame_filename = os.path.join(output_dir, f"frame_{time.time()}.jpg") 
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        time.sleep(time_between_frames)

    camera.release()
    end_sound()
    print(f"Picture frames for {person_num} persons: {frame_count} frames.")

def main():
    capture_image(1, 10, 1)

if __name__ == "__main__":
    main()