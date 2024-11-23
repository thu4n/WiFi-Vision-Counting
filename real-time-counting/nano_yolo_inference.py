import cv2
import torch
import os
import numpy as np
import time
import utils.utils
import model.detector
import pandas as pd
import psutil
import sys
from info_logger import info_logger

# Define resource limits
MAX_CPU_PERCENT = 85  # Maximum CPU usage percentage
MAX_MEMORY_PERCENT = 85  # Maximum memory usage percentage

def check_resources():
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU usage: {cpu_percent}%")
    if cpu_percent > MAX_CPU_PERCENT:
       sys.exit(1)

    # Check memory usage
    memory_info = psutil.virtual_memory()
    print(f"Memory usage: {memory_info.percent}%")
    if memory_info.percent > MAX_MEMORY_PERCENT:
       sys.exit(1)

'''
YOLO Section
'''

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

if __name__ == '__main__':
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    check_resources()
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit(1)
    
    # Load configuration and model
    cfg = utils.utils.load_datafile('./data/coco.data')
    model_path = 'modelzoo/yolofv2-nano-190-epoch-0.953577ap-model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    yolo_model.load_state_dict(torch.load(model_path, map_location=device))
    check_resources()
    yolo_model.eval()
    LABEL_NAMES = ['person']

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                check_resources()
                # Resize frame to model input size
                res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
                img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
                img = torch.from_numpy(img.transpose(0, 3, 1, 2)).to(device).float() / 255.0

                # Model inference
                start = time.perf_counter()
                preds = yolo_model(img)
                end = time.perf_counter()
                print(f"Inference time: {(end - start) * 1000:.2f} ms")
                check_resources()

                # Procescsi_data_ready_events predictions
                output = utils.utils.handel_preds(preds, cfg, device)
                output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

                h, w, _ = frame.shape
                scale_h, scale_w = h / cfg["height"], w / cfg["width"]
                cv_count = 0
                # Draw bounding boxes
                for box in output_boxes[0]:
                    box = box.tolist()
                    obj_score = box[4]
                    category = LABEL_NAMES[int(box[5])]

                    if category == 'person':
                        cv_count += 1
                        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, f'{category} {obj_score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(frame, f"CV count: {cv_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Display result
                cv2.imshow("CSI Camera Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(2)

        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("Stopping YOLO inference")