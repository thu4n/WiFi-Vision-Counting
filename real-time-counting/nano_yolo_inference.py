import cv2
import torch
import numpy as np
import time
import utils.utils
import model.detector
import psutil
import sys
import threading
from info_logger import info_logger

# Define resource limits
MAX_CPU_PERCENT = 90  # Maximum CPU usage percentage
MAX_MEMORY_PERCENT = 90  # Maximum memory usage percentage

# Define the CSI camera pipeline
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=15,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

def check_resources(logger):
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU usage: {cpu_percent}%")
    # if cpu_percent > MAX_CPU_PERCENT:
    #    sys.exit(1)

    # Check memory usage
    memory_info = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory_info.percent}%")
    # if memory_info.percent > MAX_MEMORY_PERCENT:
    #    sys.exit(1)

interval = 2
cv_count = 0
csi_count = 0
last_time = time.time()

def inference(frame):
    global cv_count, last_time
    current_time = time.time()

    if current_time - last_time >= interval:
        # Resize frame to model input size
        res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
        img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).to(device).float() / 255.0

        # Model inference
        start = time.perf_counter()
        preds = model(img)
        end = time.perf_counter()
        logger.info(f"Inference time: {(end - start) * 1000:.2f} ms")
        check_resources(logger)

        # Process predictions
        output = utils.utils.handel_preds(preds, cfg, device)
        output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

        cv_count = 0
        # Draw bounding boxes
        for box in output_boxes[0]:
            box = box.tolist()
            # obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            if category == 'person':
                cv_count += 1

        last_time = current_time

if __name__ == '__main__':
    # Logging
    logger = info_logger()
    # Load configuration and model
    cfg = utils.utils.load_datafile('./data/coco.data')
    model_path = 'modelzoo/coco2017-0.241078ap-model.pth'
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    check_resources(logger)

    # Start CSI camera capture

    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        LABEL_NAMES = [line.strip() for line in f.readlines()]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Frame error found, stopping...")
            break

        threading.Thread(target=inference, args=(frame,)).start()

        print("------------Prediction------------")
        logger.info(f"CV count: {cv_count}")
        cv2.putText(frame, f"CV count: {cv_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #cv2.putText(frame, f"CSI count: {csi_count[0][0]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display result
        cv2.imshow("CSI Camera Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
