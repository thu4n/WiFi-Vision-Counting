import cv2
import torch
import numpy as np
import time
import utils.utils
import model.detector
import psutil
import sys
from info_logger import info_logger
from socket_setup import setup_server

# Define resource limits
MAX_CPU_PERCENT = 90  # Maximum CPU usage percentage
MAX_MEMORY_PERCENT = 90  # Maximum memory usage percentage

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
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    #cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Failed to open CSI camera")
    #     exit()

    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        LABEL_NAMES = [line.strip() for line in f.readlines()]

    last_time = time.time()
    interval = 2
    cv_count = 0
    csi_count = 0

    conn, addr = setup_server('192.168.1.10', '65533')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()

        if current_time - last_time >= interval:
            #csi_count = csi_model.predict(df)

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

            # h, w, _ = frame.shape
            # scale_h, scale_w = h / cfg["height"], w / cfg["width"]
            cv_count = 0
            # Draw bounding boxes
            for box in output_boxes[0]:
                box = box.tolist()
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]

                if category == 'person':
                    cv_count += 1
                    # x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                    # x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                    #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    #cv2.putText(frame, f'{category} {obj_score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            last_time = current_time

            print("------------Prediction------------")
            logger.info(f"CV count: {cv_count}")

        # cv2.putText(frame, f"CV count: {cv_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #cv2.putText(frame, f"CSI count: {csi_count[0][0]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display result
        # cv2.imshow("CSI Camera Detection", frame)

        csi_count = conn.recv(1024)
        if not csi_count:
            continue
        else:
            print("Received:", csi_count)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
