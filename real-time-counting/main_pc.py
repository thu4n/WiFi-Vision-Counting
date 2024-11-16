import cv2
import torch
import os
import numpy as np
import time
import utils.utils
import model.detector
import tensorflow as tf
import pandas as pd
import serial
import csv
from csi_preprocessor import process_csi_from_csv
import threading

# Output directory
output_dir = f"csi_raw/"
if not os.path.exists(output_dir): # Check if the folder exists
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir) # Create the folder

output_file = f"{output_dir}/csi_data.csv" # This file will be continously overwritten

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

def capture_csi():
    print("CSI Thread: Running")
    # ESP specs
    serial_port = "COM3"  # Set the serial port number
    baud_rate = 921600  # Set the baud rate
    ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=0.1) # Configure the serial port
   
    header = [
    "type", "role", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", 
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", 
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", 
    "ant", "sig_len", "rx_state", "real_time_set", "real_timestamp", "len", 
    "CSI_DATA", "machine_timestamp"
    ]
    with open(output_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)  
        while True:
            start_time = time.time()  # Start time for 2-second window
            frame = 0 # Initialize the frame counter
            while time.time() - start_time < 2:
                try:
                    data = ser.readline().decode("utf-8").strip()
                    if data.startswith("CSI_DATA") and data.endswith("]"): # Check if the data is a CSI frame
                        timestamp  = str(time.time())
                        appended_data = data + ',' + timestamp 
                        data_list = appended_data.split(',')
                        csv_writer.writerow(data_list)
                        frame += 1

                    if frame == 1:
                        print(f"--- Start capturing CSI frames ---")
                except Exception as e:
                    print("CSI Data Exception:",{e})
                    pass

def process_csi():
    global csi_count
    csi_path = 'modelzoo/1611_model_fold_2.keras' 
    csi_model = tf.keras.models.load_model(csi_path)
    while True:
        with open(output_file, mode="r") as csvfile:
            csv_reader = csv.reader(csvfile)
            rows = list(csv_reader)
            
            if len(rows) > 1:  # Check if there's any new data
                # Get the latest CSI data from the last 2-second collection
                processed_csi = process_csi_from_csv(output_file)
                csi_count = csi_model.predict(processed_csi)
                print("CSI count:", csi_count)

        time.sleep(2)  # Process CSI data every 2 seconds after collecting it

def process_yolo():
    print("YOLO Thread: Running")
    # Load configuration and model
    cfg = utils.utils.load_datafile('./data/coco.data')
    model_path = 'modelzoo/coco2017-0.241078ap-model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    yolo_model.load_state_dict(torch.load(model_path, map_location=device))
    yolo_model.eval()
    # Start CSI camera capture
    #cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
    LABEL_NAMES = ['person']

    last_time = time.time()
    interval = 2
    cv_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()

        if current_time - last_time >= interval:
            # Resize frame to model input size
            res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
            img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).to(device).float() / 255.0

            # Model inference
            start = time.perf_counter()
            preds = yolo_model(img)
            end = time.perf_counter()
            print(f"Inference time: {(end - start) * 1000:.2f} ms")

            # Process predictions
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

            last_time = current_time

        cv2.putText(frame, f"CV count: {cv_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if csi_count:
            try:
                cv2.putText(frame, f"CSI count: {csi_count[0][0]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except:
                continue
        # Display result
        cv2.imshow("CSI Camera Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Start threads for collecting CSI data, processing YOLO, and combining outputs
writer_thread = threading.Thread(target=capture_csi, daemon=True)
yolo_thread = threading.Thread(target=process_yolo, daemon=True)
csi_thread = threading.Thread(target=process_csi, daemon=True)

writer_thread.start()
yolo_thread.start()
csi_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Program stopped.")
