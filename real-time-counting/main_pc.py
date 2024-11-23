import cv2
import torch
import os
import time
import utils.utils
import model.detector
import tensorflow as tf
import serial
import csv
from csi_preprocessor_pc import process_csi_from_csv
import multiprocessing
from info_logger import info_logger

def calculate_combined_count(is_dark, cv_count, csi_count):
    cv_weight = 0.6
    csi_weight = 0.4   
    if is_dark:
        cv_weight = 0.3
        csi_weight = 0.7
    combined_count = cv_weight * cv_count + csi_weight * csi_count
    return combined_count

def csi_writer(output_file, stop_event, data_ready_event, csi_inference_done_event,serial_port="COM3"):
    print("CSI Thread: Running")
    # ESP specs
    baud_rate = 921600  # Set the baud rate
    ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=0.1) # Configure the serial port
   
    header = [
    "type", "role", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", 
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", 
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", 
    "ant", "sig_len", "rx_state", "real_time_set", "real_timestamp", "len", 
    "CSI_DATA", "machine_timestamp"
    ]
    while not stop_event.is_set():
        with open(output_file, mode="w+", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)

            start_time = time.time()
            frame = 0

            while time.time() - start_time < 2:
                try:
                    data = ser.readline().decode("utf-8").strip()

                    if data.startswith("CSI_DATA") and data.endswith("]"):
                        timestamp = str(time.time())
                        appended_data = data + ',' + timestamp
                        data_list = appended_data.split(',')
                        if(len(data_list) == 27):
                            csv_writer.writerow(data_list)
                            frame += 1

                        if frame == 1:
                            print("--- Start capturing CSI frames ---")

                except Exception as e:
                    print("CSI Data Writing Exception:", {e})
                    pass

            data_ready_event.set()
            print("Waiting for inference_done_event")
            csi_inference_done_event.wait()
            csi_inference_done_event.clear()

def csi_inference(output_file, stop_event, data_ready_event, csi_inference_done_event, csi_count, csi_finish):
    csi_path = 'modelzoo/1611_model_fold_2.keras' 
    csi_model = tf.keras.models.load_model(csi_path)
    try:
        while not stop_event.is_set():
            print("Waiting for data_ready_event")
            data_ready_event.wait()
            data_ready_event.clear()
            with open(output_file, mode="r") as csvfile:
                csv_reader = csv.reader(csvfile)
                rows = list(csv_reader)
                
                if len(rows) > 1:
                    processed_csi = process_csi_from_csv(output_file)
                    csi_count_array = csi_model.predict(processed_csi)
                    csi_count.value = csi_count_array[0][0]
                    csi_finish.value = True

            csi_inference_done_event.set()
    except Exception as e:
            print("Exception occurred when processing CSI data:", e)

def yolo_inference(stop_event, csi_count, csi_finish):
    logger = info_logger()
    print("YOLO Thread: Running")
    # Load configuration and model
    cfg = utils.utils.load_datafile('./data/coco.data')
    model_path = 'modelzoo/coco2017-0.241078ap-model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    yolo_model.load_state_dict(torch.load(model_path, map_location=device))
    yolo_model.eval()
    
    # Start CSI camera capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        LABEL_NAMES = [line.strip() for line in f.readlines()]

    last_time = time.time()
    interval = 2
    cv_count = 0
    combined_count = 0

    while cap.isOpened() and not stop_event.is_set():
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
            logger.info(f"CV count: {cv_count}")
            
            if csi_finish.value:
                combined_count = calculate_combined_count(is_dark=False, cv_count=cv_count, csi_count=csi_count.value)
                logger.info(f"CSI count: {csi_count.value}")
                logger.info(f"Combined CV + CSI count: {combined_count}")
                csi_finish.value = False

        cv2.putText(frame, f"CV count: {cv_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"CSI count: {csi_count.value}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Combined count: {combined_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display result
        cv2.imshow("CSI Camera Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Output directory
    output_dir = f"csi_raw/"
    if not os.path.exists(output_dir): # Check if the folder exists
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir) # Create the folder

    output_file = f"{output_dir}/csi_data.csv" # This file will be continously overwritten

    csi_count = multiprocessing.Value('d', 0.0) # Share count value with YOLO process
    csi_finish = multiprocessing.Value('b', False)
    process_stop_event = multiprocessing.Event()
    csi_data_ready_event = multiprocessing.Event()
    csi_inference_done_event = multiprocessing.Event()

    # Start processes for collecting CSI data, processing YOLO, and combining outputs
    csi_writer_process = multiprocessing.Process(target=csi_writer, args=(output_file, process_stop_event, csi_data_ready_event,csi_inference_done_event, "COM3" ))
    yolo_inference_process = multiprocessing.Process(target=yolo_inference, args=(process_stop_event, csi_count, csi_finish))
    csi_inference_process = multiprocessing.Process(target=csi_inference,args=(output_file, process_stop_event, csi_data_ready_event, csi_inference_done_event, csi_count, csi_finish))

    csi_writer_process.start()
    csi_inference_process.start()
    yolo_inference_process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        process_stop_event.set()
        csi_writer_process.join()
        csi_inference_process.join()
        yolo_inference_process.join()
        print("Program stopped.")
