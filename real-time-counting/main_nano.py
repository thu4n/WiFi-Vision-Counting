import cv2
import torch
import os
import numpy as np
import time
import utils.utils
import model.detector
import pandas as pd
import serial
import csv
from csi_preprocessor import process_csi_from_csv
import multiprocessing

# Nano specific libs
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

'''
CSI Section 
'''

# Output directory
output_dir = f"csi_raw/"
if not os.path.exists(output_dir): # Check if the folder exists
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir) # Create the folder

output_file = f"{output_dir}/csi_data.csv" # This file will be continously overwritten

def capture_csi(stop_event, serial_port="/dev/ttyUSB0"):
    print("CSI Write Process: Running")
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
    with open(output_file, mode="w+", newline="") as csvfile:
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
                    print("CSI Data Writing Exception:",{e})
                    pass

# Function to load TensorRT engine from a '.engine' file
def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

# Function to allocate device memory and copy data to the device
def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    return h_input, d_input, h_output, d_output

# Function to perform inference with TensorRT engine
def csi_inference(engine, h_input, d_input, h_output, d_output, csi_data):
    stream = cuda.Stream()
    
    np.copyto(h_input, csi_data.ravel())

    # Copy input data to the device
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run inference
    with engine.create_execution_context() as context:
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    
    # Copy output data to the host
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return h_output

def process_csi(stop_event,csi_count):
    print("CSI Inference Process: Starting")
    # Path to the TensorRT engine file
    csi_engine_path = '/home/thu4n/1611_model_fold_2.trt'

    # Load TensorRT engine
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')
    csi_model = load_engine(csi_engine_path)
    h_input, d_input, h_output, d_output = allocate_buffers(csi_model)
    print("CSI Inference Process: Running")
    while not stop_event.is_set():
        with open(output_file, mode="r") as csvfile:
            csv_reader = csv.reader(csvfile)
            rows = list(csv_reader)
            
            if len(rows) > 1:  # Check if there's any new data
                # Get the latest CSI data from the last 2-second collection
                processed_csi = process_csi_from_csv(output_file)
                try:
                    csi_count_array = csi_inference(csi_model, h_input, d_input, h_output, d_output, processed_csi)
                    csi_count.value = csi_count_array[0][0]
                    print("CSI count:", csi_count.value)
                except Exception:
                    print("Exception occured when running CSI inference")

        time.sleep(2)  # Process CSI data every 2 seconds after collecting it
'''
YOLO Section
'''
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=15,
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

def capture_frame(stop_event, frame_queue):
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)
        time.sleep(0.1) # Let the nano breathe a little bit
    cap.release()

def process_yolo(stop_event,csi_count,frame_queue):
    print("YOLO Process: Running")
    # Load configuration and model
    cfg = utils.utils.load_datafile('./data/coco.data')
    model_path = 'modelzoo/yolofv2-nano-190-epoch-0.953577ap-model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    yolo_model.load_state_dict(torch.load(model_path, map_location=device))
    yolo_model.eval()
    # Start CSI camera capture
    # cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    # if not cap.isOpened():
    #     print("Error: Could not open camera.")
    #     return
    LABEL_NAMES = ['person']

    cv_count = 0

    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
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

            cv2.putText(frame, f"CV count: {cv_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if csi_count:
                try:
                    cv2.putText(frame, f"CSI count: {csi_count.value}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                except:
                    continue

            # Display result
            cv2.imshow("CSI Camera Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        time.sleep(2)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    stop_event = multiprocessing.Event()
    csi_count = multiprocessing.Value('d', 0.0) # Share count value with YOLO process
    frame_queue = multiprocessing.Queue(maxsize=5) 

    # Start processes for collecting CSI data, processing YOLO, and combining outputs
    csv_process = multiprocessing.Process(target=capture_csi, args=(stop_event, "/dev/ttyUSB0"))
    csi_process = multiprocessing.Process(target=process_csi, args=(stop_event, csi_count))
    camera_process = multiprocessing.Process(target=capture_frame, args=(stop_event, frame_queue))
    yolo_process = multiprocessing.Process(target=process_yolo, args=(stop_event, csi_count, frame_queue))

    csv_process.start()
    camera_process.start()
    yolo_process.start()
    csi_process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Processes stopping...")
        stop_event.set()
        csv_process.join()
        camera_process.join()
        yolo_process.join()
        csi_process.join()
        print("All processes stopped.")
