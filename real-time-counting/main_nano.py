import cv2
import torch
import os
import numpy as np
import time
import utils.utils
import model.detector
import serial
import csv
from csi_preprocessor_nano import process_csi_from_csv
import multiprocessing
import psutil
from info_logger import info_logger

# Nano specific libs
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# Define resource limits
MAX_CPU_PERCENT = 85  # Maximum CPU usage percentage
MAX_MEMORY_PERCENT = 85  # Maximum memory usage percentage

def check_resources(pretext, logger):
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"{pretext} - CPU usage: {cpu_percent}%")

    # Check memory usage
    memory_info = psutil.virtual_memory()
    logger.info(f"{pretext} - Memory usage: {memory_info.percent}%")

'''
CSI Section 
'''

def capture_csi(output_file, stop_event, data_ready_event, inference_done_event, serial_port="/dev/ttyUSB0"):
    print("CSI Write Process: Running")
    # ESP specs
    baud_rate = 921600  # Set the baud rate
    ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=1) # Configure the serial port

    # Expected header of raw CSI data
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
            inference_done_event.wait()
            inference_done_event.clear()
                
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
def csi_inference(stream, context, h_input, d_input, h_output, d_output, csi_data, logger):
    print("Running actual inference")
    np.copyto(h_input, csi_data.ravel())

    # Copy input data to the device
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run inference
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    check_resources("CSI model inference",logger)

    # Copy output data to the host
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return h_output

'''
YOLO Section
'''

def check_dark_frame(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image)
    if brightness < 50:
        return True
    return False

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

def capture_frame(stop_event, frame_queue, logger):
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    check_resources("capture_frame",logger)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)
    cap.release()

def process_yolo(stop_event,csi_count,frame_queue, logger, output_dir):
    print("YOLO Process: Running")
    # Load configuration and model
    cfg = utils.utils.load_datafile('./data/coco.data')
    model_path = 'modelzoo/coco2017-0.241078ap-model.pth'
    device = torch.device("cpu")
    yolo_model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    yolo_model.load_state_dict(torch.load(model_path, map_location=device))
    check_resources("YOLO init",logger)
    yolo_model.eval()
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        LABEL_NAMES = [line.strip() for line in f.readlines()]

    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Resize frame to model input size
            res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
            img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).to(device).float() / 255.0
            is_dark = check_dark_frame(frame)

            # Model inference
            start = time.perf_counter()
            preds = yolo_model(img)
            end = time.perf_counter()
            print(f"Inference time: {(end - start) * 1000:.2f} ms")
            check_resources("YOLO inference", logger)

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
            if csi_count:
                try:
                    cv2.putText(frame, f"CSI count: {csi_count.value}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                except:
                    continue

            save_image(frame, output_dir) # Save image to disk

            print("------------Prediction------------")
            logger.info(f"CV count: {cv_count}")
            logger.info(f"CSI count: {csi_count.value}")
            combined_count = calculate_combined_count(is_dark, cv_count, csi_count.value)
            logger.info(f"Combined CV + CSI count: {combined_count}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    cv2.destroyAllWindows()

'''
Shared section
'''

def calculate_combined_count(is_dark, cv_count, csi_count):
    # In normal condition, Computer Vision will be better
    cv_weight = 0.6
    csi_weight = 0.4   

    # In dim light condition, WiFi Sensing isn't affected so its prediction will be more accurate
    if is_dark:
        cv_weight = 0.3
        csi_weight = 0.7
    combined_count = cv_weight * cv_count + csi_weight * csi_count
    return combined_count

def clean_image_dir(output_dir):
    if os.path.exists(output_dir):
        print(f"Cleaning directory: {output_dir}")
        files = os.listdir(output_dir)
        for file in files:
            os.remove(f"{output_dir}/{file}")

def save_image(frame, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(f"{output_dir}/frame_{time.time()}.jpg", frame)

if __name__ == '__main__':
    # Output image
    result_dir = "results/"
    clean_image_dir(result_dir)

    # Output directory
    output_dir = f"csi_raw/"
    if not os.path.exists(output_dir): # Check if the folder exists
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir) # Create the folder
    output_file = f"{output_dir}/csi_data.csv" # This file will be continously overwritten

    print("CSI Inference Engine: Starting")

    csi_engine_path = '/home/thu4n/1611_model_fold_2.trt'
    if not os.path.exists(csi_engine_path):
        print("CSI TRT Engine not found")
        exit(1)
    
    # Initialize logger
    logger = info_logger()

    # Load TensorRT engine
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')
    csi_model = load_engine(csi_engine_path)
    h_input, d_input, h_output, d_output = allocate_buffers(csi_model)
    context = csi_model.create_execution_context()
    stream = cuda.Stream()

    print("CSI Inference Engine: Running")

    # Events
    process_stop_event = multiprocessing.Event() # Signal to stop all processes
    csi_data_ready_event = multiprocessing.Event() # Called when finish writing raw CSI to a .csv file
    csi_inference_done_event = multiprocessing.Event() # Called when finish CSI preprocessing and model inference

    csi_count = multiprocessing.Value('d', 0.0) # Share count value with YOLO process
    frame_queue = multiprocessing.Queue(maxsize=5) # Put captured images in a queue share between capturing and processing (max 5 frames).

    check_resources("CSI model init",logger)

    # Start processes for collecting CSI data, processing YOLO, and combining outputs
    csi_capture_process = multiprocessing.Process(target=capture_csi, args=(output_file, process_stop_event, csi_data_ready_event, csi_inference_done_event ,"/dev/ttyUSB0"))
    camera_process = multiprocessing.Process(target=capture_frame, args=(process_stop_event, frame_queue, logger))
    yolo_process = multiprocessing.Process(target=process_yolo, args=(process_stop_event, csi_count, frame_queue, logger, result_dir))

    csi_capture_process.start()
    camera_process.start()
    yolo_process.start()

    # The main process will be used for CSI prediction.
    try:
        while True:
            try:
                print("Waiting for data_ready_event")
                csi_data_ready_event.wait()
                csi_data_ready_event.clear()

                with open(output_file, mode="r") as csvfile:
                    csv_reader = csv.reader(csvfile)
                    rows = list(csv_reader)
                    
                    if len(rows) > 1:
                        processed_csi = process_csi_from_csv(output_file)
                        csi_count.value = csi_inference(stream, context, h_input, d_input, h_output, d_output, processed_csi, logger)

                csi_inference_done_event.set()
            except Exception as e:
                    print("Exception occurred when processing CSI data:", e)

    except KeyboardInterrupt:
        print("Processes stopping...")
        process_stop_event.set()
        csi_capture_process.join()
        camera_process.join()
        yolo_process.join()
        print("All processes stopped.")