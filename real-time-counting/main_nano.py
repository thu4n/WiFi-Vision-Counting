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
from main_pc import gstreamer_pipeline, capture_csi, process_yolo
import threading

# Nano specific libs
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

# Output directory
output_dir = f"csi_raw/"
if not os.path.exists(output_dir): # Check if the folder exists
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir) # Create the folder

output_file = f"{output_dir}/csi_data.csv" # This file will be continously overwritten

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

def process_csi():
    global csi_count
    # Path to the TensorRT engine file
    csi_engine_path = '/home/1611_model_fold_2.trt'

    # Load TensorRT engine
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')
    csi_model = load_engine(csi_engine_path)
    h_input, d_input, h_output, d_output = allocate_buffers(csi_model)

    while True:
        with open(output_file, mode="r") as csvfile:
            csv_reader = csv.reader(csvfile)
            rows = list(csv_reader)
            
            if len(rows) > 1:  # Check if there's any new data
                # Get the latest CSI data from the last 2-second collection
                processed_csi = process_csi_from_csv(output_file)
                csi_count = csi_inference(csi_model, h_input, d_input, h_output, d_output, processed_csi)
                print("CSI count:", csi_count)

        time.sleep(2)  # Process CSI data every 2 seconds after collecting it

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
