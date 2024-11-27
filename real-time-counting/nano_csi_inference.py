import os
import numpy as np
import time
import pandas as pd
import serial
import csv
from csi_preprocessor_nano import process_csi_from_csv
import threading
import psutil
import sys
from info_logger import info_logger
from socket_setup import setup_client
import struct

# Nano specific libs
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# Define resource limits
MAX_CPU_PERCENT = 85  # Maximum CPU usage percentage
MAX_MEMORY_PERCENT = 85  # Maximum memory usage percentage

def check_resources(logger):
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU usage: {cpu_percent}%")
    if cpu_percent > MAX_CPU_PERCENT:
       sys.exit(1)

    # Check memory usage
    memory_info = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory_info.percent}%")
    if memory_info.percent > MAX_MEMORY_PERCENT:
       sys.exit(1)

def capture_csi(output_file, stop_event, data_ready_event, inference_done_event, serial_port="/dev/ttyUSB0"):
    print("CSI Write Process: Running")
    # ESP specs
    baud_rate = 921600  # Set the baud rate
    ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=1) # Configure the serial port
   
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
def csi_inference(stream, context, h_input, d_input, h_output, d_output, csi_data):
    print("Running actual inference")
    np.copyto(h_input, csi_data.ravel())

    # Copy input data to the device
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run inference
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    
    # Copy output data to the host
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return h_output

if __name__ == '__main__':
    # Logging
    logger = info_logger()

    # Output directory
    output_dir = f"csi_raw/"
    if not os.path.exists(output_dir): # Check if the folder exists
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir) # Create the folder
    output_file = f"{output_dir}/csi_data.csv" # This file will be continously overwritten

    # Load TensorRT engine
    print("CSI Inference Engine: Starting")
    csi_engine_path = 'modelzoo/haitc_2111_model_fold_2.trt'

    if not os.path.exists(csi_engine_path):
        print("CSI TRT Engine not found")
        exit(1)

    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')
    csi_model = load_engine(csi_engine_path)
    h_input, d_input, h_output, d_output = allocate_buffers(csi_model)
    context = csi_model.create_execution_context()
    stream = cuda.Stream()

    print("CSI Inference Engine: Running")
    print("Setting up socket connection...")
    client_socket = setup_client('192.1268.1.10', 65533)

    # Threading prep
    process_stop_event = threading.Event()
    csi_data_ready_event = threading.Event()
    csi_inference_done_event = threading.Event()

    csi_capture_process = threading.Thread(target=capture_csi, args=(output_file, process_stop_event, csi_data_ready_event, csi_inference_done_event ,"/dev/ttyUSB0"))
    csi_capture_process.start()

    try:
        while not process_stop_event.is_set():
            try:
                print("Waiting for data_ready_event")
                csi_data_ready_event.wait()
                csi_data_ready_event.clear()
                try:
                    with open(output_file, mode="r") as csvfile:
                        csv_reader = csv.reader(csvfile)
                        rows = list(csv_reader)
                        
                        if len(rows) > 1:
                            start = time.perf_counter()
                            processed_csi = process_csi_from_csv(output_file)
                            pred = csi_inference(stream, context, h_input, d_input, h_output, d_output, processed_csi)
                            end = time.perf_counter()

                            logger.info(f"Inference time (preprocessing included): {(end - start) * 1000:.2f} ms")
                            logger.info(f"CSI count: {pred}")

                            packed_data = struct.pack('f', pred)
                            client_socket.sendall(packed_data)
                            print("CSI count sent.")
                            check_resources(logger)
                except Exception as e:
                    print("Super exception when reading csv: ", e)
                    pass

                csi_inference_done_event.set()
            except Exception as e:
                    print("Exception occurred when processing CSI data:", e)
    except KeyboardInterrupt:
        print("Processes stopping...")
        client_socket.close()
        process_stop_event.set()
        csi_capture_process.join()
        print("All processes stopped.")
