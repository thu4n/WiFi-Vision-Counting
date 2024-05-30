#!/usr/bin/python3
import serial
import time

# def inference(engine, h_input, d_input, h_output, d_output, input_data):
#     stream = cuda.Stream()
#     input_data = np.array(input_data)
#     input_data /= 255
#     # Set the input data (dummy data in this example)
#     np.random.seed(123)
#     #input_data = np.random.rand(*h_input.shape).astype(np.float32)
#     np.copyto(h_input, input_data.ravel())

#     # Copy input data to the device
#     cuda.memcpy_htod_async(d_input, h_input, stream)

#     # Run inference
#     with engine.create_execution_context() as context:
#         context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    
#     # Copy output data to the host
#     cuda.memcpy_dtoh_async(h_output, d_output, stream)
#     stream.synchronize()

#     return h_output

# def process_data(data):
#     # Add your data processing logic here
#     print(data)

# def process(res):
#     # Parser
#     all_data = res.split(',')
#     csi_data = all_data[25].split(" ")
#     csi_data[0] = csi_data[0].replace("[", "")
#     csi_data[-1] = csi_data[-1].replace("]", "")

#     csi_data.pop()
#     csi_data = [int(c) for c in csi_data if c]
#     imaginary = []
#     real = []
#     for i, val in enumerate(csi_data):
#         if i % 2 == 0:
#             imaginary.append(val)
#         else:
#             real.append(val)

#     csi_size = len(csi_data)
#     amplitudes = []
#     if len(imaginary) > 0 and len(real) > 0:
#         for j in range(int(csi_size / 2)):
#             amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
#             amplitudes.append(amplitude_calc)
#     df = pd.DataFrame(amplitudes)
#     return df

def main():
    serial_port = "/dev/ttyUSB0"  # Use the specified serial port
    baud_rate = 921600  # Set the baud rate to 921600
    # count = 0
    # Configure the serial port
    ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=0.1)
    dfs = []

    #engine_file_path = '3act_cnn.engine'
    # Load TensorRT engine
    #trt_logger = trt.Logger(trt.Logger.INFO)
    #trt.init_libnvinfer_plugins(trt_logger, '')
    #engine = load_engine(engine_file_path)
    #h_input, d_input, h_output, d_output = allocate_buffers(engine)
    try:
        duration = 10
        frame_count = 0
        start_time = time.time()
        while True:
            try:
                data = ser.readline().decode("utf-8").strip()
                if "CSI_DATA" in data:
                    print(data.strip() + "," + str(time.time()))
                    frame_count += 1
                if time.time() - start_time > duration:
                    print(f"Frames: {frame_count-1}")
                    break
                    # df = process(data)
                    # df_transposed = df.transpose() 
                    #print(df_transposed.shape)
                    # if df_transposed.shape[1] == 64:
                    #     # Append the DataFrame to the list
                    #     dfs.append(df_transposed)
                    #     count += 1
                    # if(count == 200):
                    #     #print("Chunk", len(perm_amp))
                    #     result_df = pd.concat(dfs, axis=0)
                    #     #result_df = result_df.reset_index(drop=True)
                    #     print(result_df.shape)
                    #     dfs = []
                    #     count = 0
                    #     predict(result_df)
                        
            except Exception as e:
                print("Error:",{e})
                pass

    except KeyboardInterrupt:
        print("Exiting gracefully.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()