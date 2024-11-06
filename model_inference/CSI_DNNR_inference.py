import tensorflow as tf
import numpy as np
import serial
import data_preprocessor 

# Load the DNNR model
model_name = 'WiCount_FE_DNNR_loo_2days.keras' # Download from Google Drive first, another model choice is WiCount_FE_DNNR_2days.keras
model = tf.keras.models.load_model(model_name)

# Set up serial communication
serial_port = 'COM3'  # '/dev/ttyUSB0' on Linux
baud_rate = 921600
ser = serial.Serial(serial_port, baud_rate, timeout=1)

buffer = []  # Buffer to store collected data

print("Starting to read from the serial port and predict...")

while True:
    try:
        # Read a line from the serial port
        data = ser.readline().decode("utf-8").strip()
        if "CSI_DATA" in data:
            buffer.append(data)
            if len(buffer) > 200:
                buffer = buffer[-200:]  # Keep only the last 200 CSI packets

            if len(buffer) == 200:
                raw_csi = np.array(buffer)
                raw_amp = data_preprocessor.extract_amplitude(raw_csi)
                filtered_amp = data_preprocessor.denoise_data(raw_amp)
                filtered_amp_with_rssi = data_preprocessor.add_rssi(filtered_amp)
                features_df = data_preprocessor.extract_features(filtered_amp_with_rssi)
                # Model inference
                if features_df is not None:
                    prediction = model.predict(features_df)
                    print(f"Prediction: {prediction[0][0]}")
        
    except KeyboardInterrupt:
        print("Stopping prediction...")
        break
    except Exception as e:
        print(f"Error: {e}")

ser.close()