#!/usr/bin/python3
import serial
import time
import os
from data_clean import clear_directory
from sound_alert import esp_start_sound, end_sound

def capture_csi(device_num, serial_port, session, time_duration, person_num):
    # ESP specs
    serial_port = f"COM{serial_port}"  # Set the serial port number
    baud_rate = 921600  # Set the baud rate
    ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=0.1) # Configure the serial port
    duration = time_duration # Set the duration 
    frame = 0 # Initialize the frame counter

    # Output directory
    folder = f"csi_main_{device_num}" # Set the folder name
    output_dir = f"{folder}/session_{session}/{person_num}_person" # Set the output directory

    if not os.path.exists(output_dir): # Check if the folder exists
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir) # Create the folder
    else:
        clear_directory(output_dir)

    # Set the file name
    file = f"{output_dir}/{person_num}_persons.csv" # Set the file name
    header = "type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,real_timestamp,len,CSI_DATA,machine_timestamp"

    if not os.path.exists(file):
        csi_file = open(f"{file}", "w")
        csi_file.write(header + "\n") # Write the header to the file

    start_time = time.time() # Get the current time

    while True:
        try:
            if time.time() - start_time >= duration:
                break
            data = ser.readline().decode("utf-8").strip()

            if data.startswith("CSI_DATA") and data.endswith("]"): # Check if the data is a CSI frame
                csi_file.write(data + "," + str(time.time()) + "\n")
                frame += 1

            if frame == 1:
                print(f"--- Start capturing CSI frames on device {device_num} for {person_num} person(s) ---")
                esp_start_sound() # Play the first sound

        except Exception as e:
            print("Error:",{e})
            pass
        
    csi_file.close() # Close the file
    end_sound() # Play the last sound
    print(f"CSI frames for {person_num} persons: {frame} frames.") # Print the number of frames

def main():
    capture_csi(0, 5, 1, 10, 1)
    # pass

if __name__ == "__main__":
    main()