#!/usr/bin/python3
import serial
import time
import sys

def capture_csi(dev_num, ser_port, d_time, num_per):
    serial_port = f"COM{ser_port}"  # Set the serial port number
    baud_rate = 921600  # Set the baud rate
    ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=0.1) # Configure the serial port
    duration = d_time # Set the duration 
    folder = f"csi_dev_{dev_num}" # Set the folder name
    header = "type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,real_timestamp,len,CSI_DATA,machine_timestamp"
    file = f"{folder}/{num_per}_persons.csv" # Set the file name

    try:
        start_time = time.time() # Get the current time
        csi_file = open(f"{file}", "w") # Open the file to write the CSI data
        csi_file.write(header + "\n") # Write the header to the file
        frame = 0 # Initialize the frame counter
        while True:
            try:
                if time.time() - start_time >= duration:
                    print(f"Captured frames on device {dev_num} for {num_per} persons: {frame} frames.")
                    break
                data = ser.readline().decode("utf-8").strip()
                if data.startswith("CSI_DATA") and data.endswith("]"): # Check if the data is a CSI frame
                    csi_file.write(data + "," + str(time.time()) + "\n")
                    frame += 1
            except Exception as e:
                print("Error:",{e})
                pass
    except KeyboardInterrupt:
        print("Exiting gracefully.")
    finally:
        ser.close() # Close the serial port
        csi_file.close() # Close the file

def main():
    return

if __name__ == "__main__":
    main()