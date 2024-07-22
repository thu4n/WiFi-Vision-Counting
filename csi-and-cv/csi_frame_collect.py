#!/usr/bin/python3
import serial
import time
import sys

def main():
    dev= sys.argv[1]  # Get the serial port number from the command line
    if not dev.isdigit():
        print("Invalid serial port number")
        sys.exit(1)
    serial_port = f"COM{dev}"  # Set the serial port number
    baud_rate = 921600  # Set the baud rate
    ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=0.1) # Configure the serial port
    header = "type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,real_timestamp,len,CSI_DATA,machine_timestamp"
    duration = 10 # Set the duration 
    try:
        start_time = time.time() # Get the current time
        csi_file = open("csi_data.csv", "w") # Open the file to write the CSI data
        csi_file.write(header + "\n") # Write the header to the file
        while True:
            try:
                if time.time() - start_time >= duration:
                    break
                data = ser.readline().decode("utf-8").strip()
                if data.startswith("CSI_DATA") and data.endswith("]"): # Check if the data is a CSI frame
                    csi_file.write(data + "," + str(time.time()) + "\n")
            except Exception as e:
                print("Error:",{e})
                pass
    except KeyboardInterrupt:
        print("Exiting gracefully.")
    finally:
        ser.close() # Close the serial port
        csi_file.close() # Close the file
if __name__ == "__main__":
    main()