#!/usr/bin/python3
import serial
import time
import sys

def main():
    dev= sys.argv[1] if len(sys.argv) > 0 else 0
    serial_port = f"/dev/ttyUSB{dev}"  # Use the specified serial port
    baud_rate = 921600  # Set the baud rate to 921600
    # Configure the serial port
    ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=0.1)
    try:
        duration = int(sys.argv[2])  # capture duration in seconds
        frame_count = 0
        start_time = time.time()
        while True:
            try:
                data = ser.readline().decode("utf-8").strip()
                if "CSI_DATA" in data and data.endswith("]"):
                    print(data.strip() + "," + str(time.time()))
                    frame_count += 1
                if time.time() - start_time > duration:
                    print(f"Frames: {frame_count-1}")
                    break
                        
            except Exception as e:
                print("Error:",{e})
                pass

    except KeyboardInterrupt:
        print("Exiting gracefully.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
