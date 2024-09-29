from csi_frame_collect import capture_csi
from cv_frame_collect import capture_image
from multiprocessing import Process

# CSI Devices and Ports
CSI_DEVICES = [0,1]
CSI_DEVICE_PORTS = [5,6]

# Session specs
num_per = 1
duration = 10
session = 1

def main():
    print(f"Start capturing {num_per} person(s) for {duration} seconds.")
    p1 = Process(target=capture_image, args=(session, duration, num_per))
    p1.start()
    p2 = Process(target=capture_csi, args=(CSI_DEVICES[0], CSI_DEVICE_PORTS[0], session, duration, num_per))
    p2.start()
    p1.join()
    p2.join()
    return

if __name__ == "__main__":
    main()