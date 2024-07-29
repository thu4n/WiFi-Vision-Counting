from csi_frame_collect import capture_csi
from cv_frame_collect import capture_image
from multiprocessing import Process

def main():
    duration = 10
    num_per = 1
    p1 = Process(target=capture_csi, args=(0, 5, duration, num_per))
    p1.start()
    p2 = Process(target=capture_csi, args=(1, 6, duration, num_per))
    p2.start()
    p3 = Process(target=capture_image, args=(duration, num_per))
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    return

if __name__ == "__main__":
    main()