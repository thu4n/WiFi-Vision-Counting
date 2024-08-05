#!/usr/bin/python3
import cv2

def main():
    try:
        cam = cv2.VideoCapture(0)
        while True:  
            ret_val, frame = cam.read()
            frame = cv2.flip(frame, 1)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            if not ret_val:
                break
            cv2.imshow('Camera Test', frame)
    except:
        print("Unable to open camera")
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()