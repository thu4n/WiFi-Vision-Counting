import winsound

def camera_start_sound():
    frequency = 2000  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def esp_start_sound():
    frequency = 1500  # Set Frequency To 1500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def end_sound():
    frequency = 1000  # Set Frequency To 1000 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def main():
    pass

if __name__ == "__main__":
    main()