import winsound

def make_sound_1():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def make_sound_2():
    frequency = 1500  # Set Frequency To 1500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def main():
    pass

if __name__ == "__main__":
    main()