# WiFi-Vision-Counting

This project aims to advance crowd counting techniques by combining Computer Vision and WiFi Sensing. As our graduation thesis, this project will explore the potential of these technologies as well as creating a public crowd-counting dataset consist of images and CSI data.

## Dataset Collection

...

## Data Processing

...

## Model Training

...

## System Deployment

### PC (Windows, Linux)

Python version: 3.10

Steps:

0. Set up your camera and a pair of ESP32 microcontroller.

1. Clone this repository.

    ```
    git clone https://github.com/thu4n/WiFi-Vision-Counting
    ```

2. Go to the `real-time-counting` directory.

    ```
    cd WiFi-Vision-Counting/real-time-counting
    ```

3. Initialize `venv`, I highly recommend using it because we will install a lot of dependencies.

    ```
    python3 -m venv venv
    ``` 

4. Activate `venv`

    ```
    # Windows

    .\venv\Scripts\activate

    # Linux

    source /venv/bin/activate
    ```
5. Install Python dependencies (it will take quite some time).

    ```
    pip install -r requirements_pc.txt
    ```

6. Run the script.

    ```
    python main_pc.py
    ```

### Jetson Nano

NOTE: The Jetson Nano is outdated so lots of problem might occur during installation and deployment (we only used it because we didn't have any other choice).

To be written later...