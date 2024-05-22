#!/bin/bash

# Run minicom with capture and filtering
sudo minicom -b 921600 -D /dev/ttyUSB0 | grep "CSI_DATA" > csi-frame.csv &

# Capture the process ID of the background minicom job
minicom_pid=$!

# Wait for 3 seconds
sleep 3

# Simulate Ctrl+A+X+Enter keypresses to minicom
echo -ne "\001x\r" >&"${minicom_pid}"

# Wait for minicom to finish (optional)
#wait "${minicom_pid}"

# Exit script
exit 0