#!/bin/bash
BAURATE=921600
PORT=/dev/ttyUSB0

# Run minicom with capture and filtering
timeout 3s minicom -b $BAURATE -D $PORT | grep "CSI_DATA" > csi-frame.csv &

# Capture the process ID of the background minicom job
minicom_pid=$!
echo "minicom PID: ${minicom_pid}"

# Wait for minicom to finish (optional)
wait "${minicom_pid}"

# Exit script
exit 0