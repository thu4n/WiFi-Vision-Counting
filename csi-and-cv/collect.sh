#!/bin/bash
SESS=$1

# Collects the data from the CSI and CV experiments
./csi_frame_collect.py > sess_${SESS}_csi.csv &
CSI_PID=$!
./cv_frame_collect.py &
CV_PID=$!

echo "CSI_PID: $CSI_PID"
echo "CV_PID: $CV_PID"

wait
exit 0