#!/bin/bash
SESS=$1

if [ $# -eq 0 ]; then
    SESS=1
fi

echo "Session: $0"

# Collects the data from the CSI and CV experiments
sudo python3 csi_frame_collect.py > sess_${SESS}_csi.csv &
CSI_PID=$!
./cv_frame_collect.py $SESS &
CV_PID=$!

echo "CSI_PID: $CSI_PID"
echo "CV_PID: $CV_PID"

wait
exit 0
