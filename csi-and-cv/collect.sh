#!/bin/bash
SESS=$1
DEV_0="csi_dev_0"
DEV_1="csi_dev_1"

if [ ! -d "$DEV_0" ]; then
    mkdir $DEV_0
fi

if [ ! -d "$DEV_1" ]; then
    mkdir $DEV_1
fi

if [ $# -eq 0 ]; then
    SESS=1
fi

echo "Session: $SESS"

# Collects the data from the CSI and CV experiments
sudo python3 csi_frame_collect.py 0 > ./$DEV_0/sess_${SESS}_csi.csv &
sudo python3 csi_frame_collect.py 1 > ./$DEV_1/sess_${SESS}_csi.csv &

if [ $? -eq 0 ]; then
    ./cv_frame_collect.py $SESS &
fi
#CSI_PID=$!
#CV_PID=$!

#echo "CSI_PID: $CSI_PID"
#echo "CV_PID: $CV_PID"

wait
exit 0
