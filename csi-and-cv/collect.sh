#!/bin/bash
SESS=$1
DEV_0="csi_dev_0"
DEV_1="csi_dev_1"

# Check if the directories exist, if not create them
if [ ! -d "$DEV_0" ]; then
    mkdir $DEV_0
fi

if [ ! -d "$DEV_1" ]; then
    mkdir $DEV_1
fi

# Check if the session number is provided, if not set it to 1
if [ $# -eq 0 ]; then
    SESS=1
fi

echo "Session: $SESS"

# Get the duration of the experiment
read -p "Duration(seconds): " DURATION

# Check if the duration is provided, if not set it to 10 seconds
if [ -z "$DURATION" ]; then
    DURATION=10
fi

# Get the password for sudo
read -s -p "Enter Password for sudo: " PASSWORD

# Check if the password is correct
echo $PASSWORD | sudo -S echo "Password accepted"

# Collects the data from the CSI and CV experiments
sudo python3 csi_frame_collect.py 0 $DURATION > ./$DEV_0/sess_${SESS}_csi.csv &
sudo python3 csi_frame_collect.py 1 $DURATION > ./$DEV_1/sess_${SESS}_csi.csv &

# Check if the CSI experiment is running
if [ $? -eq 0 ]; then
    ./cv_frame_collect.py $SESS $DURATION &
fi

# Wait for the experiment to finish
wait

# Exit script
exit 0