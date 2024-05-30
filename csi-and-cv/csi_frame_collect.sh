#!/bin/bash
BAURATE=921600
PORT=/dev/ttyUSB0
SES=s1

# Run minicom with capture and filtering
timeout 10 sudo minicom -b $BAURATE -D $PORT | grep "CSI_DATA" | perl -ne 'print time(),",$_"' > frame-csi-$SES.csv &

MINICOM_PID=$$
echo $MINICOM_PID

# Exit script
exit 0
