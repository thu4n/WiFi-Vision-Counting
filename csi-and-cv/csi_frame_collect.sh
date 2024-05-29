#!/bin/bash
BAURATE=921600
PORT=/dev/ttyUSB$1
SES=s$2
SESS_TIME=10

# Run minicom with capture and filtering
sudo minicom -b $BAURATE -D $PORT | grep "CSI_DATA" | perl -ne 'print time(),",$_"' > frame-csi-$SES.csv &

MINICOM_PID=$$
echo $MINICOM_PID

sleep $SESS_TIME
echo -ne "\x01x\r"

# Exit script
exit 0