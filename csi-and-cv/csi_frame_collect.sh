#!/bin/bash
BAURATE=921600
PORT=/dev/ttyUSB0
SES=s1

# Trap termination signal (SIGINT)
#trap 'kill -INT %1; exit' SIGINT

function killSubproc(){
    kill $(jobs -p -r)
}

# Run minicom with capture and filtering
sudo minicom -b $BAURATE -D $PORT | grep "CSI_DATA" | perl -ne 'print time(),",$_"' > frame-csi-$SES.csv &

MINICOM_PID=$$
echo $MINICOM_PID

sleep 10
echo -ne "\x01x\r"
trap killSubproc INT

# Exit script
exit 0