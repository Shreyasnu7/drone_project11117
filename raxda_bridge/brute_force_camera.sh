#!/bin/bash
echo "🔨 BRUTE FORCING CAMERA SETTINGS (IMX219 - Pi Cam V2.1)..."

# Constants - IMX219 uses I2C address 0x10
SENSOR="'m00_b_imx219 2-0010'"
DPHY="'rockchip-csi2-dphy0'"
CSI="'rkisp-csi-subdev'"
ISP="'rkisp-isp-subdev'"
MAIN="'rkisp_mainpath'"

# IMX219 Bayer formats to try
FORMATS=("SRGGB10_1X10" "SRGGB8_1X8" "SGBRG10_1X10" "SBGGR10_1X10" "SGRBG10_1X10")
RESOLUTIONS=("1920x1080" "3280x2464" "1640x1232" "640x480")

for FMT in "${FORMATS[@]}"; do
    for RES in "${RESOLUTIONS[@]}"; do
        echo "👉 Trying $FMT @ $RES..."
        
        # Reset
        media-ctl -r -d /dev/media0 > /dev/null 2>&1
        
        ERR=0
        
        # 1. SENSOR -> DPHY
        media-ctl -d /dev/media0 -l "$SENSOR:0->$DPHY:0 [1]" 2>/dev/null
        media-ctl -d /dev/media0 -V "$SENSOR:0 [fmt:$FMT/$RES field:none]" || ERR=1
        
        if [ $ERR -eq 0 ]; then
             # 2. DPHY -> CSI
             media-ctl -d /dev/media0 -l "$DPHY:1->$CSI:0 [1]" 2>/dev/null
             media-ctl -d /dev/media0 -V "$DPHY:1 [fmt:$FMT/$RES field:none]" || ERR=1
             
             if [ $ERR -eq 0 ]; then
                 echo "   ✅ SUCCESS! Found Working Config: $FMT @ $RES"
                 
                 echo "   Finalizing Pipeline..."
                 media-ctl -d /dev/media0 -l "$CSI:1->$ISP:0 [1]"
                 media-ctl -d /dev/media0 -V "$CSI:1 [fmt:$FMT/$RES field:none]"
                 
                 media-ctl -d /dev/media0 -l "$ISP:2->$MAIN:0 [1]"
                 media-ctl -d /dev/media0 -V "$ISP:2 [fmt:YUYV8_2X8/1920x1080 field:none]"
                 media-ctl -d /dev/media0 -V "$MAIN:0 [fmt:YUYV8_2X8/1920x1080 field:none]"
                 
                 echo "🏁 CAMERA FIXED."
                 exit 0
             fi
        fi
        
        echo "   ❌ Failed."
    done
done

echo "💀 ALL COMBINATIONS FAILED."
exit 1
