#!/bin/bash
echo "🔍 DIAGNOSING HARDWARE..."

# 1. Check Lidar (USB)
echo "--- LIDAR CHECK (/dev/ttyUSB*) ---"
ls -l /dev/ttyUSB* 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ NO /dev/ttyUSB devices found!"
    lsusb
else
    echo "✅ Found USB Serial devices."
    # Check who is using it
    echo "   Checking for interference..."
    sudo fuser -v /dev/ttyUSB0
fi

# 2. Check Camera (Video)
echo -e "\n--- CAMERA CHECK (/dev/video0) ---"
if [ -e /dev/video0 ]; then
    echo "✅ /dev/video0 exists."
    # Try to grab 5 frames at hardware level
    echo "Snapshotting 5 frames to test pipeline..."
    v4l2-ctl -d /dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=NV12 --stream-mmap --stream-count=5
    if [ $? -eq 0 ]; then
        echo "✅ KERNEL STREAMING WORKS! (OpenCV is the issue)"
    else
        echo "❌ KERNEL STREAMING FAILED! (Pipeline/Format is wrong)"
        echo "   Dumping topology..."
        media-ctl -d /dev/media0 -p
    fi
else
    echo "❌ /dev/video0 MISSING!"
fi

# 3. Check ESP32
echo -e "\n--- ESP32 CHECK ---"
ls -l /dev/ttyS* | grep "ttyS[0-4]"
