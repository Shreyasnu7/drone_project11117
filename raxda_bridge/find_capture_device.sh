#!/bin/bash
echo "🔍 FINDING CAPTURE DEVICE..."
echo ""

# Add user to video group
echo "1. Adding current user to 'video' group..."
sudo usermod -a -G video $USER
echo "   (You may need to logout/login for this to take effect)"

echo ""
echo "2. Scanning all video devices for capture capability..."
for i in {0..9}; do
    DEV="/dev/video$i"
    if [ -e "$DEV" ]; then
        echo "--- $DEV ---"
        v4l2-ctl -d $DEV --list-formats 2>&1 | head -5
        v4l2-ctl -d $DEV --all 2>&1 | grep -i "Video Capture\|capture"
        echo ""
    fi
done

echo ""
echo "3. Re-testing capture with correct permissions..."
sudo v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=480,pixelformat=YUYV
sudo timeout 3 v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=1 --stream-to=/tmp/frame.raw 2>&1

if [ -f /tmp/frame.raw ]; then
    SIZE=$(stat -c%s /tmp/frame.raw)
    echo "✅ Frame captured: $SIZE bytes"
    if [ $SIZE -gt 10000 ]; then
        echo "   Frame size looks valid!"
    else
        echo "   ⚠️ Frame size too small, might be error"
    fi
else
    echo "❌ Still no frame"
fi
