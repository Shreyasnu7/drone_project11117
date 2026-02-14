#!/bin/bash
echo "🔬 CAMERA HARDWARE TEST"
echo "======================="

echo ""
echo "1. Checking video devices..."
ls -l /dev/video* 2>/dev/null || echo "No video devices found"

echo ""
echo "2. Trying to capture 1 frame from /dev/video0..."
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=NV12
timeout 5 v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=1 --stream-to=/tmp/test_frame.raw

if [ -f /tmp/test_frame.raw ]; then
    SIZE=$(stat -c%s /tmp/test_frame.raw)
    echo "✅ SUCCESS! Captured frame: $SIZE bytes"
    rm /tmp/test_frame.raw
else
    echo "❌ FAILED: No frame captured"
fi

echo ""
echo "3. Checking if sensor is powered..."
media-ctl -d /dev/media0 -p | grep -A 10 "imx219"

echo ""
echo "4. Testing with gst-launch (if available)..."
which gst-launch-1.0 && timeout 3 gst-launch-1.0 v4l2src device=/dev/video0 num-buffers=1 ! fakesink || echo "gstreamer not available"
