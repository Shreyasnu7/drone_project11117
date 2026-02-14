#!/bin/bash
echo "🚑 REVIVING CAMERA SENSOR (IMX219 - Pi Cam V2.1)..."

# 1. Reload Driver (IMX219 is built-in, but try anyway)
echo "   Attempting driver reload..."
sudo rmmod imx219 2>/dev/null || echo "   (imx219 is built-in, cannot unload - this is OK)"
sudo rmmod v4l2_async 2>/dev/null

echo "   Reloading..."
sudo modprobe v4l2_async 2>/dev/null || true
sudo modprobe imx219 2>/dev/null || echo "   (imx219 built-in, modprobe not needed)"

sleep 1

# 2. Check if it appeared
echo "   Checking Topology..."
SENSOR_COUNT=$(media-ctl -d /dev/media0 -p | grep -i "imx219" | wc -l)

if [ "$SENSOR_COUNT" -gt 0 ]; then
    echo "✅ IMX219 SENSOR IS BACK! ($SENSOR_COUNT found)"
    echo "   Now run ./setup_camera.sh"
else
    echo "❌ SENSOR STILL MISSING."
    echo "   Dmesg:"
    dmesg | grep -i imx219 | tail -n 5
    echo ""
    echo "   Check:"
    echo "   1. Is the ribbon cable seated properly? (blue side faces connector lock)"
    echo "   2. Is the correct overlay enabled? (radxa-zero3-rpi-camera-v2)"
    echo "   3. Run: cat /boot/armbianEnv.txt | grep overlay"
fi
