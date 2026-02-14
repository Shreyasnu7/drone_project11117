#!/bin/bash
# verify_hardware.sh
# Checks if Lidar and 4G Dongle are detected by the OS.

echo "========================================"
echo "🔍 VERIFYING HARDWARE CONNECTIONS"
echo "========================================"

# 1. CHECK LIDAR (Expected: /dev/ttyUSB0 or similar, often CP210x driver)
echo -n "📡 Checking for YDLidar (ttyUSB)... "
if ls /dev/ttyUSB* 1> /dev/null 2>&1; then
    LIDAR_PORT=$(ls /dev/ttyUSB* | head -n 1)
    echo "✅ FOUND at $LIDAR_PORT"
    # Basic ping check requires specific baud rate, skipping for now, just seeing port exists is good.
else
    echo "❌ NOT FOUND! (Is it plugged in? Check cable)"
fi

# 2. CHECK 4G DONGLE (Expected: WWAN interface or specific USB ID)
echo -n "📶 Checking for 4G Dongle (wwan)... "
# Check for network interface first
if ip link show | grep -q "wwan"; then
    echo "✅ FOUND (Network Interface Active)"
    ip addr show wwan0 | grep "inet"
elif lsusb | grep -qi "Qualcomm\|Huawei\|ZTE\|Quectel"; then
    echo "⚠️  USB Device Detected, but Network Interface missing. (Driver issue?)"
    lsusb | grep -i "Qualcomm\|Huawei\|ZTE\|Quectel"
else
    echo "❌ NOT FOUND! (Check USB, Power)"
fi

# 3. CHECK CAMERA (Expected: /dev/video0)
echo -n "📷 Checking for Camera (/dev/video0)... "
if [ -e /dev/video0 ]; then
    echo "✅ FOUND"
else
    echo "❌ NOT FOUND"
fi

echo "========================================"
echo "DONE."
