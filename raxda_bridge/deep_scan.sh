#!/bin/bash
# deep_scan.sh
# Gathers all hardware status in one report

echo "========================================"
echo "🔍 DEEP HARDWARE SCAN REPORT"
echo "========================================"
echo ""
echo "1. CONFIG CHECK (/boot/armbianEnv.txt)"
echo "----------------------------------------"
cat /boot/armbianEnv.txt
echo ""

echo "2. USB DEVICES (lsusb)"
echo "----------------------------------------"
lsusb # Should show CP210x or CH340 for Lidar
echo ""

echo "3. SERIAL DEVICES (/dev/tty*)"
echo "----------------------------------------"
ls -l /dev/ttyUSB* /dev/ttyS* 2>/dev/null | grep -E "ttyS2|ttyS4|ttyUSB"
echo ""

echo "4. VIDEO DEVICES (/dev/video*)"
echo "----------------------------------------"
ls -l /dev/video* 2>/dev/null
echo ""

echo "5. KERNEL LOGS (Last 20 lines of USB/Cam/UART)"
echo "----------------------------------------"
dmesg | grep -E "tty|usb|Video|imx219|Unicam|dwb" | tail -n 25
echo ""
echo "========================================"
