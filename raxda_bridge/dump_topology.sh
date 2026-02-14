#!/bin/bash
echo "🔍 DUMPING CAMERA TOPOLOGY..."

echo "--- 1. V4L2 DEVICES ---"
v4l2-ctl --list-devices

echo -e "\n--- 2. MEDIA TOPOLOGY ---"
media-ctl -d /dev/media0 -p

echo -e "\n--- 3. SENSOR KERNEL LOG ---"
dmesg | grep -i imx219 | tail -n 20
