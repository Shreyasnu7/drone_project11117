#!/bin/bash
echo "🔍 DEBUGGING USB MODE..."

echo "--- 1. CURRENT USB DEVICES ---"
lsusb

echo -e "\n--- 2. USB TOPOLOGY ---"
lsusb -t

echo -e "\n--- 3. OVERLAYS (armbianEnv.txt) ---"
cat /boot/armbianEnv.txt

echo -e "\n--- 4. AVAILABLE USB OVERLAYS ---"
find /boot/dtb/rockchip/overlay/ -name "*usb*"

echo -e "\n--- 5. KERNEL USB LOGS ---"
dmesg | grep -i usb | tail -n 20
