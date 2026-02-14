#!/bin/bash
# list_overlays.sh
# Check what overlays actually exist in the kernel folder

echo "🔍 CHECKING AVAILABLE OVERLAYS..."
echo "Path: /boot/dtb/rockchip/overlay/"

ls /boot/dtb/rockchip/overlay/ | grep -E "uart|ov5647|imx219|spi|i2c"

echo ""
echo "🔍 CHECKING LOADED MODULES..."
lsmod | grep -E "imx219|videobuf2|v4l2"
