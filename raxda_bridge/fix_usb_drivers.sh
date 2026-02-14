#!/bin/bash
echo "🔌 LOADING USB DRIVERS..."

# Load common Serial Drivers
sudo modprobe cp210x
sudo modprobe ch341
sudo modprobe ftdi_sio
sudo modprobe usbserial
sudo modprobe cdc_acm

echo "✅ Drivers Loaded."
echo "   Checking dmesg for new devices..."
dmesg | grep -E "cp210x|ch341|ftdi|ttyUSB|ttyACM" | tail -n 10
