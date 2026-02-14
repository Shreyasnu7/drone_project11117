#!/bin/bash
echo "💀 Clearing ttyS3..."
sudo fuser -k /dev/ttyS2 2>/dev/null || true
sudo fuser -k /dev/ttyS1 2>/dev/null || true
sudo fuser -k /dev/ttyS3 2>/dev/null || true
sudo fuser -k /dev/ttyS4 2>/dev/null || true
sleep 0.5
echo "🚀 Starting Bridge..."
exec sudo /usr/bin/python3 /mnt/sdcard/drone_project/raxda_bridge/real_bridge_service.py
