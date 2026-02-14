#!/bin/bash
# Unified startup script for drone bridge system
set -e

echo "🚁 RADXA DRONE BRIDGE LAUNCHER"
echo "================================"

# Step 1: Initialize Camera Pipeline
echo ""
echo "📷 Step 1: Configuring Camera Pipeline..."
if [ -f ~/setup_camera.sh ]; then
    ~/setup_camera.sh
else
    echo "⚠️ setup_camera.sh not found, skipping camera init"
fi

# Step 2: Launch Bridge Service
echo ""
echo "🚀 Step 2: Launching Bridge Service..."
/mnt/sdcard/venv/bin/python3 /mnt/sdcard/drone_project/raxda_bridge/real_bridge_service.py
