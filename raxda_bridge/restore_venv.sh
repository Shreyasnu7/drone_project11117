#!/bin/bash
set -e

echo "🚑 RESTORING PYTHON ENVIRONMENT..."

# 1. Unmount and Remount SD Card to be safe
echo "🔄 Remounting SD Card..."
sudo umount -l /mnt/sdcard || true
sudo mount /dev/mmcblk1p1 /mnt/sdcard

# 2. Re-create Venv
if [ ! -f /mnt/sdcard/venv/bin/python3 ]; then
    echo "⚠️ Python binary missing! Recreating venv..."
    sudo rm -rf /mnt/sdcard/venv
    sudo apt-get install -y python3-venv python3-pip
    python3 -m venv /mnt/sdcard/venv
    
    # Enable site-packages for cv2 access if installed via apt (optional but good)
    echo "include-system-site-packages = true" >> /mnt/sdcard/venv/pyvenv.cfg
fi

# 3. Install Dependencies
echo "📦 Installing Libraries..."
source /mnt/sdcard/venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Core Libs
pip install pyserial aiohttp websockets pymavlink opencv-python-headless smbus2

# Lidar (If not in pip, try direct install or skip if built from source previously)
# For YDLidar, we might need to build from source or install 'ydlidar' pip package if available
# The user's code imports 'ydlidar', usually from the SDK. 
# Attempting pip install first.
pip install ydlidar || echo "⚠️ 'ydlidar' pip install failed. Use SDK build if needed."

echo "✅ Environment Restored!"
/mnt/sdcard/venv/bin/python3 --version
