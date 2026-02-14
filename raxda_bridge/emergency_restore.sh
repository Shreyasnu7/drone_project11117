#!/bin/bash
set -e
echo "🚑 EMERGENCY RESTORE OF PYTHON VENV..."

# 1. Remount & Fix Read-Only Error
echo "🛠 Checking Filesystem Write Access..."
if ! touch /mnt/sdcard/.write_test 2>/dev/null; then
    echo "❌ Filesystem is READ-ONLY. Fixing..."
    
    echo "   Killing processes using /mnt/sdcard..."
    sudo fuser -mk /mnt/sdcard 2>/dev/null || true
    
    echo "   Unmounting..."
    sudo umount /mnt/sdcard
    
    echo "   Running Disk Doctor (fsck)..."
    sudo fsck -y /dev/mmcblk1p1 || true
    
    echo "   Remounting RW..."
    sudo mount -o rw /dev/mmcblk1p1 /mnt/sdcard
    
    # Verify again
    if ! touch /mnt/sdcard/.write_test; then
        echo "🚨 CRITICAL: Still Read-Only! SD Card might be failing."
        exit 1
    fi
    rm /mnt/sdcard/.write_test
    echo "✅ Filesystem Fixed & Writable."
else
    echo "✅ Filesystem is Writable."
    rm /mnt/sdcard/.write_test
fi

# 2. Recreate Venv
if [ ! -f /mnt/sdcard/venv/bin/python3 ]; then
    echo "Creating venv..."
    sudo rm -rf /mnt/sdcard/venv
    python3 -m venv /mnt/sdcard/venv --system-site-packages
else
    echo "Venv exists."
fi

# 3. Pip Install
echo "Installing pip libs..."
source /mnt/sdcard/venv/bin/activate
pip install pyserial aiohttp websockets pymavlink opencv-python-headless smbus2

# 4. Restore Lidar Bindings (Copy from /usr/local)
# Requires 'sudo make install' to have been run previously (it was).
echo "Linking Lidar..."
SITE_PACKAGES=$(find /mnt/sdcard/venv/lib -name "site-packages" | head -n 1)

if [ -d "$SITE_PACKAGES" ]; then
    echo "   Target: $SITE_PACKAGES"
    # Try global python locations
    find /usr/local/lib/python3*/dist-packages -name "ydlidar.py" -exec cp {} "$SITE_PACKAGES/" \;
    find /usr/local/lib/python3*/dist-packages -name "_ydlidar.so" -exec cp {} "$SITE_PACKAGES/" \;
    
    # Try local build folder as backup
    if [ -d "/home/shreyash/YDLidar-SDK/build" ]; then
         find /home/shreyash/YDLidar-SDK/build -name "ydlidar.py" -exec cp {} "$SITE_PACKAGES/" \;
         find /home/shreyash/YDLidar-SDK/build -name "_ydlidar.so" -exec cp {} "$SITE_PACKAGES/" \;
    fi
else
    echo "❌ Could not find venv site-packages!"
fi

echo "✅ DONE. Try running the bridge now."
