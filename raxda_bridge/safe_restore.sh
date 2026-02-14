#!/bin/bash
set -e
echo "🛡️ SAFE RESTORE OF PYTHON VENV..."

# 1. Check Write Access (Non-Destructive)
echo "🛠 Checking Filesystem Write Access..."
if ! touch /mnt/sdcard/.write_test 2>/dev/null; then
    echo "❌ Filesystem is READ-ONLY."
    echo "   Attempting safe remount (RW)..."
    
    # Try to remount RW (Works if errors were transient)
    if sudo mount -o remount,rw /mnt/sdcard 2>/dev/null; then
        echo "   ✅ Remount Success! We are back in business."
    else
        # If remount fails, try remounting ROOT (if sdcard is root)
        if sudo mount -o remount,rw / 2>/dev/null; then
             echo "   ✅ Root Remount Success!"
        else
             echo "   🚨 FAIL: Cannot write to disk."
             echo "   👉 ACTION REQUIRED: Run 'sudo reboot' and try again."
             exit 1
        fi
    fi
else
    echo "✅ Filesystem is Writable."
    rm /mnt/sdcard/.write_test
fi

# 2. Recreate Venv (If missing)
if [ ! -f /mnt/sdcard/venv/bin/python3 ]; then
    echo "🐍 Creating venv..."
    sudo rm -rf /mnt/sdcard/venv
    
    # PERMISSION FIX: Ensure user owns the mount point
    echo "   Fixing permissions for $USER..."
    sudo chown -R $USER:$USER /mnt/sdcard
    
    python3 -m venv /mnt/sdcard/venv --system-site-packages
fi

# 3. Pip Install
echo "📦 Installing Dependencies..."
source /mnt/sdcard/venv/bin/activate
pip install pyserial aiohttp websockets pymavlink opencv-python-headless smbus2

# 4. Bindings
echo "🔗 Linking Lidar Bindings..."
SITE_PACKAGES=$(find /mnt/sdcard/venv/lib -name "site-packages" | head -n 1)
# Copy from global cache
find /usr/local/lib/python3*/dist-packages -name "ydlidar.py" -exec cp {} "$SITE_PACKAGES/" \;
find /usr/local/lib/python3*/dist-packages -name "_ydlidar.so" -exec cp {} "$SITE_PACKAGES/" \;

echo "✅ RESTORE COMPLETE."
