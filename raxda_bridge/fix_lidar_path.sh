#!/bin/bash
set -e

echo "🔍 SEARCHING FOR LOST YDLIDAR LIBRARY..."

# Locations to check
CANDIDATES=(
    "/usr/local/lib/python3/dist-packages"
    "/usr/local/lib/python3.11/site-packages"
    "/usr/local/lib/python3.11/dist-packages"
    "/mnt/sdcard/venv/lib/python3.11/site-packages"
    "/home/shreyash/YDLidar-SDK/build/python"
    "/home/radxa/YDLidar-SDK/build/python"
)

FOUND_PY=""
FOUND_SO=""

# Find .py
for dir in "${CANDIDATES[@]}"; do
    if [ -f "$dir/ydlidar.py" ]; then
        FOUND_PY="$dir/ydlidar.py"
        echo "✅ Found ydlidar.py at: $FOUND_PY"
        break
    fi
done

# Find .so
for dir in "${CANDIDATES[@]}"; do
    if [ -f "$dir/_ydlidar.so" ]; then
        FOUND_SO="$dir/_ydlidar.so"
        echo "✅ Found _ydlidar.so at: $FOUND_SO"
        break
    fi
done

# If not found, broaden search
if [ -z "$FOUND_PY" ]; then
    echo "⚠️ Not found in standard paths. Searching entire /usr..."
    FOUND_PY=$(find /usr -name ydlidar.py | head -n 1)
fi

if [ -z "$FOUND_SO" ]; then
     FOUND_SO=$(find /usr -name _ydlidar.so | head -n 1)
fi


if [ -n "$FOUND_PY" ] && [ -n "$FOUND_SO" ]; then
    echo "🚀 Installing to SYSTEM PYTHON (/usr/lib/python3/dist-packages)..."
    sudo cp "$FOUND_PY" /usr/lib/python3/dist-packages/
    sudo cp "$FOUND_SO" /usr/lib/python3/dist-packages/
    
    echo "✅ Library Restored!"
    python3 -c "import ydlidar; print('SUCCESS: Imported ydlidar')"
else
    echo "❌ COULD NOT FIND LIBRARY. Please run install_lidar_sdk.sh again!"
    exit 1
fi

sudo systemctl restart drone-bridge
