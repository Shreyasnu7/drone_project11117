#!/bin/bash
# fix_dependencies.sh v2
USER_HOME="/home/shreyash"

echo "========================================"
echo "🔧 FORCE INSTALLING DEPENDENCIES (v2)"
echo "========================================"

# 1. FIX SERIAL (Force Repip install aiohttp opencv-python-headless pyserial it)
echo "📦 Force-Reinstalling Pyserial..."
sudo python3 -m pip install pyserial --break-system-packages --force-reinstall
sudo python3 -m pip install ydlidar --break-system-packages --force-reinstall

# 2. INSTALL YDLIDAR (Requested)
echo "📦 Installing YDLidar SDK..."
# We use the pip binary which avoids building from source if possible, or builds smaller.
sudo python3 -m pip install ydlidar --break-system-packages

echo "========================================="
echo "✅ DEPENDENCIES FIXED."
echo "Now run: sudo ~/drone_project/raxda_bridge/fix_launcher_v2.sh"
echo "========================================="
