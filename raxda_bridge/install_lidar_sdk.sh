#!/bin/bash
set -e

echo "🔭 INSTALLING YDLIDAR SDK (FROM SOURCE)..."

# 1. Install Build Tools
echo "📦 Installing CMake, Swig, Git..."
sudo apt-get update
sudo apt-get install -y cmake swig git build-essential python3-dev

# 2. Clone SDK
cd ~
if [ -d "YDLidar-SDK" ]; then
    echo "   Removing old SDK folder..."
    rm -rf YDLidar-SDK
fi

echo "⬇️ Cloning YDLidar-SDK..."
git clone https://github.com/YDLIDAR/YDLidar-SDK.git
cd YDLidar-SDK

# 3. Build C++ Library
echo "🔨 Building C++ Library..."
mkdir build
cd build
cmake ..
make
sudo make install

# 4. Install Python Bindings (Manual Copy)
echo "🐍 Linking Python Bindings..."

# CMake installed them to /usr/local/lib/python3/dist-packages (or similar)
# We need them in our VENV.

VENV_SITE="/mnt/sdcard/venv/lib/python3.13/site-packages"
mkdir -p "$VENV_SITE"

# Find and copy ydlidar.py
echo "   Copying ydlidar.py..."
find /usr/local/lib/python3/dist-packages -name "ydlidar.py" -exec cp {} "$VENV_SITE/" \;

# Find and copy _ydlidar.so
echo "   Copying _ydlidar.so..."
find /usr/local/lib/python3/dist-packages -name "_ydlidar.so" -exec cp {} "$VENV_SITE/" \;

echo "✅ YDLIDAR SDK INSTALLED!"
/mnt/sdcard/venv/bin/python3 -c "import ydlidar; print('YDLidar Import Success');"
