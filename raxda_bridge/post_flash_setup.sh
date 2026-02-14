#!/bin/bash
# =====================================================
# RADXA ZERO 3W - COMPLETE DISASTER RECOVERY SCRIPT
# =====================================================
# Run this after fresh OS flash to restore ALL settings
# Everything except OS is stored on SD card at /mnt/sdcard
# =====================================================

set -e

echo "🔧 RADXA ZERO 3W - POST-FLASH SETUP"
echo "==================================="
echo ""

# ===== 1. MOUNT SD CARD =====
echo "📁 Step 1: Mounting SD card..."
sudo mkdir -p /mnt/sdcard
if ! grep -qs '/mnt/sdcard' /proc/mounts; then
    sudo mount /dev/mmcblk1p1 /mnt/sdcard 2>/dev/null || echo "⚠️ SD card already mounted or not found"
fi
sudo chown -R $USER:$USER /mnt/sdcard
echo "✅ SD card accessible at /mnt/sdcard"

# ===== 2. INSTALL SYSTEM DEPENDENCIES =====
echo ""
echo "📦 Step 2: Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    cmake \
    build-essential \
    v4l-utils \
    media-ctl \
    i2c-tools \
    || echo "⚠️ Some packages may already be installed"
echo "✅ System packages installed"

# ===== 3. USB-TO-SERIAL DRIVERS =====
echo ""
echo "🔌 Step 3: Configuring USB serial drivers..."
# Load cp210x driver for YDLidar
sudo modprobe cp210x
# Persist across reboots
if ! grep -q "cp210x" /etc/modules 2>/dev/null; then
    echo "cp210x" | sudo tee -a /etc/modules
fi
echo "✅ USB drivers configured"

# ===== 4. SERIAL PORT OVERLAYS (BOOT CONFIG) =====
echo ""
echo "⚙️ Step 4: Configuring serial port overlays..."
# Backup original
sudo cp /boot/armbianEnv.txt /boot/armbianEnv.txt.bak 2>/dev/null || true

# Enable UART2 (ttyS2 for FC) and UART4 (ttyS4 for ESP32)
if ! grep -q "overlays=uart2 uart4" /boot/armbianEnv.txt 2>/dev/null; then
    echo "   Adding UART overlays to boot config..."
    sudo sed -i 's/^overlays=.*/overlays=uart2 uart4/' /boot/armbianEnv.txt
    # If line doesn't exist, append it
    if ! grep -q "^overlays=" /boot/armbianEnv.txt; then
        echo "overlays=uart2 uart4" | sudo tee -a /boot/armbianEnv.txt
    fi
fi
echo "✅ Serial overlays configured (requires reboot to apply)"

# ===== 5. SERIAL PORT PERMISSIONS =====
echo ""
echo "🔐 Step 5: Setting up serial port permissions..."
sudo usermod -a -G dialout $USER
sudo usermod -a -G video $USER
sudo chmod 666 /dev/ttyS2 2>/dev/null || true  # FC
sudo chmod 666 /dev/ttyS4 2>/dev/null || true  # ESP32
sudo chmod 666 /dev/ttyUSB0 2>/dev/null || true # Lidar
echo "✅ Serial permissions configured"

# ===== 5. PYTHON VIRTUAL ENVIRONMENT =====
echo ""
echo "🐍 Step 5: Restoring Python environment..."
if [ ! -d "/mnt/sdcard/venv" ]; then
    echo "   Creating new venv..."
    python3 -m venv /mnt/sdcard/venv
fi

echo "   Installing Python dependencies..."
/mnt/sdcard/venv/bin/pip install --upgrade pip
/mnt/sdcard/venv/bin/pip install \
    pyserial \
    aiohttp \
    websockets \
    pymavlink \
    opencv-python-headless \
    smbus2 \
    numpy

echo "✅ Python environment ready"

# ===== 6. YDLIDAR SDK =====
echo ""
echo "🔭 Step 6: Installing YDLidar SDK..."
if [ ! -d "$HOME/YDLidar-SDK" ]; then
    cd $HOME
    git clone https://github.com/YDLIDAR/YDLidar-SDK.git
    cd YDLidar-SDK/build
    cmake ..
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    
    # Copy Python bindings to venv
    SITE_PKG=$(/mnt/sdcard/venv/bin/python3 -c "import site; print(site.getsitepackages()[0])")
    sudo cp ../python/ydlidar.py "$SITE_PKG/"
    sudo cp ../build/python/_ydlidar.so "$SITE_PKG/"
    echo "✅ YDLidar SDK installed"
else
    echo "✅ YDLidar SDK already installed"
fi

# ===== 7. CAMERA SETUP SCRIPT =====
echo ""
echo "📷 Step 7: Creating camera setup script..."
cat > /mnt/sdcard/drone_project/raxda_bridge/setup_camera.sh << 'CAMERA_EOF'
#!/bin/bash
# Camera pipeline configuration for Pi Camera V2.1 (IMX219)
set -e

sudo media-ctl -r -d /dev/media0

SENSOR="'m00_b_imx219 2-0010'"
DPHY="'rockchip-csi2-dphy0'"
CSI="'rkisp-csi-subdev'"
ISP="'rkisp-isp-subdev'"
MAIN="'rkisp_mainpath'"

sudo media-ctl -d /dev/media0 -l "$SENSOR:0->$DPHY:0 [1]"
sudo media-ctl -d /dev/media0 -l "$DPHY:1->$CSI:0 [1]"
sudo media-ctl -d /dev/media0 -l "$CSI:1->$ISP:0 [1]"
sudo media-ctl -d /dev/media0 -l "$ISP:2->$MAIN:0 [1]"

sudo media-ctl -d /dev/media0 -V "$SENSOR:0 [fmt:SRGGB10_1X10/1920x1080]"
sudo media-ctl -d /dev/media0 -V "$ISP:2 [fmt:YUYV8_2X8/1920x1080]"

sudo v4l2-ctl -d /dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=NV12

echo "✅ Camera V2.1 (IMX219) configured at 1920x1080 NV12"
CAMERA_EOF

chmod +x /mnt/sdcard/drone_project/raxda_bridge/setup_camera.sh
echo "✅ Camera setup created"

# ===== 8. LAUNCH SCRIPT =====
echo ""
echo "🚀 Step 8: Creating launch script..."
cat > /mnt/sdcard/drone_project/raxda_bridge/launch_bridge.sh << 'LAUNCH_EOF'
#!/bin/bash
echo "🚁 RADXA DRONE BRIDGE LAUNCHER"
echo "================================"
echo ""

# Configure camera
echo "📷 Step 1: Configuring Camera..."
/mnt/sdcard/drone_project/raxda_bridge/setup_camera.sh

echo ""
echo "🚀 Step 2: Launching Bridge Service..."
/mnt/sdcard/venv/bin/python3 /mnt/sdcard/drone_project/raxda_bridge/real_bridge_service.py
LAUNCH_EOF

chmod +x /mnt/sdcard/drone_project/raxda_bridge/launch_bridge.sh
echo "✅ Launch script created"

# ===== 9. CREATE SYSTEMD SERVICE (AUTO-START) =====
echo ""
echo "⚡ Step 9: Setting up auto-start service..."
sudo tee /etc/systemd/system/drone-bridge.service > /dev/null << SERVICE_EOF
[Unit]
Description=Radxa Drone Bridge Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/mnt/sdcard/drone_project/raxda_bridge
ExecStartPre=/bin/sleep 10
ExecStartPre=/mnt/sdcard/drone_project/raxda_bridge/setup_camera.sh
ExecStart=/mnt/sdcard/venv/bin/python3 /mnt/sdcard/drone_project/raxda_bridge/real_bridge_service.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE_EOF

sudo systemctl daemon-reload
echo "✅ Auto-start service created (disabled by default)"
echo "   To enable: sudo systemctl enable drone-bridge"
echo "   To start:  sudo systemctl start drone-bridge"

# ===== 10. CREATE HOME DIRECTORY SYMLINK =====
echo ""
echo "🔗 Step 10: Creating home symlink..."
ln -sf /mnt/sdcard/drone_project $HOME/drone_project
echo "✅ Symlink created: ~/drone_project -> /mnt/sdcard/drone_project"

# ===== FINAL SUMMARY =====
echo ""
echo "================================================"
echo "✅ POST-FLASH SETUP COMPLETE!"
echo "================================================"
echo ""
echo "📋 STATUS:"
echo "  • SD Card:      /mnt/sdcard"
echo "  • Python:       /mnt/sdcard/venv"
echo "  • Project:      /mnt/sdcard/drone_project"
echo "  • FC Port:      /dev/ttyS2 (19200 baud)"
echo "  • ESP32 Port:   /dev/ttyS4 (115200 baud)"
echo "  • Lidar Port:   /dev/ttyUSB0 (128000 baud)"
echo "  • Camera:       /dev/video0 (640x480 UYVY)"
echo ""
echo "🚀 TO START DRONE:"
echo "   cd /mnt/sdcard/drone_project/raxda_bridge"
echo "   ./launch_bridge.sh"
echo ""
echo "⚠️ IMPORTANT: Logout and login for group permissions!"
echo "================================================"
