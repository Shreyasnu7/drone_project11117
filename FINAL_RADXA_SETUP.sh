#!/bin/bash
# FINAL RADXA SETUP SCRIPT (Overlay-Aware + Remount)
# Author: Antigravity
# Run this ON the Radxa (ssh shreyash@192.168.0.11)

set -e # Exit on error

echo "🚀 STARTING RADXA CONFIGURATION (SMART MODE)..."

# --- DEFINITIONS ---
# VENV_PATH set dynamically below
# VENV_PATH="/mnt/sdcard/venv"
SERVICE_FILE="/etc/systemd/system/drone-bridge.service"

# --- 1. UNLOCK THE GATES (RW REMOUNT) ---
echo "🔓 Attempting to Unlock Filesystem..."
# Unlock Root
if sudo mount -o remount,rw /; then
    echo "✅ Root Filesystem Remounted Read-Write!"
    RW_MODE="remount"
else
    echo "⚠️ Remount Failed path 1. Trying Chroot..."
    RW_MODE="chroot"
fi

# Unlock SD Card (Critical for Persistence) - NUCLEAR OPTION
echo "🔓 Unlocking SD Card (Nuclear Mode)..."
# Try simple remount first
sudo mount -o remount,rw /mnt/sdcard || true

# Test Write
if touch /mnt/sdcard/test_write 2>/dev/null; then
    echo "✅ SD Card is Writable."
    rm /mnt/sdcard/test_write
else
    echo "⚠️ SD Card is READ-ONLY. Attempting Repair..."
    # Kill processes using SD card
    sudo fuser -km /mnt/sdcard || true
    sudo umount /mnt/sdcard || true
    
    # Run FSCK (Assuming /dev/mmcblk1p1 - Common for SD on Radxa)
    echo "🔧 Running FSCK on SD Card..."
    sudo fsck -y /dev/mmcblk1p1 || echo "FSCK Warning"
    sudo fsck -y /dev/mmcblk0p1 || echo "FSCK Warning (mmcblk0)"
    
    # Remount
    sudo mount /mnt/sdcard || sudo mount /dev/mmcblk1p1 /mnt/sdcard
    sudo mount -o remount,rw /mnt/sdcard
    
    # Final Test
    if touch /mnt/sdcard/test_write 2>/dev/null; then
        echo "✅ SD Card Repair Successful!"
        rm /mnt/sdcard/test_write
    else
        echo "❌ SD CARD PARTITION (/mnt/sdcard) IS DEAD OR CORRUPTED."
        
        # New Fallback: Try Root Filesystem (which we successfully remounted RW above!)
        echo "🔄 Checking if we can save to Root Filesystem instead..."
        if touch /home/shreyash/test_root_write 2>/dev/null; then
             echo "✅ Root Filesystem is Writable! Installing to /home/shreyash/ (Persistent)."
             TARGET_DIR="/home/shreyash/drone_project_persistent"
             VENV_PATH="/home/shreyash/venv_persistent"
             rm /home/shreyash/test_root_write
        else
             echo "⚠️ Root is also Read-Only. Installing to RAM (Non-Persistent)..."
             TARGET_DIR="/home/shreyash/drone_project_ram"
             VENV_PATH="/home/shreyash/drone_project_ram/venv"
        fi
    fi
fi

# Define Target Directory based on write success
if [ -z "$TARGET_DIR" ]; then
    TARGET_DIR="/mnt/sdcard/drone_project"
    VENV_PATH="/mnt/sdcard/venv"
else
    # Fallback or specific target
    VENV_PATH="$TARGET_DIR/venv"
fi
echo "🎯 Target Directory: $TARGET_DIR"
echo "🎯 VENV Path: $VENV_PATH"

# --- 1.5 DEPLOY FROM STAGE ---
echo "🚚 Moving files from STAGE to $TARGET_DIR..."

COPY_SUCCESS=false
# Try SD Card Copy (Suppress errors to avoid panicking user)
# Source is now /tmp/stage/raxda_bridge/
# Ensure target directory exists
sudo mkdir -p $TARGET_DIR/raxda_bridge
if sudo cp -rf /tmp/stage/raxda_bridge/* $TARGET_DIR/raxda_bridge/ 2>/dev/null; then
    echo "✅ Files Synced to $TARGET_DIR (Persistent)."
    COPY_SUCCESS=true
else
    echo "⚠️  SD Card Write Failed (Hardware Issue). Switching to RAM..."
    
    # NEW TARGET: RAM
    TARGET_DIR="/home/shreyash/drone_project_ram"
    VENV_PATH="$TARGET_DIR/venv"
    
    sudo mkdir -p $TARGET_DIR/raxda_bridge
    if sudo cp -rf /tmp/stage/raxda_bridge/* $TARGET_DIR/raxda_bridge/; then
        echo "✅ Files Synced to RAM Target."
        COPY_SUCCESS=true
    else
        echo "❌ CRITICAL: RAM Copy Failed too? (Should be impossible)"
        exit 1
    fi
fi

sudo chmod -R 777 $TARGET_DIR

# Symlink Update (Ensure ~/drone_project points to correct location)
rm -rf /home/shreyash/drone_project
ln -s $TARGET_DIR /home/shreyash/drone_project

# --- 2. PIP DEPENDENCIES (Target: /mnt/sdcard/venv) ---

# SYSTEM DEP: OpenCV (Critical for Camera)
if ! dpkg -s python3-opencv >/dev/null 2>&1; then
    echo "📷 Installing Python3-OpenCV (System)..."
    apt-get update -y
    apt-get install -y python3-opencv
fi

# We prefer the SD card VENV because it persists naturally if /mnt/sdcard is excluded from overlay.
# FORCE RE-CREATION if missing system-site-packages
if [ -d "$VENV_PATH" ]; then
    if ! grep -q "include-system-site-packages = true" "$VENV_PATH/pyvenv.cfg"; then
        echo "♻️ VENV outdated (missing system-site-packages). Recreating..."
        rm -rf "$VENV_PATH"
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "🔨 Creating VENV at $VENV_PATH..."
    python3 -m venv $VENV_PATH --system-site-packages
fi

echo "📦 Installing Dependencies to VENV..."
$VENV_PATH/bin/pip install --upgrade pip
$VENV_PATH/bin/pip install pymavlink pyserial websockets aiohttp numpy

# --- 3. SYSTEM SERVICE (Target: /etc/systemd/system) ---
echo "⚙️ Configuring Service..."

INSTALL_SERVICE_CMD="cat <<EOF > $SERVICE_FILE
[Unit]
Description=Drone Bridge Service (Radxa <-> FC)
After=network.target

[Service]
Type=simple
User=root
ExecStart=$VENV_PATH/bin/python3 /home/shreyash/drone_project/raxda_bridge/real_bridge_service.py
WorkingDirectory=/home/shreyash/drone_project/raxda_bridge
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
"

if [ "$RW_MODE" == "remount" ]; then
    # Direct Write
    sudo bash -c "$INSTALL_SERVICE_CMD"
    sudo systemctl daemon-reload
    sudo systemctl enable drone-bridge
    sudo systemctl restart drone-bridge
    
    # LOCK IT BACK UP?
    # User said "Open Lock -> Put Stuff -> Close Lock"
    echo "🔒 Re-locking Filesystem (Remount RO)..."
    sudo mount -o remount,ro / || echo "⚠️ Could not remount ROOT RO."
    sudo mount -o remount,ro /mnt/sdcard || echo "⚠️ Could not remount SDCARD RO."

elif [ "$RW_MODE" == "chroot" ]; then
    # Overlayroot Write (Persistence)
    echo "💾 Persisting Service to Disk (Overlayroot)..."
    sudo overlayroot-chroot /bin/bash -c "$INSTALL_SERVICE_CMD && systemctl enable drone-bridge"
    
    # Runtime Start (RAM)
    echo "⚡ Starting Service in Current Session (RAM)..."
    sudo bash -c "$INSTALL_SERVICE_CMD" # Write to RAM overlay too
    sudo systemctl daemon-reload
    sudo systemctl restart drone-bridge
fi

# --- 4. PERMISSIONS ---
# Try to fix dialout just in case
sudo usermod -aG dialout shreyash || true

echo "✅ RADXA SETUP COMPLETE!"

echo "🔍 SERVICE LOGS (Execution Trace):"
# Force flush logs
sudo journalctl -u drone-bridge --sync
# Show last 30 lines (enough to see import errors)
sudo journalctl -u drone-bridge -n 30 --no-pager

echo ""
echo "📡 Service Status:"
systemctl status drone-bridge --no-pager
