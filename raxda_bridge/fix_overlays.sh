#!/bin/bash
set -e

# CONFIG PATH
CONFIG="/boot/armbianEnv.txt"

echo "🔧 FIXING BOOT CONFIGURATION (Adding UART2)..."

# 1. Backup
cp $CONFIG $CONFIG.bak2
echo "✅ Backup created at $CONFIG.bak2"

# 2. Write Corrected Config (Adding uart2-m0)
# We keep uart3-m0 (FC?)
# We keep uart4-m1 (ESP32 - Fixed)
# We ADD uart2-m0 (FC - Requested)
cat <<EOF > $CONFIG
verbosity=1
bootlogo=false
console=display
extraargs=usb_otg.dr_mode=host
overlay_prefix=rk3568
fdtfile=rockchip/rk3566-radxa-zero3.dtb
rootdev=UUID=abafe666-60ee-47ab-84d9-7bfe4aa199d8
rootfstype=ext4
# CHANGED: Added uart2-m0
overlays=uart2-m0 uart3-m0 uart4-m1
param_uart3_console=off
user_overlays=radxa-zero3-rpi-camera-v2
param_uart1_console=off
param_uart4_console=off
param_uart2_console=off
usbstoragequirks=0x2537:0x1066:u,0x2537:0x1068:u
EOF

echo "✅ Config Updated: uart2-m0 + uart3-m0 + uart4-m1 enabled."
echo "📜 New Content:"
cat $CONFIG
