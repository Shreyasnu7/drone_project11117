#!/bin/bash
# ultra_repair.sh
# Fixes OS Corruption caused by power loss

echo "🚑 STARTING ULTRA REPAIR..."

# 1. FORCE FILESYSTEM CHECK
echo "🔍 Scheduling Disk Check on Reboot..."
sudo touch /forcefsck

# 2. CLEAN APT CACHE (Fixes broken installs)
echo "🧹 Cleaning Package Manager..."
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock*
sudo dpkg --configure -a
sudo apt-get clean

# 3. RESET CONFIG (To Safe State)
echo "⚙️ Restoring Safe Overlays..."
bash -c 'cat > /boot/armbianEnv.txt <<EOF
verbosity=1
bootlogo=false
console=display
overlay_prefix=rk3568
fdtfile=rockchip/rk3566-radxa-zero3.dtb
rootdev=UUID=abafe666-60ee-47ab-84d9-7bfe4aa199d8
rootfstype=ext4
overlays=uart2-m0 uart4-m1
user_overlays=radxa-zero3-rpi-camera-v2
param_uart4_m1=on
param_uart2_m0=on
param_ov5647=on
usbstoragequirks=0x2537:0x1066:u,0x2537:0x1068:u
EOF'

echo "========================================="
echo "✅ REPAIR SCHEDULED."
echo "PLEASE REBOOT NOW."
echo "The startup will take longer (fixing disk)."
echo "========================================="
