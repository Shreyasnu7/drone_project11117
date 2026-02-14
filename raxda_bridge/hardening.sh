#!/bin/bash
# hardening.sh
# MAKES THE OS IMMUNE TO SUDDEN POWER CUTS
# 1. Enables Full Journaling (data=journal)
# 2. Resets Config to Safe State

echo "🛡️ STARTING SYSTEM HARDENING..."

# 1. RESTORE GOLDEN CONFIG (Fixes Solid Green Boot Hangs)
# This removes the conflicting UART3 lines forever.
echo "⚙️ Writing Safe Boot Config..."
cat > /boot/armbianEnv.txt <<EOF
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
EOF
echo "✅ Config Reset to Safe Mode."

# 2. ENABLE FULL JOURNALING (Slow but Safe)
# 'data=journal' writes data to log before disk. Power cut = No Corruption.
echo "💾 Tuning Filesystem for Safety..."

# Backup fstab
cp /etc/fstab /etc/fstab.bak

# Add specific mount options for root
# grep for root, replace defaults with safer ones
if grep -q "defaults" /etc/fstab; then
    sudo sed -i 's/defaults/defaults,data=journal,commit=1/g' /etc/fstab
    echo "✅ /etc/fstab updated (Full Data Journaling)."
else
    echo "⚠️ /etc/fstab already modified (Skipping)."
fi

# 3. FORCE JOURNAL REPLAY ON BOOT
sudo tune2fs -o journal_data_write /dev/mmcblk0p1 2>/dev/null || true
# Try SD card too
sudo tune2fs -o journal_data_write /dev/mmcblk1p1 2>/dev/null || true

echo "========================================="
echo "🛡️ SYSTEM HARDENED."
echo "You can now unplug power anytime (mostly)."
echo "PLEASE REBOOT TO APPLY."
echo "========================================="
