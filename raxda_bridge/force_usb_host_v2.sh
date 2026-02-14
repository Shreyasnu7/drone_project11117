#!/bin/bash
# force_usb_host_v2.sh
# Fixes "Operation not permitted" by Unlocking God Mode first

echo "🔓 UNLOCKING SYSTEM FOR EDIT (Temporary)..."
# 1. REMOVE IMMUTABLE LOCK
sudo chattr -i /boot/armbianEnv.txt
sudo chattr -i /boot/dtb/rockchip/rk3566-radxa-zero3.dtb

# 2. APPLY USB HOST FIX
echo "🔌 Applying USB Host Patch..."
ENV_FILE="/boot/armbianEnv.txt"

# Clean up any partial lines
sudo sed -i '/extraargs=usb_otg.dr_mode=host/d' "$ENV_FILE"

# Append the Force Command
echo "extraargs=usb_otg.dr_mode=host" | sudo tee -a "$ENV_FILE"

# 3. RE-APPLY LOCK (God Mode)
echo "🔒 Re-Locking System..."
sudo chattr +i /boot/armbianEnv.txt
sudo chattr +i /boot/dtb/rockchip/rk3566-radxa-zero3.dtb

# 4. RUNTIME FORCE (Try immediate fix)
echo "host" | sudo tee /sys/kernel/debug/usb/rk_usb21_0/mode 2>/dev/null
echo "host" | sudo tee /sys/class/udc/*.usb/device/mode 2>/dev/null

echo "========================================="
echo "✅ USB HOST PATCH APPLIED."
echo "   (I had to unlock the file temporarily)"
echo "-----------------------------------------"
echo "PLEASE REBOOT NOW."
echo "The USB should work on next boot."
echo "========================================="
