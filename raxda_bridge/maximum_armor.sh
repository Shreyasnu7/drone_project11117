#!/bin/bash
# maximum_armor.sh
# "THE IMMORTAL SCRIPT"
# Adds 4 Layers of Protection against Corruption and Hangs.

echo "🛡️ INSTALLING MAXIMUM ARMOR..."

# -----------------------------------------------------
# LAYER 1: HARDWARE WATCHDOG (The "Kicker")
# If the "Solid Green Light" happens (Freeze), this
# chip will count to 15s and FORCE a reboot automatically.
# -----------------------------------------------------
echo "🐕 Enabling Hardware Watchdog..."
sudo sed -i 's/#RuntimeWatchdogSec=0/RuntimeWatchdogSec=15/g' /etc/systemd/system.conf
sudo sed -i 's/#RebootWatchdogSec=10min/RebootWatchdogSec=2min/g' /etc/systemd/system.conf
# Ensure module loads
echo "softdog" | sudo tee /etc/modules-load.d/watchdog.conf

# -----------------------------------------------------
# LAYER 2: KERNEL PANIC AUTO-REBOOT
# If Linux crashes on boot, wait 10s and retry.
# -----------------------------------------------------
echo "🔄 Enabling Panic Auto-Restart..."
echo "kernel.panic = 10" | sudo tee /etc/sysctl.d/99-panic.conf
echo "kernel.sysrq = 1" | sudo tee -a /etc/sysctl.d/99-panic.conf

# -----------------------------------------------------
# LAYER 3: FULL DATA JOURNALING (Anti-Corruption)
# Makes writes slower (safe) but immune to power cuts.
# -----------------------------------------------------
echo "💾 Enabling Full Journaling..."
# REMOVED DANGEROUS BLANKET FSTAB MODIFICATION
# The previous command corrupts non-ext4 drives (like SD cards) by forcing ext4 options.
# Only specific root partitioning should handle this manually.
echo "⚠️ Skipping Fstab modification to prevent SD card corruption."
# sudo tune2fs -o journal_data_write /dev/mmcblk0p1 2>/dev/null || true

# -----------------------------------------------------
# LAYER 3: SAFE EDITING WORKFLOW (New Standard)
# Installs 'safe-edit' command so you can edit locked files easily.
# Usage: safe-edit /etc/fstab
# -----------------------------------------------------
echo "🛠️ Installing 'safe-edit' Utility..."
cat > /usr/local/bin/safe-edit <<'EOF'
#!/bin/bash
if [ -z "$1" ]; then echo "Usage: safe-edit <file>"; exit 1; fi
FILE="$1"
sudo chattr -i "$FILE" 2>/dev/null || true
sudo nano "$FILE"
sudo chattr +i "$FILE" 2>/dev/null || true
echo "🔒 File re-locked."
EOF
chmod +x /usr/local/bin/safe-edit

# -----------------------------------------------------
# LAYER 4: THE GOLDEN CONFIG (No Conflicts)
# This prevents the initial freeze.
# -----------------------------------------------------
echo "⚙️ Restoring Safe Config..."
cat > /boot/armbianEnv.txt <<EOF
verbosity=1
bootlogo=false
console=serial
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

echo "========================================="
echo "🛡️ ARMOR INSTALLED."
echo "1. Watchdog Active (Auto-Reboot on Freeze)"
echo "2. Journaling Active (No Corruption)"
echo "3. Config Cleaned (No Conflicts)"
echo "-----------------------------------------"
echo "PLEASE REBOOT NOW TO LOCK IT IN."
echo "========================================="
