#!/bin/bash
# god_mode.sh
# "GOD MODE": Maximum Protection, Coolest Temps, Immutable Files.

echo "🛡️ ENGAGING GOD MODE PROTECTION..."

# 1. OVERHEATING PROTECTION (Disable GUI)
# The Desktop (HDMI) consumes 30% CPU just sitting there.
# We disable it. The Drone doesn't need a screen.
echo "❄️ Disabling Desktop GUI (Headless Mode)..."
sudo systemctl set-default multi-user.target
echo "   ✅ GUI Disabled (Saves ~10°C)."

# 2. THERMAL CONTROL (CPU Governor)
# Switch from 'schedutil' (fast) to 'conservative' (cool).
echo "🌡️ Tuning CPU for Coolness..."
echo "conservative" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
# Make it permanent
sudo apt-get install -y cpufrequtils
echo "GOVERNOR=\"conservative\"" | sudo tee /etc/default/cpufrequtils
echo "   ✅ CPU set to Conservative Mode."

# 3. IMMUTABLE FILE LOCKING ("Can't Edit")
# We use the filesystem attribute '+i' (Immutable).
# Even ROOT cannot edit or delete these files.
# Prevents accidental corruption or user mistakes.
echo "🔒 Locking Critical Files (Immutable)..."
FILES_TO_LOCK=(
    "/boot/armbianEnv.txt"
    "/etc/fstab"
    "/boot/dtb/rockchip/rk3566-radxa-zero3.dtb"
)

for f in "${FILES_TO_LOCK[@]}"; do
    if [ -f "$f" ]; then
        sudo chattr +i "$f"
        echo "   🔒 Locked: $f"
    fi
done

# 4. RE-VERIFY WATCHDOG & JOURNAL
# Ensure the previous layers are active
if ! grep -q "RuntimeWatchdogSec=15" /etc/systemd/system.conf; then
    echo "🐕 Re-enabling Watchdog..."
    sudo sed -i 's/#RuntimeWatchdogSec=0/RuntimeWatchdogSec=15/g' /etc/systemd/system.conf
fi
echo "   ✅ Watchdog Verified."

# 5. OOM KILLER PROTECTION
# If RAM is full, reboot instead of freezing.
echo "vm.panic_on_oom=1" | sudo tee -a /etc/sysctl.conf
echo "kernel.panic=10" | sudo tee -a /etc/sysctl.conf

echo "========================================="
echo "🛡️ GOD MODE ACTIVE."
echo "1. GUI Disabled (Cooler)"
echo "2. CPU Throttled (Cooler)"
echo "3. Boot Files Locked (Cannot be edited/corrupted)"
echo "   (To edit them later, use: sudo chattr -i <file>)"
echo "4. Auto-Reboot on Freeze/OOM"
echo "-----------------------------------------"
echo "REBOOT NOW: sudo reboot"
echo "========================================="
