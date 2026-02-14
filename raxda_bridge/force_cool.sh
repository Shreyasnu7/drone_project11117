#!/bin/bash
# force_cool.sh
# LIMITS CPU SPEED TO REDUCE HEAT (Underclocking)
# Target: ~1.1GHz (Factory max is ~1.8GHz)

echo "❄️ ACTIVATING FORCE COOL..."

# 1. Install Utils
if ! command -v cpufreq-set &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y cpufrequtils
fi

# 2. Set Max Frequency (1.1 GHz)
# This drops temp by 15-20C under load.
echo "📉 Limiting CPU to 1.1 GHz..."

# Apply to all cores
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
    echo "1100000" | sudo tee "$cpu" > /dev/null
done

# 3. Persist setting
echo "ENABLE=\"true\"" | sudo tee /etc/default/cpufrequtils
echo "GOVERNOR=\"conservative\"" | sudo tee -a /etc/default/cpufrequtils
echo "MAX_SPEED=\"1100000\"" | sudo tee -a /etc/default/cpufrequtils
echo "MIN_SPEED=\"408000\"" | sudo tee -a /etc/default/cpufrequtils

# Restart service
sudo systemctl restart cpufrequtils

echo "========================================="
echo "❄️ COOLING ACTIVE."
echo "Max Speed: 1.1 GHz"
echo "Check with: cpufreq-info | grep 'current policy'"
echo "========================================="
