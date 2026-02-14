#!/bin/bash
# force_cool_v2.sh
# Fixed for Debian Trixie (cpufrequtils -> linux-cpupower)

echo "❄️ ACTIVATING FORCE COOL V2..."

# 1. Install Modern Tool
if ! command -v cpupower &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y linux-cpupower
fi

# 2. Apply Limits (Using cpupower)
if command -v cpupower &> /dev/null; then
    echo "📉 Setting max freq to 1.1GHz (cpupower)..."
    sudo cpupower frequency-set --governor conservative
    sudo cpupower frequency-set --max 1100000
else
    # Fallback to manual verify
    echo "⚠️ cpupower missing, using raw sysfs..."
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
        echo "1100000" | sudo tee "$cpu" > /dev/null
    done
fi

# 3. Create Persistent Service (So it stays after reboot)
echo "🔒 Creating boot service..."
cat <<EOF | sudo tee /etc/systemd/system/force_cool.service
[Unit]
Description=Limit CPU Frequency for Heat Management
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo 1100000 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq'

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable force_cool.service
sudo systemctl start force_cool.service

echo "========================================="
echo "❄️ COOLING ACTIVE (Service Installed)."
echo "   It will auto-start on every boot."
echo "========================================="
