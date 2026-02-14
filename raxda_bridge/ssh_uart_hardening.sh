#!/bin/bash
# =====================================================
# RADXA ZERO 3W - BULLETPROOF SSH & UART HARDENING
# =====================================================
# Run this ON the Radxa to ensure SSH and UART NEVER die
# This script survives reboots, crashes, and power cuts
# =====================================================

echo "🛡️ RADXA SSH & UART HARDENING"
echo "=============================="

# ===== 1. STATIC IP (Prevents DHCP IP changes) =====
echo ""
echo "📌 Step 1: Setting Static IP (192.168.0.11)..."
ACTIVE_CON=$(nmcli -t -f NAME con show --active | head -1)
if [ -n "$ACTIVE_CON" ]; then
    # Check if already static
    CURRENT_METHOD=$(nmcli -t -f ipv4.method con show "$ACTIVE_CON")
    if echo "$CURRENT_METHOD" | grep -q "manual"; then
        echo "   ✅ Already static IP"
    else
        sudo nmcli con modify "$ACTIVE_CON" ipv4.addresses 192.168.0.11/24 ipv4.method manual ipv4.gateway 192.168.0.1 ipv4.dns "8.8.8.8,8.8.4.4"
        echo "   ✅ Static IP set to 192.168.0.11"
    fi
else
    echo "   ⚠️ No active connection found. Set manually after connecting to WiFi."
fi

# ===== 2. SSH SERVICE HARDENING =====
echo ""
echo "🔒 Step 2: Hardening SSH service..."

# Unmask SSH (in case it was disabled)
sudo systemctl unmask ssh 2>/dev/null || true
sudo systemctl unmask sshd 2>/dev/null || true

# Enable SSH to start on boot
sudo systemctl enable ssh
sudo systemctl start ssh

# Ensure SSH config allows password auth and root login
sudo sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sudo sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
sudo sed -i 's/^#*PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Remove nologin file if exists (blocks all logins)
sudo rm -f /run/nologin /etc/nologin

echo "   ✅ SSH service enabled and configured"

# ===== 3. SSH KEEPALIVE SERVICE (Auto-Recovery) =====
echo ""
echo "⚡ Step 3: Installing SSH auto-recovery service..."

cat <<'EOF' | sudo tee /etc/systemd/system/ssh-keepalive.service
[Unit]
Description=SSH Keepalive - Auto Recovery Service
After=network.target

[Service]
Type=simple
ExecStart=/bin/bash -c 'while true; do sleep 30; rm -f /run/nologin /etc/nologin; if ! systemctl is-active --quiet ssh; then systemctl restart ssh; echo "$(date): SSH restarted by keepalive" >> /var/log/ssh-keepalive.log; fi; done'
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ssh-keepalive.service
sudo systemctl start ssh-keepalive.service
echo "   ✅ SSH auto-recovery service installed"

# ===== 4. UART CONSOLE RESCUE =====
echo ""
echo "🔌 Step 4: Setting up UART console fallback..."

# Create a getty service on ttyFIQ0 (the Radxa's debug UART output)
# This ensures UART login always works via the debug port
sudo systemctl enable serial-getty@ttyFIQ0.service 2>/dev/null || true
sudo systemctl start serial-getty@ttyFIQ0.service 2>/dev/null || true

# Also enable on ttyS0 as fallback
sudo systemctl enable serial-getty@ttyS0.service 2>/dev/null || true

echo "   ✅ UART console configured on ttyFIQ0"

# ===== 5. WATCHDOG (Auto-Reboot on Freeze) =====
echo ""
echo "🐕 Step 5: Enabling hardware watchdog..."

# Install watchdog if not present
if ! command -v watchdog &>/dev/null; then
    sudo apt-get install -y watchdog 2>/dev/null || echo "   ⚠️ watchdog package not available"
fi

# Configure watchdog
if [ -f /etc/watchdog.conf ]; then
    sudo sed -i 's|^#*watchdog-device.*|watchdog-device = /dev/watchdog|' /etc/watchdog.conf
    sudo sed -i 's|^#*watchdog-timeout.*|watchdog-timeout = 15|' /etc/watchdog.conf
    sudo sed -i 's|^#*max-load-1.*|max-load-1 = 24|' /etc/watchdog.conf
    sudo systemctl enable watchdog 2>/dev/null || true
    sudo systemctl start watchdog 2>/dev/null || true
    echo "   ✅ Hardware watchdog enabled (15s timeout)"
else
    echo "   ⚠️ watchdog.conf not found, skipping"
fi

# ===== 6. NETWORK RECOVERY SERVICE =====
echo ""
echo "📡 Step 6: Installing network recovery service..."

cat <<'EOF' | sudo tee /etc/systemd/system/network-recovery.service
[Unit]
Description=Network Recovery - Auto Reconnect WiFi
After=network.target

[Service]
Type=simple
ExecStart=/bin/bash -c 'while true; do sleep 60; if ! ping -c 1 -W 3 192.168.0.1 &>/dev/null; then echo "$(date): Network down, restarting..." >> /var/log/network-recovery.log; nmcli networking off; sleep 2; nmcli networking on; sleep 10; fi; done'
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable network-recovery.service
sudo systemctl start network-recovery.service
echo "   ✅ Network auto-recovery installed"

# ===== 7. PREVENT OVERLAYFS FROM KILLING WIFI =====
echo ""
echo "🔧 Step 7: Checking OverlayFS status..."
if grep -q 'overlayroot="tmpfs' /etc/overlayroot.conf 2>/dev/null; then
    echo "   ⚠️ OverlayFS is ENABLED — WiFi credentials are stored in RAM only!"
    echo "   ⚠️ To make WiFi permanent, run:"
    echo "      sudo overlayroot-chroot"
    echo "      # Then edit /etc/overlayroot.conf and set overlayroot=\"\""
    echo "      exit && sudo reboot"
else
    echo "   ✅ OverlayFS is disabled (safe)"
fi

# ===== SUMMARY =====
echo ""
echo "================================================"
echo "✅ SSH & UART HARDENING COMPLETE!"
echo "================================================"
echo ""
echo "  📌 Static IP:          192.168.0.11"
echo "  🔒 SSH:                enabled + auto-recovery"
echo "  🔌 UART:               ttyFIQ0 (Ctrl+C for login)"
echo "  🐕 Watchdog:           15s auto-reboot on freeze"
echo "  📡 Network Recovery:   auto-reconnect WiFi"
echo ""
echo "  SSH: ssh shreyash@192.168.0.11"
echo "  UART: PuTTY → COM port → 1500000 baud"
echo "================================================"
