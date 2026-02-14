#!/bin/bash
# Fix SD Card Mount to Prevent Corruption from Battery Disconnects

echo "🛡️ HARDENING SD CARD AGAINST POWER LOSS..."

# 1. Add 'sync' mount option to reduce corruption risk
# This forces immediate writes instead of buffering
echo "📝 Updating /etc/fstab with 'sync' option..."

# Backup fstab
sudo cp /etc/fstab /etc/fstab.backup

# Check if SD card entry exists
if grep -q "/mnt/sdcard" /etc/fstab; then
    # Update existing entry to add sync
    sudo sed -i 's|\(/mnt/sdcard.*defaults\)|\1,sync|g' /etc/fstab
    echo "✅ Updated existing fstab entry"
else
    # Add new entry (adjust device name if needed)
    echo "/dev/mmcblk1p1  /mnt/sdcard  ext4  defaults,sync  0  2" | sudo tee -a /etc/fstab
    echo "✅ Added new fstab entry"
fi

# 2. Remount SD card with new options
echo "🔄 Remounting SD card..."
sudo umount /mnt/sdcard 2>/dev/null
sudo mount /mnt/sdcard

if mountpoint -q /mnt/sdcard; then
    echo "✅ SD card remounted successfully with sync option"
else
    echo "❌ Failed to remount. Check /dev/mmcblk1p1 exists"
    exit 1
fi

# 3. Create boot script to auto-fix if filesystem becomes read-only
echo "📜 Creating auto-fix boot script..."
sudo tee /usr/local/bin/fix_sdcard_mount.sh > /dev/null <<'EOF'
#!/bin/bash
# Auto-fix SD card if it becomes read-only after crash

if ! touch /mnt/sdcard/.writetest 2>/dev/null; then
    echo "SD card is read-only. Attempting fix..."
    sudo mount -o remount,rw /mnt/sdcard
    rm -f /mnt/sdcard/.writetest
    echo "SD card remounted as read-write"
else
    rm -f /mnt/sdcard/.writetest
fi
EOF

sudo chmod +x /usr/local/bin/fix_sdcard_mount.sh

# 4. Add to systemd to run at boot
sudo tee /etc/systemd/system/fix-sdcard.service > /dev/null <<EOF
[Unit]
Description=Fix SD Card Mount at Boot
Before=drone-bridge.service
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/fix_sdcard_mount.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable fix-sdcard.service

echo "✅ Auto-fix service installed!"
echo ""
echo "🔋 BATTERY DISCONNECT PROTECTION SUMMARY:"
echo "   1. SD card now uses 'sync' mode (immediate writes)"
echo "   2. Auto-fix runs at every boot"
echo "   3. Battery disconnects are SAFER (but still risky)"
echo ""
echo "⚠️ RECOMMENDATION: Add a physical shutdown button to your drone"
echo "   (Connect GPIO pin to ground with a button, trigger 'shutdown now')"
