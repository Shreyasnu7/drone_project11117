#!/bin/bash
# Updated systemd service with USB device wait and boot resilience

sudo tee /etc/systemd/system/drone-bridge.service > /dev/null <<'EOF'
[Unit]
Description=Drone Bridge Service
After=network.target local-fs.target fix-sdcard.service
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/mnt/sdcard/drone_project/raxda_bridge
# Wait 10 seconds for USB devices to enumerate
ExecStartPre=/bin/sleep 10
ExecStartPre=/bin/bash -c 'echo "Waiting for USB devices..." && ls /dev/ttyUSB* /dev/ttyS* 2>/dev/null || true'
ExecStart=/bin/bash /mnt/sdcard/drone_project/raxda_bridge/run_clean.sh
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Updated drone-bridge.service with:"
echo "   - 10 second USB device wait"
echo "   - Dependency on SD card fix service"
echo "   - Better logging"
echo ""
echo "Reloading systemd..."
sudo systemctl daemon-reload
sudo systemctl enable drone-bridge.service

echo ""
echo "✅ DONE! Reboot to test:"
echo "   sudo reboot"
