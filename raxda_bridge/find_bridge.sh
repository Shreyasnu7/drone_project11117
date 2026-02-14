#!/bin/bash
echo "🔍 SEARCHING FOR real_bridge_service.py..."
echo ""
echo "=== Checking /mnt/sdcard ==="
ls -la /mnt/sdcard/ 2>/dev/null || echo "  (not accessible)"
echo ""
echo "=== Searching for Python file ==="
find /mnt/sdcard -name "*.py" 2>/dev/null | head -20
find /home/shreyash -name "real_bridge_service.py" 2>/dev/null
echo ""
echo "=== Checking common directories ==="
ls -la /mnt/sdcard/drone_project/ 2>/dev/null || echo "  /mnt/sdcard/drone_project/ not found"
ls -la /home/shreyash/drone_project/ 2>/dev/null || echo "  /home/shreyash/drone_project/ not found"
