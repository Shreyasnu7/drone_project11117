#!/bin/bash
echo "🗺️ COMPLETE FILESYSTEM MAP"
echo "================================"
echo ""
echo "=== ROOT DIRECTORY STRUCTURE ==="
ls -la / | grep -E "mnt|media|home"
echo ""
echo "=== ALL MOUNTS ==="
df -h | grep -E "mmcblk|sd"
mount | grep -E "mmcblk|sd"
echo ""
echo "=== /mnt CONTENTS ==="
ls -laR /mnt 2>/dev/null | head -100
echo ""
echo "=== /media CONTENTS ==="
ls -laR /media 2>/dev/null | head -50
echo ""
echo "=== /home/shreyash CONTENTS ==="
ls -laR /home/shreyash 2>/dev/null | head -100
echo ""
echo "=== SEARCHING FOR .py FILES ==="
find /mnt -name "*.py" 2>/dev/null | grep -v "__pycache__" | head -20
find /media -name "*.py" 2>/dev/null | grep -v "__pycache__" | head -20
find /home -name "real_bridge_service.py" 2>/dev/null
echo ""
echo "=== SEARCHING FOR drone_project DIRECTORY ==="
find / -type d -name "drone_project" 2>/dev/null
find / -type d -name "raxda_bridge" 2>/dev/null
