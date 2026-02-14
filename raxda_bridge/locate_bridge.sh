#!/bin/bash
echo "🔎 SEARCHING FOR BRIDGE SERVICE..."

# Check common locations
LOC1=$(find /home/shreyash -name "real_bridge_service.py" 2>/dev/null)
LOC2=$(find /mnt/sdcard -name "real_bridge_service.py" 2>/dev/null)

if [ ! -z "$LOC1" ]; then
    echo "   FOUND: $LOC1"
fi
if [ ! -z "$LOC2" ]; then
    echo "   FOUND: $LOC2"
fi

if [ -z "$LOC1" ] && [ -z "$LOC2" ]; then
    echo "   ❌ NOT FOUND in /home or /mnt."
fi

echo -e "\n🔎 CHECKING CAMERA TOPOLOGY..."
if [ -f "./dump_topology.sh" ]; then
    chmod +x ./dump_topology.sh
    ./dump_topology.sh
else
    echo "   ⚠️ dump_topology.sh missing. Extracting manual dump..."
    media-ctl -d /dev/media0 -p
fi
