#!/bin/bash
echo "💾 MAKING DRIVERS PERMANENT..."

# Backup /etc/modules
sudo cp /etc/modules /etc/modules.bak

# Add cp210x if not present
if ! grep -q "cp210x" /etc/modules; then
    echo "cp210x" | sudo tee -a /etc/modules
    echo "   Added cp210x"
else
    echo "   cp210x already present"
fi

# Add ch341 if not present
if ! grep -q "ch341" /etc/modules; then
    echo "ch341" | sudo tee -a /etc/modules
    echo "   Added ch341"
else
    echo "   ch341 already present"
fi

echo "✅ Drivers set to auto-load on boot."
cat /etc/modules
