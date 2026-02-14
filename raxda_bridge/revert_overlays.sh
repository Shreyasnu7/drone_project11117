#!/bin/bash
# revert_overlays.sh
# Disables the UART overlays to see if Camera comes back

echo "🔧 REVERTING OVERLAYS (DIAGNOSTIC MODE)..."

CFG="/boot/armbianEnv.txt"
if [ -f "$CFG" ]; then
    # Comment out the overlay lines
    sudo sed -i 's/^overlays=/#overlays=/g' "$CFG"
    sudo sed -i 's/^user_overlays=/#user_overlays=/g' "$CFG"
    sudo sed -i 's/^param_uart/#param_uart/g' "$CFG"
    
    echo "✅ Overlays disabled in $CFG"
else
    echo "⚠️ /boot/armbianEnv.txt not found."
fi

# Try runtime disable if possible
if command -v rsetup &> /dev/null; then
    sudo rsetup overlay disable uart3-m0
    sudo rsetup overlay disable uart4-m1
fi

echo "✅ DONE. REBOOT and check if Camera returns."
