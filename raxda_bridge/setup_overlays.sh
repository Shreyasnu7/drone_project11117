#!/bin/bash
# setup_overlays.sh
# CONFIGURATION:
# UART2 (ttyS2) -> Flight Controller
# UART4 (ttyS4) -> ESP32
# CSI           -> Camera (IMX219 - Pi Cam V2.1)
# USB           -> Lidar (Requires 5V Power!)

echo "🔧 CONFIGURING OVERLAYS (UART2 + UART4 + CAMERA)..."

CFG="/boot/armbianEnv.txt"

add_or_update() {
    key=$1
    val=$2
    file=$3
    if grep -q "^$key=" "$file"; then
        sudo sed -i "s|^$key=.*|$key=$val|" "$file"
    else
        echo "$key=$val" | sudo tee -a "$file"
    fi
}

if [ -f "$CFG" ]; then
    # Enable UART2 (FC), UART4 (ESP32), and IMX219 (Camera V2.1)
    # Note: 'uart3' is DISABLED to protect USB/Lidar conflict
    add_or_update "overlays" "uart2-m0 uart4-m1" "$CFG"
    add_or_update "user_overlays" "radxa-zero3-rpi-camera-v2" "$CFG"
    
    # Specific Params
    add_or_update "param_uart2_m0" "on" "$CFG"
    add_or_update "param_uart4_m1" "on" "$CFG"
    # IMX219 doesn't need a separate param line
    
    # Free up UART2 from Console
    add_or_update "console" "display" "$CFG"
    
    echo "✅ /boot/armbianEnv.txt updated."
    echo "👉 overlays=uart2-m0 uart4-m1"
    echo "👉 user_overlays=radxa-zero3-rpi-camera-v2"
    echo "👉 console=display"
else
    echo "⚠️ /boot/armbianEnv.txt NOT FOUND."
fi

# Runtime Attempt
if command -v rsetup &> /dev/null; then
    sudo rsetup overlay enable uart2-m0 2>/dev/null || true
    sudo rsetup overlay enable uart4-m1 2>/dev/null || true
    # Camera usually requires reboot, rsetup might not work for MIPI
fi

echo "✅ DONE. REBOOT REQUIRED."
