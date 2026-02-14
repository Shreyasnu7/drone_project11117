#!/bin/bash
# setup_esp32_uart.sh (Armbian / Radxa Config)
# Enables UART3 (GPIOX_9/10) for ESP32 interaction.

echo "========================================"
echo "🔧 CONFIGURING ESP32 UART (UART3/4)"
echo "   Detected OS: Armbian (likely)"
echo "========================================"

ARMBIAN_CFG="/boot/armbianEnv.txt"

# 1. ARMBIAN CONFIG VIA FILE I/O
if [ -f "$ARMBIAN_CFG" ]; then
    echo "   -> Found $ARMBIAN_CFG"
    
    # Check if overlays= line exists
    if grep -q "^overlays=" "$ARMBIAN_CFG"; then
        # Line exists, check for UART3
        if grep -q "uart3" "$ARMBIAN_CFG"; then
             echo "   -> UART3 already present in overlays."
        else
             echo "   -> Appending uart3 to existing overlays..."
             # Use sed to append uart3 to the end of the overlays line
             sudo sed -i '/^overlays=/ s/$/ uart3/' "$ARMBIAN_CFG"
        fi
        
        # Check for UART4
        if grep -q "uart4" "$ARMBIAN_CFG"; then
             echo "   -> UART4 already present in overlays."
        else
             echo "   -> Appending uart4 to existing overlays..."
             sudo sed -i '/^overlays=/ s/$/ uart4/' "$ARMBIAN_CFG"
        fi
    else
        # Line does not exist, create it
        echo "   -> Creating 'overlays' line..."
        echo "overlays=uart3 uart4" | sudo tee -a "$ARMBIAN_CFG"
    fi
    
    # Enable parameters if needed (sometimes required for Armbian specific pins)
    # echo "param_uart3_console=off" | sudo tee -a "$ARMBIAN_CFG"
    
    echo "   -> Armbian Config Updated."

# 2. RADXA OFFICIAL UTILITY (Fallback)
elif command -v rsetup &> /dev/null; then
    echo "   -> Found 'rsetup' (Official Radxa)."
    sudo rsetup overlay enable uart3-m0
    sudo rsetup overlay enable uart4-m0
    echo "   -> rsetup commands executed."

# 3. GENERIC UENV (Fallback)
elif [ -f "/boot/uEnv.txt" ]; then
     echo "   -> Found uEnv.txt (Generic)."
     if ! grep -q "uart3" "/boot/uEnv.txt"; then
         echo "overlays=uart3-m0 uart4-m0" | sudo tee -a "/boot/uEnv.txt"
     fi
else
    echo "❌ ERROR: Could not find valid config file (armbianEnv.txt or uEnv.txt)!"
    exit 1
fi

# 4. PERMISSIONS
echo "   -> Setting Permissions for ttyS*..."
sudo usermod -a -G dialout shreyash
sudo chmod 666 /dev/ttyS*

echo "========================================"
echo "✅ SETUP DONE. REBOOT REQUIRED!"
echo "   Run: sudo reboot"
echo "========================================"
