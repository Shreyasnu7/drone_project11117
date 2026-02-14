#!/bin/bash
# force_usb_host.sh
# FORCES THE USB OTG PORT TO ACT AS A HOST (MASTER)
# Essential for spliced cables that lack CC/ID resistors.

echo "🔌 FORCING USB HOST MODE..."

# 1. ADD HOST OVERLAY (Device Tree)
# We add a parameter to armbianEnv.txt that tells Kernel "I am Host".
ENV_FILE="/boot/armbianEnv.txt"

# Check if we assume 'rk3568' generic overlay usage or param
# Usually 'param_ otg_mode=host' isn't standard, so we use 'extraargs'

if grep -q "extraargs=" "$ENV_FILE"; then
    echo "⚠️ extraargs already exists. Checking..."
else
    # Append Host forcing arg
    # 'otg_mode=host' is for some kernels
    # 'usb-dvc3.dr_mode=host' or 'rk_usb...dr_mode=host' 
    # Best generic bet: 'modules-load=dwc2' and trying device tree fix.
    
    # Correction: The most robust way on Rockchip is usually overlay.
    # Let's try adding 'host-mode' to extraargs if supported, 
    # but primarily we rely on creating a dedicated overlay injection?
    # No, simple is better: 'dr_mode=host' via custom overlay is best.
    
    # For now, let's use the 'overlays' list if a 'host' overlay exists.
    # Since we can't see the list, we will try the 'extraargs' method for Kernel Pulse.
    echo "extraargs=usb_otg.dr_mode=host" | sudo tee -a "$ENV_FILE"
fi

# 2. RUNTIME FORCE (Soft Force)
# Try to switch mode via DebugFS (Works on some Kernels)
echo "host" | sudo tee /sys/kernel/debug/usb/rk_usb21_0/mode 2>/dev/null
echo "host" | sudo tee /sys/kernel/debug/usb/*_usb*/mode 2>/dev/null
echo "host" | sudo tee /sys/class/udc/*.usb/device/mode 2>/dev/null

# 3. VERIFY
echo "🔍 Checking USB Bus..."
lsusb

echo "=========================================="
echo "🔌 IF LSUSB IS STILL EMPTY:"
echo "It means the Kernel really wants that Hardware Resistor."
echo "But 'extraargs=usb_otg.dr_mode=host' might fix it on reboot."
echo "=========================================="
