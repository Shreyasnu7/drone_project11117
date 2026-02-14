#!/bin/bash

echo "☢️ INITIATING NUCLEAR USB FIX ☢️"

# 1. Kill the Screaming I2C Driver (FUSB302)
# We try multiple IDs because it might vary
echo "🔇 Silencing USB-C Smart Controller..."
for i in {0..5}; do
    echo "3-0022" | tee /sys/bus/i2c/drivers/typec_fusb302/unbind 2>/dev/null
    echo "0-0022" | tee /sys/bus/i2c/drivers/typec_fusb302/unbind 2>/dev/null
    echo "fe5c0000.i2c" | tee /sys/bus/platform/drivers/rk3x-i2c/unbind 2>/dev/null # Extreme measure: Kill I2C controller if specific device fails
done

# 2. Force USB2 PHY to Host
echo "🔌 Forcing USB2 PHY to HOST..."
echo host | tee /sys/devices/platform/fe8a0000.usb2-phy/otg_mode
echo host | tee /sys/kernel/debug/usb/fe8a0000.usb/mode
echo host | tee /sys/kernel/debug/usb/rk_usb/mode

# 3. Force USB3 PHY (if applicable)
echo "🚀 Forcing USB3 PHY..."
echo host | tee /sys/kernel/debug/usb/fcc00000.dwc3/mode 2>/dev/null

# 4. Disable Autosuspend (Prevent sleep)
echo "⚡ Disabling Power Saving..."
for i in /sys/bus/usb/devices/*/power/control; do
    echo on > "$i"
done

# 5. Wait and Check
echo "⏳ Waiting 3 seconds for stabilization..."
sleep 3

echo "📋 FINAL DEVICE LIST:"
lsusb
echo "--------------------------------"
echo "If you see your devices above, IT WORKED."
echo "If the list is empty, restart the board and run this script immediately."
