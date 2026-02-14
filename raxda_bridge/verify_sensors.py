import serial
import time
import os
import sys

print("🔍 DEEP SENSOR DIAGNOSTIC")
print("==========================")

# 1. ESP32 UART DUMP
print("\n📡 CHECKING ESP32 (UART4)...")
try:
    s = serial.Serial('/dev/ttyS4', 115200, timeout=1)
    print("   ✅ Port Opened. Listening for 5 seconds...")
    t0 = time.time()
    data_found = False
    while time.time() - t0 < 5:
        if s.in_waiting:
            raw = s.read(s.in_waiting)
            try:
                print(f"   📨 RX: {raw.decode().strip()}")
                data_found = True
                break
            except:
                print(f"   📨 RX (Hex): {raw.hex()}")
                data_found = True
        time.sleep(0.1)
    
    if not data_found:
        print("   ❌ SILENCE: Port open but NO DATA received.")
        print("   👉 TIP: Swap TX/RX wires. Check ESP32 Power.")
    else:
        print("   ✅ ESP32 IS TALKING!")
    s.close()
except Exception as e:
    print(f"   ❌ Port Error: {e}")

# 2. LIDAR CHECK
print("\n📡 CHECKING LIDAR (USB)...")
if os.path.exists("/dev/ttyUSB0"):
    print("   ✅ /dev/ttyUSB0 Found!")
else:
    print("   ❌ /dev/ttyUSB0 MISSING. (Check Cable/Power)")

# 3. CAMERA CHECK
print("\n📡 CHECKING CAMERA (VIDEO0)...")
if os.path.exists("/dev/video0"):
    print("   ✅ /dev/video0 Found!")
else:
    print("   ❌ /dev/video0 MISSING. (Did you run setup_camera.sh for IMX219?)")
    
print("\n==========================")
