import os
import sys
import subprocess
import time

def check_command(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return output.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.output.decode('utf-8')}"

def main():
    print("--- RADXA HARDWARE AUDIT ---")
    
    # 1. USB DEVICES
    print("\n[1] CHECKING USB BUS (Lidar & Dongle)...")
    lsusb = check_command("lsusb")
    print(lsusb)
    
    has_silicon_labs = "Silicon Labs" in lsusb # CP210x (Lidar often uses this)
    has_huawei = "Huawei" in lsusb or "ZTE" in lsusb or "Qualcomm" in lsusb
    
    if has_silicon_labs: print("✅ FOUND POTENTIAL LIDAR (CP210x/Silicon Labs)")
    else: print("⚠️ LIDAR USB NOT FOUND")
    
    if has_huawei: print("✅ FOUND 4G DONGLE")
    else: print("⚠️ 4G DONGLE NOT DETECTED (Check Power)")

    # 2. ESP32 UART
    print("\n[2] CHECKING ESP32 UART (/dev/ttyAML1)...")
    if os.path.exists("/dev/ttyAML1"):
        print("✅ /dev/ttyAML1 EXISTS")
        # Check permissions
        perm = check_command("ls -l /dev/ttyAML1")
        print(f"   {perm}")
        if "dialout" in perm or "tty" in perm:
             print("   (Permissions look OK)")
    else:
        print("❌ /dev/ttyAML1 MISSING (Check overlays)")

    # 3. INTERNET
    print("\n[3] CHECKING INTERNET (4G)...")
    ping = check_command("ping -c 1 8.8.8.8")
    if "1 received" in ping:
        print("✅ INTERNET CONNECTED")
    else:
        print("❌ NO INTERNET")
        
    print("\n--- AUDIT COMPLETE ---")

if __name__ == "__main__":
    main()
