import os
import sys

def check_step(name, status, details=""):
    icon = "✅" if status else "❌"
    print(f"{icon} {name}: {details}")
    return status

print("\n🔍 SYSTEM DIAGNOSTIC REPORT")
print("===========================")

# 1. CHECK LIDAR LIB
try:
    import ydlidar
    check_step("Lidar Library", True, f"Installed (v{ydlidar.__version__ if hasattr(ydlidar, '__version__') else 'Unknown'})")
except ImportError:
    check_step("Lidar Library", False, "Module 'ydlidar' NOT FOUND. Run install_deps.py again.")
except Exception as e:
    check_step("Lidar Library", False, f"Error: {e}")

# 2. CHECK ESP32 PORT (UART4)
# Mapped to /dev/ttyS4 or /dev/ttyAML4 depending on kernel
ports = ["/dev/ttyS4", "/dev/ttyAML4"]
esp_found = False
for p in ports:
    if os.path.exists(p):
        check_step("ESP32 Port", True, f"Found at {p}")
        esp_found = True
        break

if not esp_found:
    check_step("ESP32 Port", False, "Missing /dev/ttyS4. (Did you run setup_overlays.sh and REBOOT?)")

# 3. CHECK CAMERA
cam_found = False
for i in range(10):
    if os.path.exists(f"/dev/video{i}"):
        check_step("Camera Device", True, f"Found /dev/video{i}")
        cam_found = True
        break
if not cam_found:
    check_step("Camera Device", False, "No /dev/videoX found. Check Physical Connection!")

print("===========================")
