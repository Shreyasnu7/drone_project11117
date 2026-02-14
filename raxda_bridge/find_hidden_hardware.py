import serial
import time
import os
import glob

print("🕵️ SEARCHING FOR HARDWARE (LIDAR & CAMERA)...")

# --- 1. CAMERA CHECK ---
print("\n[1] CAMERA CHECK")
if os.path.exists("/dev/video0"):
    print("   ✅ /dev/video0 FOUND.")
    print("   (If logging implies 'No Frames', ensure you ran ./setup_camera.sh)")
else:
    print("   ❌ /dev/video0 NOT FOUND.")

# --- 2. SERIAL PORT SCANNER ---
print("\n[2] LIDAR SEARCH (Scanning all ports...)")

# Candidate ports
ports = glob.glob('/dev/ttyS*') + glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
# Filter out S2 (FC) if known busy, but scanning briefly is ok if we are careful
# Baud rates to try (YDLidar X2 is usually 115200)
bauds = [115200, 128000, 230400, 19200, 57600]

found_lidar = False

for port in ports:
    # Skip console serial (usually S0 or S1 depending oin board, but lets check anyway)
    
    print(f"   👉 Checking {port}...", end='', flush=True)
    
    for baud in bauds:
        try:
            s = serial.Serial(port, baud, timeout=0.5)
            # Try read
            data = s.read(100)
            s.close()
            
            if len(data) > 10:
                print(f"\n      ✅ ALIVE AT {baud} BAUD! ({len(data)} bytes rx)")
                # Heuristic: Lidar sends lots of data. FC sends heartbeat.
                # X2 Lidar often starts with 0xAA 0x55, but just raw data is a good sign.
                print(f"      Bytes: {data[:20]}")
                if b'\xaa\x55' in data:
                     print("      🎯 HEADER MATCH: 0xAA 0x55 (Likely YDLidar!)")
                found_lidar = True
            else:
                # No data
                pass
        except OSError:
            # Busy or permission error
            # print("(Busy/Error)", end='')
            pass
        except Exception as e:
            pass
            
    print("\r", end='') # Clean line

if not found_lidar:
    print("\n   ❌ NO SERIAL DATA DETECTED ON ANY PORT.")
    print("   Please double check TX/RX wiring if using UART pins.")
    print("   RX on Lidar -> TX on Radxa")
    print("   TX on Lidar -> RX on Radxa")
else:
    print("\n   ✅ SCAN COMPLETE. See above for candidates.")
