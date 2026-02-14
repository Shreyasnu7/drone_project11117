import socket
import threading
import subprocess
import os

# --- CONFIG ---
SUBNET = "192.168.0" # Based on your previous IP 192.168.0.8
RANGE_START = 1
RANGE_END = 254
TARGET_PORT = 22 # SSH

found_devices = []

def scan_host(ip):
    # 1. Ping Check (Fast)
    try:
        # Windows ping uses -n 1, Linux uses -c 1
        param = '-n' if os.name == 'nt' else '-c'
        cmd = ['ping', param, '1', '-w', '200', ip]
        
        # Suppress output
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        res = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
        
        if res == 0:
            # 2. Port Check (If ping responds)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((ip, TARGET_PORT))
            if result == 0:
                print(f"✅ FOUND SSH DEVICE: {ip}")
                found_devices.append(ip)
            sock.close()
    except:
        pass

def main():
    print(f"--- SCANNING NETWORK {SUBNET}.x ---")
    threads = []
    
    for i in range(RANGE_START, RANGE_END):
        ip = f"{SUBNET}.{i}"
        t = threading.Thread(target=scan_host, args=(ip,))
        threads.append(t)
        t.start()
        
        # Batch to prevent socket exhaustion
        if len(threads) % 50 == 0:
            for t in threads: t.join()
            threads = []
            
    for t in threads: t.join()
    
    print("\n--- RESULTS ---")
    if not found_devices:
        print("❌ No Radxa found. (Is it powered? Is your PC on the same Wi-Fi?)")
    else:
        for ip in found_devices:
            print(f"🎯  ssh shreyash@{ip}")
            
if __name__ == "__main__":
    main()
