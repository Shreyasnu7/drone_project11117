# File: laptop_ai/lidar_driver.py
import serial
import time
import threading
import math
import numpy as np

class YDLidarDriver:
    """
    Driver for YDLIDAR X2 (USB/UART).
    Parses the standard 115200 serial stream to get 360-degree scan data.
    """
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.lock = threading.Lock()
        
        # Data storage: Array of [angle, distance]
        # X2 samples ~3000 pts/sec
        self.scan_data = {} # angle_int -> distance_mm
        self.latest_scan = [] # format: [(angle, dist_meters), ...]

    def start(self):
        """
        Auto-detect YDLIDAR on common USB ports.
        """
        ports_to_try = [self.port, "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyACM0"]
        # Remove duplicates
        ports_to_try = list(dict.fromkeys(ports_to_try))

        for port in ports_to_try:
            try:
                print(f"🔍 Probing {port} for Lidar...")
                ser = serial.Serial(port, self.baudrate, timeout=1.0)
                
                # Handshake Check (Try to read a few bytes to see if it's alive)
                ser.reset_input_buffer()
                # YDLidar usually streams data immediately
                data = ser.read(100)
                if len(data) > 10 and b'\xAA' in data and b'\x55' in data:
                    print(f"✅ FOUND Lidar on {port}")
                    self.serial = ser
                    self.port = port
                    self.running = True
                    self.thread = threading.Thread(target=self._worker, daemon=True)
                    self.thread.start()
                    return True
                else:
                     ser.close()
            except Exception:
                pass
        
        print("❌ Lidar Auto-Discovery Failed. Please check USB Hub.")
        return False

    def stop(self):
        self.running = False
        if self.serial:
            self.serial.close()

    def get_scan(self):
        """Returns list of (angle_deg, distance_meters)"""
        with self.lock:
            # return copy of latest complete scan
            return list(self.latest_scan)

    def get_obstacles(self, max_dist=2.0):
        """Returns list of (x, y) coordinates of obstacles within max_dist meters"""
        obs = []
        scan = self.get_scan()
        for angle, dist in scan:
            if 0.1 < dist < max_dist:
                # Polar to Cartesian
                rad = math.radians(angle)
                x = dist * math.cos(rad)
                y = dist * math.sin(rad)
                obs.append((x, y))
        return obs

    def _worker(self):
        # YDLIDAR Protocol Loop
        # Simplified parsing logic for X2/X4 standard format
        while self.running:
            try:
                # Wait for header 0xAA 0x55
                if self.serial.read(1) != b'\xAA':
                    continue
                if self.serial.read(1) != b'\x55':
                    continue
                
                # Packet Type (CT)
                ct = ord(self.serial.read(1))
                # Sample Count (LSN)
                lsn = ord(self.serial.read(1))
                # Start Angle (FSA)
                fsa_l = ord(self.serial.read(1))
                fsa_h = ord(self.serial.read(1))
                fsa = (fsa_h << 8) | fsa_l
                
                # End Angle (LSA)
                lsa_l = ord(self.serial.read(1))
                lsa_h = ord(self.serial.read(1))
                lsa = (lsa_h << 8) | lsa_l
                
                # Check Code (CS)
                cs_l = ord(self.serial.read(1))
                cs_h = ord(self.serial.read(1))
                
                # Read Data (LSN * 2 bytes)
                raw_data = self.serial.read(2 * lsn)
                
                # Process
                start_angle = (fsa >> 1) / 64.0
                end_angle = (lsa >> 1) / 64.0
                
                # Unwrap angle
                diff = end_angle - start_angle
                if diff < 0:
                    diff += 360
                    
                for i in range(lsn):
                    if i*2+1 >= len(raw_data): break
                    dist_l = raw_data[i*2]
                    dist_h = raw_data[i*2+1]
                    dist_mm = (dist_h << 8) | dist_l
                    dist_mm /= 4.0
                    
                    if dist_mm == 0: continue
                    
                    angle = start_angle + (diff / lsn) * i
                    # Normalize angle
                    angle = angle % 360
                    
                    # Store
                    with self.lock:
                        self.scan_data[int(angle)] = dist_mm / 1000.0
                
                # Periodically flush partial map to latest_scan
                if len(self.scan_data) > 100:
                   with self.lock:
                       self.latest_scan = [(a, d) for a, d in self.scan_data.items()]
                       self.scan_data = {} # Reset for next revolution

            except Exception:
                # Serial error, retry
                time.sleep(0.01)
