# File: laptop_ai/esp32_driver.py
# CRITICAL HARWARE REQUIREMENT:
# ESP32 V202+ uses Serial.setTimeout(5) and readStringUntil('\n').
# WE MUST SEND A NEWLINE ('\n') OR THE DRONE WILL LAG/FREEZE.
# DO NOT REMOVE THE + '\n' logic.

import serial
import json
import threading
import time
import logging

# Config
SERIAL_PORT = "/dev/ttyS2" # Radxa Debug UART (GPIOX_9/10)
BAUD_RATE = 115200

# --- MODE CONSTANTS (Mapped to Firmware) ---
M_BOOT = 0
M_SAFE_BREATHE = 1
M_RGB_BREATHE = 2
M_POLICE = 3      # V226 Pure Logic
M_CYLON = 4
M_ASCEND = 10
M_DESCEND = 11
M_FWD = 12
M_BACK = 13
M_LEFT = 14
M_RIGHT = 15
M_HOVER = 16
M_AUDI_WIPE = 21
M_MATRIX = 22
M_BRAKING = 30
M_ERROR = 99
M_DEMO_CYCLE = 100

class ESP32Driver:
    def __init__(self, port=SERIAL_PORT):
        self.ser = None
        self.port_name = port
        self.latest_telemetry = {
            "t1": -1, "t2": -1, "t3": -1, "t4": -1,
            "ax": 0, "ay": 0, "az": 0
        }
        self.running = True
        self.lock = threading.Lock()
        
        # Auto-connect loop
        self.thread = threading.Thread(target=self._serial_worker, daemon=True)
        self.thread.start()
        
    def _serial_worker(self):
        while self.running:
            try:
                if self.ser is None:
                    logging.info(f"Connecting to ESP32 on {self.port_name}...")
                    try:
                        self.ser = serial.Serial(self.port_name, BAUD_RATE, timeout=0.1) # low timeout
                        logging.info("ESP32 Connected!")
                    except:
                        pass
                
                if self.ser and self.ser.is_open:
                    # Read Line (Blocking with timeout)
                    if self.ser.in_waiting:
                        try:
                            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                            if line.startswith('{'):
                                msg = json.loads(line)
                                with self.lock:
                                    # Merge update
                                    self.latest_telemetry.update(msg)
                            elif line.startswith("DEBUG"):
                                logging.debug(f"ESP32: {line}")
                        except Exception as e:
                            logging.warning(f"Parse Error: {e}")
                    else:
                        time.sleep(0.005) # Yield CPU
                else:
                    time.sleep(1)
                    
            except serial.SerialException:
                logging.error("Serial Disconnected. Retrying...")
                if self.ser:
                    self.ser.close()
                self.ser = None
                time.sleep(2)
            except Exception as e:
                logging.error(f"Serial Error: {e}")
                time.sleep(1)

    def get_telemetry(self):
        with self.lock:
            return self.latest_telemetry.copy()
            
    def get_sensors(self):
        """Helper for AI usage"""
        t = self.get_telemetry()
        return [t.get('t1', -1), t.get('t2', -1), t.get('t3', -1), t.get('t4', -1)]   

    def set_gimbal(self, pitch, yaw):
        """
        Send Gimbal positions (-90 to 90 degrees)
        """
        payload = {"gim": [int(pitch), int(yaw)]}
        self._send_json(payload)
        
    def set_mode(self, mode_id):
        """
        Set LED/Flight Mode (0-100)
        """
        logging.info(f"Setting Mode: {mode_id}")
        payload = {"mode": int(mode_id)}
        self._send_json(payload)
    
    def set_stabilization(self, enabled):
        """
        Enable/Disable Physics Stabilization
        """
        payload = {"stab": bool(enabled)}
        self._send_json(payload)

    def _send_json(self, payload):
        if self.ser and self.ser.is_open:
            try:
                # CRITICAL: NEWLINE TERMINATION
                msg = json.dumps(payload) + '\n' 
                self.ser.write(msg.encode('utf-8'))
            except Exception as e:
                logging.error(f"Write Failed: {e}")

    def close(self):
        self.running = False
        self.thread.join()
        if self.ser:
            self.ser.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    driver = ESP32Driver() 
    
    print("ESP32 Bridge Active. Cycling Verified Modes...")
    try:
        modes = [
            (M_POLICE, "POLICE"),
            (M_CYLON, "CYLON"),
            (M_AUDI_WIPE, "AUDI WIPE"),
            (M_ASCEND, "ASCEND"),
            (M_HOVER, "HOVER")
        ]
        
        while True:
            for mode, name in modes:
                print(f"---> MODE: {name}")
                driver.set_mode(mode)
                
                # Wait 2 seconds and print sensors
                for _ in range(20):
                    time.sleep(0.1)
                
                sens = driver.get_sensors()
                print(f"     Sensors: {sens}")
                
    except KeyboardInterrupt:
        driver.close()
