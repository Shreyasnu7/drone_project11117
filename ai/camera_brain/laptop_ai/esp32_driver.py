import socket
import json
import threading
import time
import logging

# Config
ESP32_IP = "192.168.4.2" # Usually the first client connected to Hotspot, but we might need to scan or use broadcast. 
# actually ESP32 connects to Radxa (192.168.4.1). Radxa can see ESP32 IP from ARP or just broadcast.
# Ideally ESP32 should listen on a port too. 
# Firmware says `udp.beginPacket(radxa_ip, localPort)` so it knows where to send.
# But for us to send TO ESP32, we need its IP.
# For now, let's assume valid IP or use Multicast if we could. 
# Simple hack: The first packet we receive from ESP32 will tell us its IP.

LISTEN_PORT = 8888
SEND_PORT = 8888

class ESP32Driver:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", LISTEN_PORT))
        self.sock.settimeout(0.1)
        
        self.esp32_addr = None # Discovered automatically
        self.latest_telemetry = {}
        self.running = True
        
        self.lock = threading.Lock()
        
        # Start Listener
        self.thread = threading.Thread(target=self._listener_worker, daemon=True)
        self.thread.start()
        
    def _listener_worker(self):
        logging.info(f"ESP32 Listener started on port {LISTEN_PORT}")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                
                # Update known ESP32 Address
                if self.esp32_addr != addr:
                    self.esp32_addr = addr
                    logging.info(f"ESP32 Connected from {addr}")
                
                # Parse JSON
                try:
                    msg = json.loads(data.decode('utf-8'))
                    with self.lock:
                        self.latest_telemetry = msg
                except json.JSONDecodeError:
                    pass
                    
            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"ESP32 UDP Error: {e}")
                time.sleep(1)

    def get_telemetry(self):
        with self.lock:
            return self.latest_telemetry.copy()
            
    def set_gimbal(self, pitch, yaw):
        """
        Send Gimbal positions (-90 to 90 degrees)
        """
        if self.esp32_addr is None:
            return
            
        payload = {"gim": [int(pitch), int(yaw)]}
        self._send_json(payload)
        
    def set_led(self, color):
        """
        Color: "RED", "BLUE", "GREEN"
        """
        if self.esp32_addr is None:
            return

        payload = {"led": color}
        self._send_json(payload)

    def _send_json(self, payload):
        try:
            msg = json.dumps(payload).encode('utf-8')
            self.sock.sendto(msg, self.esp32_addr)
        except Exception as e:
            logging.error(f"Failed to send to ESP32: {e}")

    def close(self):
        self.running = False
        self.thread.join()
        self.sock.close()

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    driver = ESP32Driver()
    
    print("Waiting for ESP32 connection (turn it on)...")
    try:
        while True:
            tel = driver.get_telemetry()
            if tel:
                print(f"Telem: {tel}")
                # Echo test: limit gimbal move
                driver.set_led("BLUE")
            else:
                print("No Data")
            time.sleep(0.5)
    except KeyboardInterrupt:
        driver.close()
