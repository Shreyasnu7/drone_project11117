import serial
import time
import json
import threading
import sys
try:
    import keyboard # pip install keyboard if needed, else use simple input
except ImportError:
    pass

# CONFIGURATION
SERIAL_PORT = "COM3" 
BAUD_RATE = 115200

def read_telemetry(ser):
    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('DEBUG') or line.startswith('CMD'):
                    print(f"\n[DRONE] {line}")
        except:
            pass
        time.sleep(0.01)

def main():
    print("--- CYBER DRONE: FLIGHT SIMULATOR ---")
    print("Controls:")
    print("  W/S: Pitch Forward/Back (White/Red LEDs)")
    print("  A/D: Roll Left/Right    (Orange Indicators)")
    print("  Q/E: Ascend/Descend     (Green/Purple Flow)")
    print("  SPACE: Hover            (Breathing White)")
    print("  1-5:  Demo Modes")
    
    port = input(f"Port [{SERIAL_PORT}]: ").strip() or SERIAL_PORT
    
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"Connected to {port}!")
    except Exception as e:
        print(f"Failed: {e}")
        return

    t = threading.Thread(target=read_telemetry, args=(ser,), daemon=True)
    t.start()
    
    print("\nREADY TO FLY. Type command (e.g., 'w') and Enter.")
    print("(For real WASD gameplay, we need the 'keyboard' library, but for now just type letters)")
    
    while True:
        key = input("Input > ").lower().strip()
        data = {}
        
        # PHYSICS SIMULATION MAPPING
        if key == 'w': data = {"ax": -5.0, "ay": 0, "az": 9.8}  # Pitch Fwd
        elif key == 's': data = {"ax": 5.0, "ay": 0, "az": 9.8} # Pitch Back
        elif key == 'a': data = {"ax": 0, "ay": 5.0, "az": 9.8} # Roll Left
        elif key == 'd': data = {"ax": 0, "ay": -5.0, "az": 9.8} # Roll Right
        elif key == 'q': data = {"ax": 0, "ay": 0, "az": 15.0}  # Ascend (>12)
        elif key == 'e': data = {"ax": 0, "ay": 0, "az": 5.0}   # Descend (<8)
        elif key == ' ': data = {"ax": 0, "ay": 0, "az": 9.8}   # Hover
        
        # DEMO MODES
        elif key == '1': data = {"mode": 25} # Police
        elif key == '2': data = {"mode": 24} # Cylon
        elif key == '3': data = {"mode": 21} # Audi
        elif key == 'x': break
        
        if data:
            js = json.dumps(data) + "\n"
            ser.write(js.encode())
            print(f"Sent: {js.strip()}")

if __name__ == "__main__":
    main()
