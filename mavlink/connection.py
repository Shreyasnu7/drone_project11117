from pymavlink import mavutil

# CHANGE THIS PORT TO YOUR REAL FC PORT
# Example:
# Linux: /dev/ttyUSB0
# Windows: COM8 -> /dev/ttyS8 (WSL)
FC_PORT = "/dev/ttyUSB0"
BAUD = 115200

def connect():
    print("Connecting to Flight Controller...")
    master = mavutil.mavlink_connection(FC_PORT, baud=BAUD)
    master.wait_heartbeat()
    print("Heartbeat received from FC")
    return master
