from pymavlink import mavutil

def connect():
    print("Connecting to Flight Controller...")
    vehicle = mavutil.mavlink_connection('/dev/ttyUSB0', baud=921600)
    vehicle.wait_heartbeat()
    print("Heartbeat received")
    return vehicle