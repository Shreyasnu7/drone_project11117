from pymavlink import mavutil

fc = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
fc.wait_heartbeat()
print("FC connected")

def arm():
    fc.mav.command_long_send(
        fc.target_system,
        fc.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0
    )

def takeoff(alt=3):
    fc.mav.command_long_send(
        fc.target_system,
        fc.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0, alt
    )