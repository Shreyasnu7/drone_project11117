def arm(vehicle):
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        400, 0,
        1, 0, 0, 0, 0, 0
    )

def takeoff(vehicle, altitude=5):
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        22, 0,
        0, 0, 0, 0, 0, altitude
    )

def land(vehicle):
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        21, 0,
        0, 0, 0, 0, 0, 0
    )