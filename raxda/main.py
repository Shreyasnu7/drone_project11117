from mavlink.connection import connect
from mavlink.commands import arm, takeoff, land
from cloud.client import get_command
import time

DRONE_ID = "DRONE001"

vehicle = connect()

while True:
    cmd = get_command(DRONE_ID)

    if not cmd:
        time.sleep(1)
        continue

    action = cmd.get("action")

    if action == "ARM":
        arm(vehicle)

    elif action == "TAKEOFF":
        takeoff(vehicle, cmd.get("altitude", 5))

    elif action == "LAND":
        land(vehicle)

    time.sleep(1)