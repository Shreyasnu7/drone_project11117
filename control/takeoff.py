from mavlink.connection import connect
from mavlink.commands import takeoff

def execute(altitude=3):
    master = connect()
    takeoff(master, altitude)
    return {"status": "takeoff", "altitude": altitude}
