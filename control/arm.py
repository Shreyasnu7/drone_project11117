from mavlink.connection import connect
from mavlink.commands import arm

def execute():
    master = connect()
    arm(master)
    return {"status": "armed"}
