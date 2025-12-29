from mavlink.connection import connect
from mavlink.commands import land

def execute():
    master = connect()
    land(master)
    return {"status": "landed"}
