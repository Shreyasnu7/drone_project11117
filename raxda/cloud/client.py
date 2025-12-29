import requests

CLOUD_URL = "https://drone-server-r0qe.onrender.com/api/command"

def get_command(drone_id):
    r = requests.get(f"{CLOUD_URL}/{drone_id}")
    if r.status_code == 200:
        return r.json()
    return None