from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
import json
import os
import time

from utils import BASE_DATA_DIR

router = APIRouter()
LOGS_FILE = os.path.join(BASE_DATA_DIR, "flight_logs.json")

class FlightLog(BaseModel):
    date: str
    duration: str
    max_alt: float
    max_speed: float
    battery_used: float
    location: str

@router.get("/logs")
async def get_logs():
    if not os.path.exists(LOGS_FILE):
        return []
    try:
        with open(LOGS_FILE, "r") as f:
            return json.load(f)
    except:
        return []

@router.post("/logs")
async def save_log(log: FlightLog):
    logs = []
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, "r") as f:
                logs = json.load(f)
        except:
            logs = []
    
    # Add ID and Timestamp
    entry = log.dict()
    entry["id"] = int(time.time())
    logs.insert(0, entry) # Newest first
    
    with open(LOGS_FILE, "w") as f:
        json.dump(logs, f, indent=2)
    
    return {"status": "saved", "id": entry["id"]}
