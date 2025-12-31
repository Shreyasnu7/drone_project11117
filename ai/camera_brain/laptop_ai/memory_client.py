# File: laptop_ai/memory_client.py
# Simple memory client for the FastAPI memory endpoints.

import requests
from config import API_BASE, AUTH_TOKEN

HEADERS = {"Authorization": AUTH_TOKEN, "Content-Type": "application/json"}

def read_memory(user_id, drone_id):
    try:
        r = requests.post(f"{API_BASE}/memory/read", json={"user_id":user_id,"drone_id":drone_id}, headers=HEADERS, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print("memory read failed", e)
    return {}

def write_memory(user_id, drone_id, key, value):
    try:
        r = requests.post(f"{API_BASE}/memory/write", json={"user_id":user_id,"drone_id":drone_id,"key":key,"value":value}, headers=HEADERS, timeout=6)
        return r.status_code == 200
    except Exception as e:
        print("memory write failed", e)
        return False
