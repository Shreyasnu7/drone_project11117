# utils.py
import json, os

# Check for Persistent Disk or Fallback
if os.path.exists("/mnt/data") and os.access("/mnt/data", os.W_OK):
    BASE_DATA_DIR = "/mnt/data"
else:
    # Fallback to local 'data' folder
    BASE_DATA_DIR = os.path.join(os.getcwd(), "storage")

JOBS_DIR = os.path.join(BASE_DATA_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def new_job(job_id, payload):
    path = f"{JOBS_DIR}/job_{job_id}.json"
    save_json(path, payload)
    return path

import time

def cleanup_job_files(directory, max_age_seconds=600):
    """Deletes files older than max_age_seconds."""
    now = time.time()
    if not os.path.exists(directory): return
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            try:
                if now - os.path.getmtime(path) > max_age_seconds:
                    os.remove(path)
                    print(f"🧹 Cleaned up: {path}")
            except Exception as e:
                print(f"Cleanup Error {path}: {e}")
