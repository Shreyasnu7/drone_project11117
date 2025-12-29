# memory_router.py
import os, json
from utils import BASE_DATA_DIR
from fastapi import APIRouter
from api_schemas import MemoryWrite, MemoryRead

MEMORY_DIR = os.path.join(BASE_DATA_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

router = APIRouter(tags=["Memory"])

def mem_path(user, drone):
    return f"{MEMORY_DIR}/{user}_{drone}.json"

@router.post("/write")
def write_memory(req: MemoryWrite):
    path = mem_path(req.user_id, req.drone_id)
    mem = {}
    if os.path.exists(path):
        mem = json.load(open(path))
    mem[req.key] = req.value
    json.dump(mem, open(path, "w"))
    return {"status": "stored"}

@router.post("/read")
def read_memory(req: MemoryRead):
    path = mem_path(req.user_id, req.drone_id)
    if not os.path.exists(path):
        return {}
    mem = json.load(open(path))
    if req.key:
        return {req.key: mem.get(req.key)}
    return mem
