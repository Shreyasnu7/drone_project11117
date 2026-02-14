from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import uuid
from dotenv import load_dotenv


load_dotenv() # Load environment variables from .env file



# --- ROUTER IMPORTS ---
from ai_router import router as ai_router
from memory_router import router as memory_router
from plan_router import router as plan_router
from ws_router import router as ws_router
from ai_command_router import router as ai_command_router
from auth_router import router as auth_router
from video_router import router as video_router
from weather_router import router as weather_router
from misc_router import router as misc_router
from logs_router import router as logs_router


# --- App Setup ---
app = FastAPI(title="AI Drone Server (Unified)", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REGISTER ROUTERS ---
# 1. AI & Drone Control
app.include_router(ai_router, prefix="/ai")
app.include_router(memory_router, prefix="/memory")
app.include_router(plan_router, prefix="/plan")
app.include_router(ws_router, prefix="/ws")
app.include_router(ai_command_router)

# 2. Auth Routes
# 2. Auth Routes
# 2. Auth Routes
app.include_router(auth_router)
app.include_router(video_router)
app.include_router(weather_router)
app.include_router(misc_router)
app.include_router(logs_router) # Real Flight Logs Logic

# --- Startup Event ---
# --- AI DIRECTOR INTEGRATION ---


@app.on_event("startup")
async def startup_event():
    print("\n\n" + "="*50)
    print("--> SERVER STARTUP: AUTH + AI ROUTERS LOADED <--")
    # AI Director is REMOVED from Server Process.
    # It runs on Laptop/Edge and connects via WebSockets.

    print("--> /session/create [READY]")
    print("--> /ai/command     [READY]")
    print("--> /ws/connect     [READY]")
    print("="*50 + "\n\n")

@app.get("/")
def read_root():
    return {"status": "Quantum Drone Server Online", "version": "3.0.0"}

# /media endpoint is handled by misc_router (GET /media) — DO NOT shadow it here

if __name__ == "__main__":
    import uvicorn
    print("--> STARTING UVICORN SERVER ON 0.0.0.0:8000 <--")
    uvicorn.run(app, host="0.0.0.0", port=8000)