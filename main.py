from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import Routers
from ai_router import router as ai_router
from memory_router import router as memory_router
from plan_router import router as plan_router
from ws_router import router as ws_router
from misc_router import router as misc_router
from ai_command_router import router as ai_command_router
from auth_router import router as auth_router
from logs_router import router as logs_router
from weather_router import router as weather_router

app = FastAPI(title="AI Drone Server (V3)", version="3.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_router, prefix="/auth")
app.include_router(ai_router, prefix="/ai")
app.include_router(memory_router, prefix="/memory")
app.include_router(plan_router, prefix="/plan")
app.include_router(ws_router, prefix="/ws")
app.include_router(ai_command_router)
app.include_router(logs_router)
app.include_router(weather_router)
app.include_router(misc_router)
from video_router import router as video_router
app.include_router(video_router)

@app.get("/")
def root():
    return {"status": "AI Drone Server Online", "version": "3.0", "real_features": True}