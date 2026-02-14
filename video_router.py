from fastapi import APIRouter, Response, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio
# from ai.camera_brain.laptop_ai.director_core import Director (Removed: Laptop Only)

router = APIRouter()

import os

FRAMES_DIR = "/tmp/frames"
if not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR, exist_ok=True)
FRAME_PATH = os.path.join(FRAMES_DIR, "latest.jpg")

_director = None 

def get_latest_frame():
    try:
        with open(FRAME_PATH, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None

async def ensure_director_started():
    global _director
    if _director is None:
        print("🎬 Starting Cinematic Director AI...")
        # Local Import to avoid circular dep
        try:
             # from ai.camera_brain.laptop_ai.director_core import Director
             pass 
        except:
             pass 

@router.post("/video/frame")
async def upload_frame(file: UploadFile = File(...)):
    # Format Check
    if file.content_type not in ["image/jpeg", "image/png"]:
        print(f"⚠️ Video Upload: Invalid Content Type {file.content_type}")
    
    content = await file.read()
    
    # ATOMIC WRITE (Write temp -> Rename) to prevent partial reads
    temp_path = FRAME_PATH + ".tmp"
    with open(temp_path, "wb") as f:
        f.write(content)
    os.replace(temp_path, FRAME_PATH)
    
    print(f"📸 Frame Updated: {len(content)} bytes")
    return {"status": "ok"}

async def frame_generator():
    while True:
        frame = get_latest_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
             # yield placeholder?
             pass
        await asyncio.sleep(0.05) 

@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/video/current")
async def get_current_frame():
    """
    Robust Snapshot Endpoint for App Polling.
    Reads from Shared File System (Works across Gunicorn Workers).
    """
    frame = get_latest_frame()
    if frame:
        return Response(content=frame, media_type="image/jpeg")
    else:
        return Response(content=b'', status_code=503)
