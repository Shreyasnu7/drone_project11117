from fastapi import APIRouter, Response, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio
from ai.camera_brain.laptop_ai.director_core import Director

router = APIRouter()

# Global Frame Buffer (In-Memory for low latency)
# In production, use Redis. For a single-instance Render free tier, memory is fine.
# Global Frame Buffer (In-Memory for low latency)
# In production, use Redis. For a single-instance Render free tier, memory is fine.
_latest_frame = None
_director = None 

def get_latest_frame():
    return _latest_frame

async def ensure_director_started():
    global _director
    if _director is None:
        print("🎬 Starting Cinematic Director AI...")
        _director = Director(client_id="server_brain", simulate=False)
        _director.set_frame_source(get_latest_frame)
        # Scan '1000 files' or init heavy models here
        await _director.start() # Connects to Msg Client


@router.post("/video/frame")
async def upload_frame(file: UploadFile = File(...)):
    global _latest_frame
    global _latest_frame
    _latest_frame = await file.read()
    
    # Lazy Start the Brain on first frame
    if _director is None:
         asyncio.create_task(ensure_director_started())
         
    return {"status": "ok"}

async def frame_generator():
    global _latest_frame
    while True:
        if _latest_frame:
            # MJPEG Format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + _latest_frame + b'\r\n')
        await asyncio.sleep(0.05) # Max 20 FPS to save bandwidth

@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
