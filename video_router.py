from fastapi import APIRouter, Response, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio

router = APIRouter()

# Global Frame Buffer (In-Memory for low latency)
# In production, use Redis. For a single-instance Render free tier, memory is fine.
_latest_frame = None

@router.post("/video/frame")
async def upload_frame(file: UploadFile = File(...)):
    global _latest_frame
    _latest_frame = await file.read()
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
