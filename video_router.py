from fastapi import APIRouter, Response, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio
# from ai.camera_brain.laptop_ai.director_core import Director (Removed: Laptop Only)

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
        try:
             await _director.start() # Connects to Msg Client
        except Exception as e:
             print(f"⚠️ Director Start Error (Video will continue): {e}")


@router.post("/video/frame")
async def upload_frame(file: UploadFile = File(...)):
    global _latest_frame
    # Format Check
    if file.content_type not in ["image/jpeg", "image/png"]:
        print(f"⚠️ Video Upload: Invalid Content Type {file.content_type}")
    
    content = await file.read()
    _latest_frame = content
    print(f"📸 Frame Rx: {len(content)} bytes")
    
    # NOTE: We do NOT start the Director here. 
    # The Laptop AI runs externally and consumes/commands via WS.
    # The Server is strictly a relay.
         
    return {"status": "ok"}

async def frame_generator():
    global _latest_frame
    while True:
        if _latest_frame:
            # MJPEG Format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + _latest_frame + b'\r\n')
            # print("Sent Frame", end='\r') # Debug noise
        else:
             print("⚠️ No Frame in Buffer (Generator)", end='\r')
        await asyncio.sleep(0.05) # Max 20 FPS to save bandwidth

@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/video/current")
async def get_current_frame():
    """
    Robust Snapshot Endpoint for App Polling.
    Solves 'Black Screen' issue where Image.network fails to handle MJPEG streams.
    """
    global _latest_frame
    if _latest_frame:
        return Response(content=_latest_frame, media_type="image/jpeg")
    else:
        # Return a placeholder or 404 if no frame yet
        return Response(content=b'', status_code=503)
