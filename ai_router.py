# ai_router.py
import os, uuid, shutil
from fastapi import APIRouter, UploadFile, File, Form
from api_schemas import TextRequest, VideoLinkRequest
from utils import save_json, new_job, cleanup_job_files
from fastapi import APIRouter
from cloud_ai.plan_generator import PlanGenerator
from cloud_ai.llm import ask_ai  # your existing LLM wrapper


planner = PlanGenerator()

router = APIRouter(tags=["AI"])

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------
#  Text → Job
# -----------------------
@router.post("/text")
async def ai_text(req: TextRequest):
    job_id = str(uuid.uuid4())
    payload = {
        "type": "text",
        "text": req.text,
        "user_id": req.user_id,
        "drone_id": req.drone_id,
        "include_vision": req.include_vision
    }
    new_job(job_id, payload)
    return {"job_id": job_id, "status": "queued"}

# -----------------------
#  Image upload (multiple)
# -----------------------
@router.post("/image")
async def ai_image(
    files: list[UploadFile] = File(...)
):
    # Auto-Cleanup: Remove files older than 5 mins (300s) to free space/ram
    cleanup_job_files(UPLOAD_DIR, 300)

    paths = []
    for file in files:
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)
        paths.append(filepath)

    job_id = str(uuid.uuid4())
    new_job(job_id, {
        "type": "image",
        "user_id": user_id,
        "drone_id": drone_id,
        "files": paths
    })

    return {"job_id": job_id, "uploaded": len(paths)}

# -----------------------
#  Video upload (≤100MB)
# -----------------------
@router.post("/video")
async def ai_video(
    user_id: str = Form(...),
    drone_id: str = Form(...),
    file: UploadFile = File(...)
):
    cleanup_job_files(UPLOAD_DIR, 300)
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = str(uuid.uuid4())
    new_job(job_id, {
        "type": "video",
        "user_id": user_id,
        "drone_id": drone_id,
        "file": filepath
    })

    return {"job_id": job_id, "status": "queued"}

# -----------------------
#  Video link
# -----------------------
@router.post("/video_link")
async def ai_video_link(req: VideoLinkRequest):
    job_id = str(uuid.uuid4())
    payload = {
        "type": "video_link",
        "user_id": req.user_id,
        "drone_id": req.drone_id,
        "url": req.url
    }
    new_job(job_id, payload)
    return {"job_id": job_id, "status": "queued"}