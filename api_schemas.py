# api_schemas.py
# server/api_schemas.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class DronePlan(BaseModel):
    thought_process: Optional[str] = None
    reasoning: Optional[str] = None
    action: str
    params: Optional[Dict[str, Any]] = {}
    emotional_context: Optional[Dict[str, float]] = {}
    confidence: float = 1.0

MAX_IMAGE_UPLOAD = 25
MAX_VIDEO_SIZE_MB = 100

class TextRequest(BaseModel):
    user_id: str
    drone_id: str
    text: str
    include_vision: bool = False

class VideoLinkRequest(BaseModel):
    user_id: str
    drone_id: str
    url: str

class AIPlan(BaseModel):
    primitive: dict
    confidence: float = Field(default=1.0)
    vision_used: bool = False
    memory_used: bool = False

class MemoryWrite(BaseModel):
    user_id: str
    drone_id: str
    key: str
    value: str

class MemoryRead(BaseModel):
    user_id: str
    drone_id: str
    key: Optional[str] = None
