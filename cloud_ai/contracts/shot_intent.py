from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class ShotIntent(BaseModel):
    # Core identity
    intent_id: str
    source: str               # "laptop_ai"
    timestamp: float

    # Cinematic intent
    shot_type: str            # tracking, orbit, static, reveal, chase
    emotion: str              # awe, tension, calm, triumph
    camera: str               # internal | external | both
    priority: str             # subject | environment | balanced

    # Motion intent (normalized -1 to 1)
    pan: float
    tilt: float
    forward_motion: float
    altitude_change: float
    yaw_rate: float

    # Camera intent
    framing: Dict[str, Any]   # rule_of_thirds, center_bias, lead_room
    zoom: float
    exposure_bias: float

    # Safety & execution hints
    max_speed: float
    allow_aggressive_motion: bool

    # Metadata
    debug: Optional[Dict[str, Any]] = None