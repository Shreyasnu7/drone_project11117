
from typing import TypedDict, List, Optional, Union

class CinematicShot(TypedDict):
    action: str  # ORBIT, FOLLOW, DOLLY, CRANE, HOVER
    subject_id: Optional[int]
    duration_s: float
    speed_ms: float
    height_m: float
    radius_m: Optional[float]
    angle_deg: Optional[float]
    style: str # ACTION, DRAMATIC, VLOG, SMOOTH

class DirectorPlan(TypedDict):
    reasoning: str
    shots: List[CinematicShot]
    safety_cushion_m: float
    lighting_preference: str
