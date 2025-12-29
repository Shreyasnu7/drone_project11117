
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EmotionalModel(BaseModel):
    vector: Dict[str, float] = {}
    peak_allowed: bool = False

class CameraPlan(BaseModel):
    shot_energy: float = 0.5
    framing: str = "wide"

class MotionPlan(BaseModel):
    path_type: str = "linear"
    speed: float = 1.0

class CameraSelection(BaseModel):
    primary: str = "main"
    secondary: Optional[str] = None

class StabilizationStrategy(BaseModel):
    mode: str = "standard"

class SafetyConstraints(BaseModel):
    min_battery_percent: int = 20
    max_distance: float = 100.0

class ManualOverridePolicy(BaseModel):
    allow_interrupt: bool = True

class CinematicIntent(BaseModel):
    intent_type: str
    confidence: float = 0.85
    priority: str = "user_requested"
    
    scene_context: Dict[str, Any] = {}
    subject_model: Dict[str, Any] = {}
    
    emotional_model: EmotionalModel
    camera_plan: CameraPlan
    motion_plan: MotionPlan
    camera_selection: CameraSelection
    stabilization_strategy: StabilizationStrategy
    safety_constraints: SafetyConstraints
    manual_override_policy: ManualOverridePolicy
    
    sequence_plan: List[Dict[str, Any]] = []
    fallbacks: Dict[str, Any] = {}