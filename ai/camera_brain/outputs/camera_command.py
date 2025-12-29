# ai/camera_brain/outputs/camera_command.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraCommand:
    """
    Canonical camera-space command emitted by Camera Brain.

    This is NOT hardware-level.
    This is NOT flight control.
    This is cinematic intent translated into normalized motion.
    """

    # Gimbal intent (normalized)
    pan: float              # -1.0 → +1.0
    tilt: float             # -1.0 → +1.0

    # Framing offsets (normalized screen space)
    framing_x: float        # -1.0 → +1.0
    framing_y: float        # -1.0 → +1.0

    # Motion intent (abstract, unitless)
    forward_motion: float   # 0.0 → 1.0

    # Lens / perception
    lens_compression: float # 0.3 → 1.0

    # Optional metadata (non-executed)
    confidence: Optional[float] = None
    intent_tag: Optional[str] = None