from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraBrainState:
    """
    Persistent perceptual state across frames.
    """

    subject_velocity: float = 0.0
    subject_distance: float = 1.0
    framing_offset_x: float = 0.0
    framing_offset_y: float = 0.0

    last_motion_vector_x: float = 0.0
    last_motion_vector_y: float = 0.0

    camera_presence: float = 0.5
    anticipation_bias: float = 0.0