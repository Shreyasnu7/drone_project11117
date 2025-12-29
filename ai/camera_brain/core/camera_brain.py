from .state import CameraBrainState
from ..framing.framing_engine import FramingEngine
from ..motion.anticipation import MotionAnticipator
from ..motion.motion_planner import MotionPlanner
from ..lens.lens_model import LensModel
from ..outputs.camera_command import CameraCommand
from ai.camera_brain.core.output_adapter import CameraOutputAdapter

class CameraBrain:
    def __init__(self):
        self.output_adapter = CameraOutputAdapter()

    def step(self, frame, telemetry):
        # Existing logic (unchanged)
        gimbal_pan = ...
        gimbal_tilt = ...
        framing_x = ...
        framing_y = ...
        motion_energy = ...
        lens_value = ...

        return self.output_adapter.adapt(
            gimbal_pan=gimbal_pan,
            gimbal_tilt=gimbal_tilt,
            framing_offset_x=framing_x,
            framing_offset_y=framing_y,
            motion_energy=motion_energy,
            lens_model_value=lens_value,
            intent_tag="cinematic_follow",
        )
