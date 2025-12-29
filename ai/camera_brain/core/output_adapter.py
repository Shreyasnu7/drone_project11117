# ai/camera_brain/core/output_adapter.py

from ai.camera_brain.outputs.camera_command import CameraCommand


class CameraOutputAdapter:
    """
    Converts internal camera-brain signals into a stable CameraCommand.
    """

    def adapt(
        self,
        *,
        gimbal_pan: float,
        gimbal_tilt: float,
        framing_offset_x: float,
        framing_offset_y: float,
        motion_energy: float,
        lens_model_value: float,
        confidence: float | None = None,
        intent_tag: str | None = None,
    ) -> CameraCommand:
        return CameraCommand(
            pan=float(gimbal_pan),
            tilt=float(gimbal_tilt),
            framing_x=float(framing_offset_x),
            framing_y=float(framing_offset_y),
            forward_motion=float(motion_energy),
            lens_compression=float(lens_model_value),
            confidence=confidence,
            intent_tag=intent_tag,
        )