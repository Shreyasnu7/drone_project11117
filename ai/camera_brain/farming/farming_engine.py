from typing import Dict
from ..core.state import CameraBrainState


class FramingEngine:
    """
    Computes subject framing offsets using perceptual rules,
    not templates.
    """

    def compute(
        self,
        state: CameraBrainState,
        saliency: Dict[str, float],
        intent_strength: float,
    ) -> tuple[float, float]:
        """
        Returns normalized framing offsets (x, y).
        """

        subject_weight = saliency.get("subject", 0.7)
        environment_weight = saliency.get("environment", 0.3)

        lead_factor = state.subject_velocity * 0.4
        presence_bias = (state.camera_presence - 0.5) * 0.6

        offset_x = (
            lead_factor * subject_weight
            + presence_bias * environment_weight
        )

        offset_y = -0.15 * intent_strength

        state.framing_offset_x = offset_x
        state.framing_offset_y = offset_y

        return offset_x, offset_y