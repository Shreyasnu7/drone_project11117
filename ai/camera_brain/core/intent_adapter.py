# ai/camera_brain/core/intent_adapter.py

from ai.shot_intent.schema import ShotIntent


class IntentToCameraParams:
    """
    Converts ShotIntent into camera-brain internal targets.
    """

    def adapt(self, intent: ShotIntent) -> dict:
        params = {}

        # Shot type → motion style
        params["motion_style"] = {
            "tracking": "follow",
            "orbit": "circular",
            "reveal": "push_out",
            "static": "locked",
            "chase": "aggressive_follow",
        }.get(intent.shot_type, "follow")

        # Emotion → motion + framing
        params["motion_energy"] = {
            "awe": 0.25,
            "tension": 0.6,
            "calm": 0.15,
            "triumph": 0.5,
            "melancholy": 0.1,
        }.get(intent.emotion, 0.3)

        # Lens feel
        params["lens_compression"] = {
            "wide": 0.4,
            "neutral": 0.7,
            "long": 0.95,
        }.get(intent.lens_feel, 0.7)

        # Presence
        params["camera_smoothness"] = 1.0 - intent.imperfection_tolerance

        return params