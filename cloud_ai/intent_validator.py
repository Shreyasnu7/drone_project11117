# server/cloud_ai/intent_validator.py

from cloud_ai.contracts.intent_schema import CinematicIntent


class IntentValidator:
    """
    Ensures intent is complete, sane, and safe.
    """

    def validate(self, intent: CinematicIntent) -> CinematicIntent:
        if intent.confidence < 0.3:
            raise ValueError("Intent confidence too low")

        if not intent.camera_selection.primary:
            raise ValueError("Primary camera not defined")

        if intent.safety_constraints.min_battery_percent < 10:
            raise ValueError("Unsafe battery threshold")

        return intent