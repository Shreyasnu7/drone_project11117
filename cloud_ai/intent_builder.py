# server/cloud_ai/intent_builder.py

from cloud_ai.contracts.intent_schema import (
    CinematicIntent,
    EmotionalModel,
    CameraPlan,
    MotionPlan,
    CameraSelection,
    StabilizationStrategy,
    SafetyConstraints,
    ManualOverridePolicy,
)


class IntentBuilder:
    """
    Builds a baseline cinematic intent from inferred meaning.
    No creativity here — only structure.
    """

    def build_base_intent(self, inferred: dict) -> CinematicIntent:
        return CinematicIntent(
            intent_type=inferred["intent_type"],
            confidence=inferred.get("confidence", 0.85),
            priority=inferred.get("priority", "user_requested"),

            scene_context=inferred.get("scene_context", {}),
            subject_model=inferred.get("subject_model", {}),

            emotional_model=EmotionalModel(**inferred["emotional_model"]),

            camera_plan=CameraPlan(**inferred["camera_plan"]),

            motion_plan=MotionPlan(**inferred["motion_plan"]),

            camera_selection=CameraSelection(**inferred["camera_selection"]),

            stabilization_strategy=StabilizationStrategy(
                **inferred["stabilization_strategy"]
            ),

            safety_constraints=SafetyConstraints(
                **inferred["safety_constraints"]
            ),

            manual_override_policy=ManualOverridePolicy(
                **inferred["manual_override_policy"]
            ),

            sequence_plan=inferred.get("sequence_plan", []),
            fallbacks=inferred.get("fallbacks", {}),
        )