# /ai/shot_intent/projection/constraint_resolver.py

from .semantic_axes import SemanticAxes
from ..schema.shot_intent_schema import ShotIntent


class ConstraintResolver:
    """
    Resolves semantic axes into concrete, bounded ShotIntent.
    """

    def resolve(self, axes: SemanticAxes) -> ShotIntent:
        motion_energy = min(
            0.4 + 0.6 * axes.pursuit_intensity,
            1.0
        )

        imperfection_tolerance = max(
            0.05,
            1.0 - axes.chaos_vs_control
        )

        risk_tolerance = min(
            axes.perceived_risk,
            0.7  # hard system ceiling
        )

        return ShotIntent(
            shot_type="tracking",  # placeholder, refined in Camera Brain
            primary_emotion="tension",
            secondary_emotion="awe",
            subject_priority=["dynamic_subject", "environment"],
            motion_energy=motion_energy,
            pacing="natural",
            camera_presence="expressive",
            lens_feel="compressed",
            imperfection_tolerance=imperfection_tolerance,
            risk_tolerance=risk_tolerance,
            reference_memory=[],
            user_style_bias={}
        )