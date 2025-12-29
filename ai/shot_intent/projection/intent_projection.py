# /ai/shot_intent/projection/intent_projection.py

from typing import Dict, Any
from .semantic_axes import SemanticAxes


class IntentProjector:
    """
    Projects free-form cinematic meaning into latent semantic axes.
    This layer does NOT understand language.
    """

    def project(self, semantic_reasoning: Dict[str, Any]) -> SemanticAxes:
        """
        Convert reasoning into latent axes using semantic presence,
        not keyword matching.
        """

        def presence(key: str) -> float:
            return 1.0 if key in semantic_reasoning.get("visual_language", "").lower() else 0.0

        pursuit_intensity = (
            0.5 + 0.3 * presence("pursuit")
        )

        reveal_timing = (
            0.7 if "reveal" in semantic_reasoning.get("story_arc", "") else 0.3
        )

        emotional_pressure = (
            0.6 + 0.2 * len(semantic_reasoning.get("viewer_emotion", []))
        )

        visual_authority = (
            0.6 if "authority" in semantic_reasoning.get("visual_language", "") else 0.4
        )

        chaos_vs_control = (
            0.3 if "smooth" in semantic_reasoning.get("visual_language", "") else 0.6
        )

        perceived_risk = (
            0.7 if semantic_reasoning.get("risk_attitude", "") == "assertive" else 0.4
        )

        return SemanticAxes(
            pursuit_intensity=min(pursuit_intensity, 1.0),
            reveal_timing=min(reveal_timing, 1.0),
            emotional_pressure=min(emotional_pressure, 1.0),
            visual_authority=min(visual_authority, 1.0),
            chaos_vs_control=min(chaos_vs_control, 1.0),
            perceived_risk=min(perceived_risk, 1.0),
        )