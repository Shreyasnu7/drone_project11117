# /ai/shot_intent/projection/semantic_axes.py

from dataclasses import dataclass


@dataclass
class SemanticAxes:
    """
    Latent semantic dimensions extracted from reasoning.
    These are NOT commands, only perceptual intensities.
    """

    pursuit_intensity: float        # feeling of chase / pursuit
    reveal_timing: float            # delayed vs immediate payoff
    emotional_pressure: float       # psychological tension
    visual_authority: float         # camera dominance vs neutrality
    chaos_vs_control: float         # instability vs smoothness
    perceived_risk: float           # how dangerous it should feel