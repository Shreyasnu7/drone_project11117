# /ai/shot_intent/memory/intent_memory.py

from typing import Dict, Any
from .episodic_buffer import EpisodicIntentBuffer
from .preference_model import PreferenceModel
from .decay import apply_decay


class ShotIntentMemory:
    """
    Orchestrates episodic + long-term preference memory.
    """

    def __init__(self):
        self.episodic = EpisodicIntentBuffer()
        self.preferences = PreferenceModel()

    def record_outcome(
        self,
        intent_signature: Dict[str, Any],
        feedback: Dict[str, float],
    ):
        """
        Called AFTER execution.
        """
        self.episodic.add(intent_signature, feedback)

        # Convert feedback into preference deltas
        deltas = {
            "slow_reveal": feedback.get("reveal_satisfaction", 0.0),
            "high_motion": feedback.get("motion_satisfaction", 0.0),
            "camera_presence": feedback.get("presence_satisfaction", 0.0),
            "risk_comfort": feedback.get("comfort", 0.0),
            "imperfection_tolerance": feedback.get("stability_tolerance", 0.0),
        }

        self.preferences.update(deltas)

    def store(self, observation: Dict[str, Any]):
        """
        Store a raw observation (ShotIntent usage).
        """
        # We treat this as a 'neutral' experience until feedback arrives
        self.episodic.add(observation, {"neutral": 1.0})

    def get_context(self) -> Dict[str, Any]:
        """
        Provides memory context to reasoning layer.
        """
        prefs = apply_decay(self.preferences.get_bias())

        return {
            "recent_intents": self.episodic.recent(),
            "user_style_bias": prefs,
        }

# Alias for backward compatibility
IntentMemory = ShotIntentMemory