# /ai/shot_intent/memory/preference_model.py

from typing import Dict
import math


class PreferenceModel:
    """
    Models user cinematic taste over time.
    This is NOT a classifier.
    """

    def __init__(self):
        self.preferences: Dict[str, float] = {
            "slow_reveal": 0.5,
            "high_motion": 0.5,
            "camera_presence": 0.5,
            "risk_comfort": 0.4,
            "imperfection_tolerance": 0.3,
        }

    def update(self, deltas: Dict[str, float], learning_rate: float = 0.05):
        """
        Gently nudge preferences based on outcome.
        """
        for key, delta in deltas.items():
            if key not in self.preferences:
                continue

            self.preferences[key] += learning_rate * delta
            self.preferences[key] = min(
                max(self.preferences[key], 0.0),
                1.0
            )

    def get_bias(self) -> Dict[str, float]:
        """
        Returns stable user style bias.
        """
        return dict(self.preferences)