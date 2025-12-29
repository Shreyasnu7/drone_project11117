# /ai/shot_intent/memory/decay.py

from typing import Dict


def apply_decay(
    preferences: Dict[str, float],
    decay_rate: float = 0.002,
) -> Dict[str, float]:
    """
    Slowly decay preferences toward neutral (0.5).
    Prevents fossilized taste.
    """
    decayed = {}
    for k, v in preferences.items():
        decayed[k] = v + decay_rate * (0.5 - v)
    return decayed