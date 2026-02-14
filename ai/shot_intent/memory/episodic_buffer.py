
from typing import Dict, Any, List
import time

class EpisodicIntentBuffer:
    """
    Stores recent intents and their outcomes (episodic memory).
    Used to prevent repetitive mistakes or repeating shots too soon.
    """
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def add(self, intent: Dict[str, Any], feedback: Dict[str, Any]):
        """
        Add an experience to the buffer.
        """
        experience = {
            "timestamp": time.time(),
            "intent": intent,
            "feedback": feedback
        }
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def recent(self, seconds=60) -> List[Dict[str, Any]]:
        """
        Get intents from the last N seconds.
        """
        now = time.time()
        return [
            item for item in self.buffer
            if (now - item["timestamp"]) < seconds
        ]
