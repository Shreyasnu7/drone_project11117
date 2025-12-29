# File: laptop_ai/shot_metadata.py

class ShotMetadata:
    """
    Stores information about the user's cinematic intent.
    No flight control in this file.
    """

    def __init__(self):
        self.history = []

    def record(self, user_text, primitive, camera_plan):
        entry = {
            "intent": user_text,
            "primitive": primitive,
            "camera": camera_plan,
        }
        self.history.append(entry)
        return entry
