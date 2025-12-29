# File: laptop_ai/ai_motion_blur_controller.py
class AIMotionBlurController:

    def __init__(self):
        self.shutter_speed = 1/120

    def choose_shutter(self, motion_level):
        """
        motion_level ∈ [0,1]
        """
        if motion_level < 0.3:
            return 1/60
        if motion_level < 0.7:
            return 1/120
        return 1/240