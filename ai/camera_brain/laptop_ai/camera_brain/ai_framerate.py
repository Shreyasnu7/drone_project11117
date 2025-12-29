class AIFrameRateBrain:

    def decide_fps(self, user_text, motion_level):
        if "slow motion" in user_text.lower():
            return 120
        if motion_level > 50:
            return 60
        return 24
