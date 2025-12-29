class AILensBrain:
    def choose_fov(self, user_text):
        if "wide" in user_text:
            return "wide"
        if "super" in user_text:
            return "superview"
        return "linear"
