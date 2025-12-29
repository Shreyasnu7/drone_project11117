# laptop_ai/camera_brain/ai_camera_brain.py

class AICameraBrain:
    def __init__(self):
        from .ai_exposure import AIExposureController
        from .ai_color import AIColorController
        from .ai_focus import AIFocusEngine
        from .ai_filters import AIFilterEngine
        from .ai_framerate import AIFrameRateBrain
        from .ai_lens import AILensBrain
        from .ai_zoom import AIZoomComposer
        from .ai_gimbal_controller import AIGimbalController

        self.exposure = AIExposureController()
        self.color = AIColorController()
        self.focus = AIFocusEngine()
        self.filters = AIFilterEngine()
        self.fps_brain = AIFrameRateBrain()
        self.lens = AILensBrain()
        self.zoom = AIZoomComposer()
        self.gimbal = AIGimbalController()

    def decide(self, user_text, frame, motion_level, bbox, fusion_state):
        exposure = self.exposure.compute_exposure(frame)
        color = self.color.compute_color_profile(frame)
        focus = self.focus.estimate_focus(frame)
        zoom = self.zoom.compute_zoom(bbox, frame.shape[:2])
        fps = self.fps_brain.decide_fps(user_text, motion_level)
        fov = self.lens.choose_fov(user_text)

        return {
            "exposure": exposure,
            "color": color,
            "focus_metric": focus,
            "zoom": zoom,
            "fps": fps,
            "fov": fov
        }
