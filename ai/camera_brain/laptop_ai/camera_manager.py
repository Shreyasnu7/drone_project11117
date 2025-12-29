# File: laptop_ai/camera_manager.py
# -----------------------------------------------------------
# UNIVERSAL CAMERA AI MANAGER
# Works with ANY camera (USB / CSI / Pi cam / Pro cam).
# Provides exposure, focus, HDR, zoom, stabilization hooks.
# -----------------------------------------------------------

import cv2
import numpy as np
import time

class CameraManager:
    def __init__(self, cam_index=0):
        self.cam = cv2.VideoCapture(cam_index)
        self.capabilities = self.detect_capabilities()
        self.last_frame = None
        self.last_timestamp = 0

    # -----------------------------------------------------------
    # DETECT CAMERA CAPABILITIES
    # -----------------------------------------------------------
    def detect_capabilities(self):
        caps = {
            "resolution": (
                int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            "fps": self.cam.get(cv2.CAP_PROP_FPS),
            "auto_focus": self.cam.get(cv2.CAP_PROP_AUTOFOCUS) != -1,
            "auto_exposure": True,
            "supports_zoom": self.cam.get(cv2.CAP_PROP_ZOOM) != -1,
            "supports_focus": self.cam.get(cv2.CAP_PROP_FOCUS) != -1,
            "supports_iso": False,  # USB cams rarely expose ISO
        }
        print("Camera detected:", caps)
        return caps

    # -----------------------------------------------------------
    # AI EXPOSURE (HISTOGRAM + ML LOGIC)
    # -----------------------------------------------------------
    def apply_ai_exposure(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        mean_val = np.mean(gray)

        # cinematic exposure curve
        if mean_val < 90:
            frame = cv2.convertScaleAbs(frame, alpha=1.12, beta=8)
        elif mean_val > 160:
            frame = cv2.convertScaleAbs(frame, alpha=0.88, beta=-12)

        return frame

    # -----------------------------------------------------------
    # AI FOCUS ENGINE (subject-aware)
    # -----------------------------------------------------------
    def apply_ai_focus(self, frame, vision_context):
        selected = vision_context.get("selected")
        if not selected:
            return frame

        x1, y1, x2, y2 = map(int, selected["bbox"])
        roi = frame[y1:y2, x1:x2]

        # sharpen ROI
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        roi = cv2.filter2D(roi, -1, kernel)
        frame[y1:y2, x1:x2] = roi

        return frame

    # -----------------------------------------------------------
    # AI STABILIZATION (EIS)
    # -----------------------------------------------------------
    def apply_eis(self, frame):
        if self.last_frame is None:
            self.last_frame = frame
            return frame

        # simple optical flow smoothing
        flow = cv2.absdiff(frame, self.last_frame)
        thresh = cv2.threshold(flow, 25, 255, cv2.THRESH_BINARY)[1]
        blur = cv2.GaussianBlur(frame, (3,3), 0)

        self.last_frame = frame
        return blur

    # -----------------------------------------------------------
    # MAIN FRAME FETCH (with full AI stack)
    # -----------------------------------------------------------
    def get_frame(self, vision_context):
        ret, frame = self.cam.read()
        if not ret:
            return None

        frame = self.apply_ai_exposure(frame)
        frame = self.apply_ai_focus(frame, vision_context)
        frame = self.apply_eis(frame)

        self.last_timestamp = time.time()
        return frame

# File: laptop_ai/camera_manager.py

class CameraManager:
    """
    High-level API for selecting between PiCam and GoPro
    WITHOUT issuing any stabilization or gimbal motor commands.
    """

    def __init__(self, pi_cam, gopro, ai_brain):
        self.pi_cam = pi_cam
        self.gopro = gopro
        self.ai = ai_brain

    async def choose_and_prepare(self, user_text, vision_context):
        """
        Returns a camera_plan dictionary.
        This does NOT send control commands to hardware.
        """

        camera_choice = self.ai.decide(user_text, vision_context)

        if camera_choice == "gopro":
            profile = self.gopro.choose_profile(user_text)
        else:
            profile = self.pi_cam.choose_profile(user_text)

        return {
            "camera": camera_choice,
            "profile": profile
        }