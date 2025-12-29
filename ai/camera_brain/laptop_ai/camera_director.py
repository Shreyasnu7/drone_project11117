# File: laptop_ai/camera_director.py

import numpy as np

class CameraDirector:
    def __init__(self):
        pass

    def compute_gimbal_target(self, subject_box, frame_size):
        """
        Compute ideal gimbal pitch/yaw to keep subject:
        - on rule-of-thirds
        - stable for cinematic framing
        """
        W, H = frame_size

        x1, y1, x2, y2 = subject_box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        desired_x = W * 0.33  # left third
        desired_y = H * 0.40  # slightly above center

        dx = (cx - desired_x) / W
        dy = (cy - desired_y) / H

        yaw_adj = -dx * 15.0
        pitch_adj = dy * 10.0

        return {"yaw": yaw_adj, "pitch": pitch_adj}
