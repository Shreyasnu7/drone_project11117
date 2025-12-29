# File: laptop_ai/ai_stabilizer.py
"""
AI Stabilizer (gyro + optical flow fusion)
- Accepts successive frames (and optional gyro readings) and estimates a smooth transform
  for digital stabilization (translation + rotation).
- Provides smoothed output for gimbal brain or post-processing warp.

Public class: AIStabilizer
Methods:
    - reset()
    - update(frame, gyro=None) -> transform dict {"dx":..., "dy":..., "dtheta":..., "ts":...}
    - get_smoothed_transform() -> last smoothed transform
Notes:
    - gyro: optional dict {"gx":..., "gy":..., "gz":...} in deg/sec or rad/s depending on your telemetry.
Dependencies: numpy, opencv-python
"""
from __future__ import annotations
import time
from collections import deque
from typing import Optional, Dict

import numpy as np
import cv2

class AIStabilizer:
    def __init__(self, history_len: int = 8, smooth_alpha: float = 0.6):
        self.prev_gray = None
        self.history = deque(maxlen=history_len)
        self.last_transform = {"dx":0.0,"dy":0.0,"dtheta":0.0,"ts":time.time()}
        self.smooth_alpha = smooth_alpha

    def reset(self):
        self.prev_gray = None
        self.history.clear()
        self.last_transform = {"dx":0.0,"dy":0.0,"dtheta":0.0,"ts":time.time()}

    def _estimate_flow_transform(self, prev_gray, cur_gray):
        # use ORB + match + estimateAffinePartial2D or use optical flow
        # We'll use cv2.calcOpticalFlowPyrLK on good features
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=7)
        if p0 is None:
            return {"dx":0.0,"dy":0.0,"dtheta":0.0}
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None, winSize=(21,21), maxLevel=3)
        # Select good points
        good_old = p0[st==1]
        good_new = p1[st==1]
        if len(good_old) < 6:
            return {"dx":0.0,"dy":0.0,"dtheta":0.0}
        m, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if m is None:
            return {"dx":0.0,"dy":0.0,"dtheta":0.0}
        dx = float(m[0,2])
        dy = float(m[1,2])
        # approximate rotation angle from matrix
        dtheta = float(np.arctan2(m[1,0], m[0,0]))  # radians
        return {"dx":dx, "dy":dy, "dtheta":dtheta}

    def update(self, frame: np.ndarray, gyro: Optional[Dict]=None) -> Dict:
        """
        Update stabilizer with new frame (+ optional gyro).
        Return smoothed transform to apply to this frame to stabilize.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transform = {"dx":0.0,"dy":0.0,"dtheta":0.0, "ts": time.time()}

        if self.prev_gray is None:
            self.prev_gray = gray
            self.history.append(transform)
            self.last_transform = transform
            return transform

        flow_t = self._estimate_flow_transform(self.prev_gray, gray)
        # merge with gyro if provided (simple complementary)
        if gyro:
            # assuming gyro['gz'] is yaw rate in rad/s and dt approximated by frame interval
            # This is a lightweight fusion; replace with Kalman filter if you have accurate dt
            gz = gyro.get("gz", 0.0)
            # convert gz influence to rotation delta (small angle)
            dt = 1/30.0
            gyro_dtheta = gz * dt
            # fuse: prefer optical flow for translation, gyro for rotation if large
            dtheta = 0.5*flow_t["dtheta"] + 0.5*gyro_dtheta
        else:
            dtheta = flow_t["dtheta"]

        transform["dx"] = flow_t["dx"]
        transform["dy"] = flow_t["dy"]
        transform["dtheta"] = dtheta
        transform["ts"] = time.time()

        # smoothing
        last = self.last_transform
        alpha = self.smooth_alpha
        smoothed = {
            "dx": alpha*transform["dx"] + (1-alpha)*last["dx"],
            "dy": alpha*transform["dy"] + (1-alpha)*last["dy"],
            "dtheta": alpha*transform["dtheta"] + (1-alpha)*last["dtheta"],
            "ts": transform["ts"]
        }

        self.history.append(smoothed)
        self.prev_gray = gray
        self.last_transform = smoothed
        return smoothed

    def get_smoothed_transform(self):
        return self.last_transform

# Example usage (requires camera)
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    st = AIStabilizer()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tr = st.update(frame)
        print("Transform:", tr)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
