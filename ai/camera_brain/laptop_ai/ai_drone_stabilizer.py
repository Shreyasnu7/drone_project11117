# File: laptop_ai/ai_drone_stabilizer.py
import numpy as np
import cv2

class AIDroneStabilizer:

    def __init__(self):
        self.last_gyro = np.zeros(3)

    def apply(self, frame, gyro):
        gx, gy, gz = gyro["gx"], gyro["gy"], gyro["gz"]
        # Placeholder: very light stabilization
        M = np.float32([[1, 0, gx*0.002], [0, 1, gy*0.002]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))