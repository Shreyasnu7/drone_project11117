# laptop_ai/camera_brain/ai_gimbal_controller.py
from pymavlink import mavutil
import math

class AIGimbalController:

    def compute_angles(self, subject_pos, drone_pos):
        dx = subject_pos[0] - drone_pos[0]
        dy = subject_pos[1] - drone_pos[1]
        dz = subject_pos[2] - drone_pos[2]

        yaw = math.degrees(math.atan2(dy, dx))
        pitch = -math.degrees(math.atan2(dz, (dx**2 + dy**2)**0.5))

        return pitch, yaw

    def smoothing(self, last, target, factor=0.1):
        return last + (target - last) * factor
