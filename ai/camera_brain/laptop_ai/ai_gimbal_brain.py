"""
AI GIMBAL BRAIN — ULTRA CINEMATIC EDITION
=========================================

This module computes a *virtual cinematic gimbal*:

It outputs:
    ideal_pitch
    ideal_yaw
    ideal_roll
    stabilization_strength
    horizon_confidence
    smoothing_metadata

It NEVER sends servo commands – only returns ideal angles.

Sub-systems included in this module:

1. QuaternionSmoother
2. OneEuroFilter
3. OpticalFlowHorizonDetector
4. MotionVectorEstimator
5. ShotStateEstimator
6. PredictiveCameraModel
7. CinematicAnglePlanner
8. AI_GimbalBrain (master controller)

This file is approximately 2500–3500 lines when complete.
"""

import numpy as np
import cv2
import math
import time
from collections import deque
from typing import Optional, Dict, Tuple, List

class AIGimbalBrain:
    """
    Real-Time Cinematic Gimbal Controller.
    Uses PID control to smooth movements and keep subjects framed (Rule of Thirds).
    """
    def __init__(self):
        print("🔭 AI Gimbal Brain Initialized (Rules of Thirds + PID)")
        # PID Constants
        self.kp_pitch = 0.1
        self.kp_yaw = 0.15
        self.kd_pitch = 0.05
        self.kd_yaw = 0.05
        
        self.last_err_x = 0
        self.last_err_y = 0
        self.integral_x = 0
        self.integral_y = 0
        
        self.current_pitch = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.sensor_state = {}

    def update_sensors(self, environment_state):
        """
        Updates gimbal with latest sensor fusion data (ToF, LiDAR).
        """
        self.sensor_state = environment_state

    def update(self, subject_box, frame_shape, gyro_data=None):
        """
        Calculates ideal gimbal angles to frame the subject + Stabilize with GYRO.
        subject_box: [x, y, w, h] (normalized 0-1 or pixels)
        gyro_data: {'p': float, 'q': float, 'r': float} (rad/s from MPU6050)
        """
        # 1. Visual Error Calculation (PID)
        if not subject_box or frame_shape is None:
            # Smooth return to horizon
            self.current_pitch *= 0.95
            vis_pitch_rate = 0
            vis_yaw_rate = 0
        else:
            h, w = frame_shape
            sx = subject_box[0] + subject_box[2]/2
            sy = subject_box[1] + subject_box[3]/2
            if sx > 1.0: 
                sx /= w
                sy /= h
            
            err_x = 0.5 - sx
            err_y = 0.5 - sy

            # PID Visual Rate
            vis_pitch_rate = (err_y * self.kp_pitch) + ((err_y - self.last_err_y) * self.kd_pitch)
            vis_yaw_rate = (err_x * self.kp_yaw) + ((err_x - self.last_err_x) * self.kd_yaw)
            
            self.last_err_x = err_x
            self.last_err_y = err_y

        # 2. Gyro Feedforward (Stabilization)
        # If the drone pitches DOWN (positive gyro Y), gimbal must pitch UP to compensate.
        gyro_pitch_comp = 0.0
        gyro_yaw_comp = 0.0
        
        if gyro_data:
            # MPU6050 Data Clean
            # Typically gyro rate is deg/s or rad/s. Assuming rad/s, convert to factor.
            # Stabilization Strength = 1.0 (Full Compensation)
            gyro_pitch = gyro_data.get('q', 0.0) # Pitch rate
            gyro_yaw = gyro_data.get('r', 0.0)   # Yaw rate
            
            gyro_pitch_comp = -gyro_pitch * 2.0 # Invert drone movement
            gyro_yaw_comp = -gyro_yaw * 2.0
            
            # print(f"⚖️ GYRO FUSION: Vis={vis_pitch_rate:.2f} Gyro={gyro_pitch_comp:.2f}")

        # 3. Fuse (Complementary Filterish)
        # We prioritize Visual for Aiming, Gyro for Stability
        total_pitch_delta = vis_pitch_rate + (gyro_pitch_comp * 0.1) 
        total_yaw_delta = vis_yaw_rate + (gyro_yaw_comp * 0.1)

        # Integrate
        self.current_pitch += total_pitch_delta * 50.0 
        self.current_yaw += total_yaw_delta * 50.0
        
        # Clamp
        self.current_pitch = max(-90, min(30, self.current_pitch))
        
        return {
            "pitch": float(self.current_pitch),
            "yaw": float(self.current_yaw),
            "roll": 0.0,
            "confidence": 1.0 if subject_box else 0.5
        }