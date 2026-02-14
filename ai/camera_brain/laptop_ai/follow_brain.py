import time
import numpy as np

class PID:
    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = output_limits
        self.reset()
        
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt <= 0: return 0.0
        
        # P
        p_term = self.kp * error
        
        # I
        self.integral += error * dt
        # Anti-windup
        self.integral = max(min(self.integral, 1.0), -1.0) 
        i_term = self.ki * self.integral
        
        # D
        d_term = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        output = p_term + i_term + d_term
        return max(min(output, self.max_out), self.min_out)

class FollowBrain:
    """
    Core Visual Servoing Logic for "Follow Me" mode.
    Translates bounding box errors into Drone Velocity Commands (Body Frame).
    
    Coordinates:
    - X Error (Horizontal) -> Yaw Rate (Rotate to center)
    - Y Error (Vertical)   -> Altitude Velocity (Throttle) OR Gimbal Pitch
    - Size Error (Depth)   -> Forward/Backward Velocity (Pitch)
    """
    def __init__(self):
        # PID Tunings (Conservative start)
        # Yaw: Rotational speed (rad/s)
        self.pid_yaw = PID(kp=1.5, ki=0.0, kd=0.1, output_limits=(-1.0, 1.0))
        
        # Pitch: Forward/Back speed (m/s)
        self.pid_dist = PID(kp=1.0, ki=0.0, kd=0.1, output_limits=(-2.0, 2.0))
        
        # Throttle: Vertical speed (m/s)
        self.pid_alt = PID(kp=1.0, ki=0.0, kd=0.05, output_limits=(-1.0, 1.0))
        
        self.target_size = 0.2  # Target subject width as % of frame width (approx 3m away)
        self.active = False
        
    def update(self, subject_box, frame_size, current_depth=None):
        """
        Calculate velocity commands to keep subject in frame.
        
        Args:
            subject_box: (x, y, w, h) normalized (0.0-1.0)
            frame_size: (width, height) pixels (reference only)
            current_depth: Optional depth in meters (from ToF/Stereo)
            
        Returns:
            dict: {
                "vx": float, # Forward/Back (m/s)
                "vy": float, # Left/Right (m/s) - usually 0 for fixed wing/simple quad
                "vz": float, # Up/Down (m/s) - Negative is Up in NED? No, usually + is down. 
                             # We'll use standard ENU for internal logic then flip for MAVLink if needed.
                             # Let's stick to standard Drone: +Vx=Forward, +Vz=Down (climb is -Vz)
                "yaw_rate": float # rad/s
            }
        """
        if not subject_box:
            return None
            
        x, y, w, h = subject_box
        center_x = x + w/2
        center_y = y + h/2
        
        # 1. Yaw Error (Horizontal deviation from 0.5)
        # If center_x > 0.5 (subject is right), we need to Turn Right (+Yaw)
        error_x = (center_x - 0.5) * 2.0 # -1.0 to 1.0
        yaw_cmd = self.pid_yaw.update(error_x)
        
        # 2. Distance Error (Size deviation)
        # If w < target (subject too small/far), frame diff is positive -> Fly Forward (+Vx)
        # Target size 0.2 means subject takes 20% of screen.
        error_size = (self.target_size - w) * 5.0 
        vx_cmd = self.pid_dist.update(error_size)
        
        # 3. Altitude Error (Vertical deviation)
        # If center_y > 0.5 (subject is low), we need to Fly Down (+Vz) or Tilt Gimbal
        # Since we might not control gimbal here, let's adjust altitude carefully.
        # Actually in "Follow Me", usually we maintain altitude or match terrain.
        # Let's try to keep subject vertically centered.
        error_y = (center_y - 0.5) * 2.0
        vz_cmd = self.pid_alt.update(error_y)
        
        return {
            "vx": vx_cmd,       # Forward
            "vy": 0.0,          # Strafe (unused for now)
            "vz": vz_cmd,       # Vertical (Down positive)
            "yaw": yaw_cmd      # Rotate Right positive
        }
