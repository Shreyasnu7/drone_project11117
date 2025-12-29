# File: laptop_ai/ultra_director.py

import time
import numpy as np
from laptop_ai.motion_curve import BezierCurve
from laptop_ai.obstacle_warp import ObstacleWarp
from laptop_ai.flow_field import FlowFieldAvoidance
from laptop_ai.safety_envelope import SafetyEnvelope

class UltraDirector:
    """
    UltraDirector = The cinematic shot decider + curve generator.
    Decides soft vs hard replan, builds curves, warps for obstacles,
    generates yaw curves, and exports a point-by-point streamable path.
    """

    def __init__(self):
        self.active_curve = None
        self.yaw_curve = None
        self.curve_start_time = None
        self.curve_duration = 4.0     # default cinematic duration
        self.last_target = None
        self.last_user_text = ""
        self.warp_engine = ObstacleWarp(safety_radius=3.0)
        self.flow = FlowFieldAvoidance()
        self.safety = SafetyEnvelope()

    # -----------------------------------------------------------
    # C1. Decide Planning Strategy
    # -----------------------------------------------------------
    def decide_planning_mode(self, user_text, subject_speed, obstacle_density):
        """
        Returns: "soft_replan", "hard_replan", or "lock_curve"
        """

        intent = user_text.lower()

        # Strong keywords → forced hard replan
        if any(x in intent for x in ["instant", "cut", "jump", "urgent", "now"]):
            return "hard_replan"

        # Slow cinematic language → prefer soft
        if any(x in intent for x in ["smooth", "slow", "cinematic", "gradual"]):
            return "soft_replan"

        # High speed subject → must react fast
        if subject_speed > 4.0:
            return "hard_replan"

        # Very busy environment → don't freeze path
        if obstacle_density > 0.4:
            return "hard_replan"

        # If nothing changed → maintain old curve
        if self.last_target is not None:
            return "lock_curve"

        return "soft_replan"

    # -----------------------------------------------------------
    # C2. Build base Bézier motion path
    # -----------------------------------------------------------
    def build_curve(self, start, target):
        p0 = np.array(start)
        p3 = np.array(target)

        direction = p3 - p0
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return None

        direction = direction / dist

        # Cinematic elevated control points
        p1 = p0 + direction * dist * 0.33 + np.array([0, 0, 1.0])
        p2 = p0 + direction * dist * 0.66 + np.array([0, 0, 1.0])

        return BezierCurve(p0, p1, p2, p3)

    # -----------------------------------------------------------
    # C3. Build yaw orientation curve
    # -----------------------------------------------------------
    def build_yaw_curve(self, start_yaw, target_pos, subject_pos):
        """
        Drone yaw faces subject, not movement direction.
        """

        target_vec = np.array(subject_pos) - np.array(target_pos)
        angle = np.degrees(np.arctan2(target_vec[1], target_vec[0]))
        angle = angle % 360

        # Simple linear interpolation for yaw
        return lambda t: start_yaw + (angle - start_yaw) * t

    # -----------------------------------------------------------
    # C4. Final motion planner entry point
    # -----------------------------------------------------------
    def plan(self, user_text, start_pos, subject_pos, obstacles):
        """
        Main entry point for motion planning.
        Called by Director AFTER camera planning.
        """

        subject_speed = np.linalg.norm(np.array(subject_pos) - np.array(self.last_target or subject_pos))
        obstacle_density = len(obstacles) / 20.0  # simple heuristic

        mode = self.decide_planning_mode(user_text, subject_speed, obstacle_density)

        if mode == "lock_curve" and self.active_curve is not None:
            # Continue existing curve smoothly
            return {
                "mode": "lock_curve",
                "duration": self.curve_duration
            }

        # Hard or soft: always build new curve
        base_curve = self.build_curve(start_pos, subject_pos)
        # First warp with obstacle warp engine (macro safety)
        macro_safe = self.warp_engine.warp_curve(base_curve, obstacles)

        # Then flow-field blending (micro smooth avoidance)
        flow_sampler = self.flow.warp_curve(macro_safe, obstacles)

        # -----------------------------
        # SAFETY ENVELOPE CHECK
        # -----------------------------
       if not self.safety.evaluate(flow_sampler, obstacles):
          self.active_curve = None
          return {
              "curve": None,
              "mode": "unsafe_hover",
              "duration": 1.0
          }


        # Replace Bézier curve with flow-field sampler
        self.active_curve = flow_sampler
        self.active_curve = warped_curve
        self.curve_start_time = time.time()
        self.curve_duration = 4.0 if mode == "soft_replan" else 2.0
        self.last_target = subject_pos
        self.last_user_text = user_text

        # Build yaw curve (face subject)
        self.yaw_curve = self.build_yaw_curve(
            start_yaw=0,
            target_pos=start_pos,
            subject_pos=subject_pos
        )

        return {
            "mode": mode,
            "duration": self.curve_duration
        }

    # -----------------------------------------------------------
    # C5. Stream curve point to drone-safe executor
    # -----------------------------------------------------------
    def sample_curve(self, now=None):
        """
        Director calls this every 50ms and forwards to safe_executor.stream_curve_point().
        Returns: (position[x,y,z], yaw_deg)
        """

        if self.active_curve is None:
            return None, None

        now = now or time.time()
        t = (now - self.curve_start_time) / self.curve_duration
        t = max(0, min(1, t))

        pos = self.active_curve.point(t)
        yaw = self.yaw_curve(t) if self.yaw_curve else 0

        # Curve finished
        if t >= 1.0:
            self.active_curve = None

        return pos, yaw
