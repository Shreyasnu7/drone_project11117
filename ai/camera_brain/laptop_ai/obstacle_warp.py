# File: laptop_ai/obstacle_warp.py
# Warps cinematic Bezier curves away from obstacles without breaking the shot.

import numpy as np


class ObstacleWarp:
    """
    Safety wrapper that bends a Bezier curve around obstacles.
    It never destroys the cinematic motion – only gently reshapes it.

    Obstacles should be supplied as:
        [{"pos": [x,y,z], "radius": r}, ...]
    """

    def __init__(self, safety_radius=2.5, max_push=4.0, samples=80):
        self.safety_radius = float(safety_radius)
        self.max_push = float(max_push)
        self.samples = int(samples)

    # ------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------
    def warp_curve(self, curve, obstacles):
        """
        Inputs:
            curve = BezierCurve
            obstacles = [{"pos":[x,y,z], "radius": r}, ...]

        Returns:
            A NEW BezierCurve where control points have been slightly moved
            to avoid collision but retain cinematic flow.
        """

        if not obstacles:
            return curve  # nothing to modify

        # Copy control points (p0, p1, p2, p3)
        p = [curve.p0.copy(), curve.p1.copy(), curve.p2.copy(), curve.p3.copy()]

        # Sample midpoints along the curve
        ts = np.linspace(0, 1, self.samples)

        for t in ts:
            point = curve.point(t)
            for obs in obstacles:
                obs_p = np.array(obs["pos"], dtype=float)
                obs_r = float(obs["radius"]) + self.safety_radius

                dist = np.linalg.norm(point - obs_p)

                # If too close → apply warp
                if dist < obs_r:
                    strength = (obs_r - dist) / obs_r
                    push_dir = (point - obs_p)
                    if np.linalg.norm(push_dir) < 1e-6:
                        continue
                    push_dir = push_dir / np.linalg.norm(push_dir)

                    # Push strength decays near endpoints (p0 and p3)
                    distance_from_start = t
                    distance_from_end = 1 - t
                    taper = min(distance_from_start, distance_from_end)

                    # Influence p1 and p2 mostly (do not break start/end positions)
                    delta = push_dir * self.max_push * strength * taper

                    # Apply deformation
                    p[1] += delta * 0.8
                    p[2] += delta

        # Build new warped curve
        from laptop_ai.motion_curve import BezierCurve
        warped = BezierCurve(p[0], p[1], p[2], p[3])
        return warped

    # ------------------------------------------------------------
    # Debug visualizer (optional)
    # ------------------------------------------------------------
    def debug_sample_points(self, curve):
        """Return curve sampled points for plotting/debug."""
        ts = np.linspace(0, 1, self.samples)
        pts = np.array([curve.point(t) for t in ts])
        return pts
