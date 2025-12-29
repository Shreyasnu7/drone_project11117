# File: laptop_ai/motion_curve.py
# Creates smooth cinematic 3D Bézier curves for drone flight.

import numpy as np


class BezierCurve:
    """
    A 4-point cubic Bezier curve in 3D.
    p0 = start
    p1 = control 1
    p2 = control 2
    p3 = end
    """

    def __init__(self, p0, p1, p2, p3):
        self.p0 = np.array(p0, dtype=float)
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.p3 = np.array(p3, dtype=float)

        # Precompute for speed
        self._pre = {
            "a": -self.p0 + 3*self.p1 - 3*self.p2 + self.p3,
            "b": 3*self.p0 - 6*self.p1 + 3*self.p2,
            "c": -3*self.p0 + 3*self.p1,
            "d": self.p0,
        }

    def point(self, t):
        """
        Returns a 3-D point on the curve.
        """
        t = np.clip(t, 0.0, 1.0)
        a = self._pre["a"]
        b = self._pre["b"]
        c = self._pre["c"]
        d = self._pre["d"]
        return ((a*t + b)*t + c)*t + d

    def tangent(self, t):
        """
        Returns the derivative vector (direction).
        """
        t = np.clip(t, 0.0, 1.0)
        a = self._pre["a"]
        b = self._pre["b"]
        c = self._pre["c"]
        return (3*a*t + 2*b)*t + c

    def sample(self, n=50):
        """
        Returns a list of points along the curve.
        """
        return np.array([self.point(i/(n-1)) for i in range(n)])

    def length(self, n=100):
        """
        Estimate curve length by sampling.
        """
        pts = self.sample(n)
        return np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))

    def subdivide(self, t):
        """
        Splits the curve into two curves at position t.
        Useful for advanced replanning.
        """
        P0 = self.p0
        P1 = self.p1
        P2 = self.p2
        P3 = self.p3

        # De Casteljau algorithm
        A = P0*(1-t) + P1*t
        B = P1*(1-t) + P2*t
        C = P2*(1-t) + P3*t

        D = A*(1-t) + B*t
        E = B*(1-t) + C*t

        F = D*(1-t) + E*t  # point on curve

        left = BezierCurve(P0, A, D, F)
        right = BezierCurve(F, E, C, P3)

        return left, right


# ----------------------------------------------------------------------
# Helper that Director uses to build cinematic curve
# ----------------------------------------------------------------------

def build_cinematic_bezier(start_pos, end_pos, lift=1.2):
    """
    Constructs a smooth cinematic Bezier between two 3-D points.

    • Adds vertical lift for beauty
    • Control points at 1/3 and 2/3 along path
    • Always produces a curve even when motion is small
    """

    p0 = np.array(start_pos, dtype=float)
    p3 = np.array(end_pos, dtype=float)

    vec = p3 - p0
    dist = np.linalg.norm(vec)

    if dist < 1e-6:
        # Degenerate movement: return hover curve
        return BezierCurve(p0, p0, p0, p0)

    direction = vec / dist

    # Control points — slightly raised upward
    p1 = p0 + direction * (dist * 0.33) + np.array([0, 0, lift])
    p2 = p0 + direction * (dist * 0.66) + np.array([0, 0, lift])

    return BezierCurve(p0, p1, p2, p3)
