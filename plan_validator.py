# File: server/plan_validator.py

class PlanValidator:
    """
    Ensures a plan is safe before sending to Radxa.
    Does NOT allow direct motor direction or unsafe paths.
    """

    MAX_ALT = 30
    MIN_ALT = 1
    MAX_DIST = 50
    MAX_DURATION = 12

    # server/plan_validator.py

    def validate_plan(plan: dict) -> bool:
     allowed_actions = {"follow", "orbit", "hover", "track", "move"}
     return plan.get("action") in allowed_actions

    def validate(self, primitive: dict):
        """
        Returns (True, primitive) or (False, reason)
        """

        if primitive.get("action") not in ("HOVER", "MOVE_TO", "FOLLOW", "ORBIT", "TRACK_PATH"):
            return False, "Unknown action"

        curve = primitive.get("plan_curve", {})
        duration = curve.get("duration", 0)

        if duration > self.MAX_DURATION:
            return False, "duration_exceeds_limit"

        # Basic sanity: ensure coordinates aren't extreme
        cps = curve.get("control_points", [])
        for p in cps:
            if abs(p[0]) > self.MAX_DIST or abs(p[1]) > self.MAX_DIST:
                return False, "cp_out_of_bounds"
            if p[2] < self.MIN_ALT or p[2] > self.MAX_ALT:
                return False, "altitude_out_of_range"

        return True, primitive
