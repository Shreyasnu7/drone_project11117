# ================================================================
# File: laptop_ai/ai_shot_planner.py  (PART 1/3)
# High-level interpretation of user request → cinematic shot intent
#
# Provides:
#   ShotIntentParser
#   ShotFeasibilityModel
#   ShotPlanner
#
# Next parts (2/3 and 3/3) will add:
#   - Shot graph
#   - Dynamic transitions
#   - Camera style intelligence
#   - Modifier stack
#   - Full UltraDirector integration hooks
#
# SAFE MODULE: No motor control, only logic & scene analysis.
# ================================================================

import re
import numpy as np
import time


# -----------------------------------------------------------
# Utility
# -----------------------------------------------------------

def _contains(text, *keywords):
    text = text.lower()
    return any(k.lower() in text for k in keywords)


# -----------------------------------------------------------
# Shot Intent Parser
# -----------------------------------------------------------

class ShotIntent:
    """Parsed representation of user's cinematic request."""

    def __init__(self):
        self.raw_text = ""
        self.shot_type = "generic"
        self.modifiers = []
        self.subject_hint = None
        self.urgency = 0.0
        self.creativity = 0.0
        self.camera_keywords = []
        self.motion_keywords = []
        self.speed_factor = 1.0
        self.distance_hint = None
        self.height_hint = None

    def to_dict(self):
        return {
            "raw_text": self.raw_text,
            "shot_type": self.shot_type,
            "modifiers": self.modifiers,
            "subject_hint": self.subject_hint,
            "urgency": self.urgency,
            "creativity": self.creativity,
            "camera_keywords": self.camera_keywords,
            "motion_keywords": self.motion_keywords,
            "speed_factor": self.speed_factor,
            "distance_hint": self.distance_hint,
            "height_hint": self.height_hint,
        }


class ShotIntentParser:
    """
    Natural-language → structured cinematic intent.
    Does not choose a camera or curve — only interprets meaning.
    """

    SHOT_KEYWORDS = {
        "orbit": ["orbit", "circle around", "360", "round shot"],
        "follow": ["follow", "track", "chase"],
        "push_in": ["push in", "move closer", "approach"],
        "pull_back": ["pull back", "move away", "reveal backward"],
        "reveal": ["reveal", "unveil"],
        "topdown": ["topdown", "bird view", "straight down"],
        "side_tracking": ["side track", "parallel follow", "strafe"],
        "establishing": ["establish", "wide view", "overview"],
        "closeup": ["closeup", "tight shot"],
    }

    SUBJECT_PATTERNS = [
        r"the ([a-zA-Z0-9 ]+)",  # "the red car"
        r"focus on ([a-zA-Z0-9 ]+)",
        r"track ([a-zA-Z0-9 ]+)",
        r"follow ([a-zA-Z0-9 ]+)",
    ]

    SPEED_HINTS = {
        "slow": 0.5,
        "slowly": 0.5,
        "fast": 1.6,
        "quick": 1.3,
        "aggressive": 1.8,
        "smooth": 0.8,
        "cinematic": 0.7,
    }

    def parse(self, text: str) -> ShotIntent:
        intent = ShotIntent()
        intent.raw_text = text

        t = text.lower()

        # Determine shot type
        for shot, keys in self.SHOT_KEYWORDS.items():
            if _contains(t, *keys):
                intent.shot_type = shot

        # Extract subject hint
        for pattern in self.SUBJECT_PATTERNS:
            m = re.search(pattern, t)
            if m:
                intent.subject_hint = m.group(1).strip()
                break

        # Extract speed modifiers
        for k, v in self.SPEED_HINTS.items():
            if k in t:
                intent.speed_factor *= v
                intent.modifiers.append(k)

        # Extract distance hints
        if "close" in t:
            intent.distance_hint = 2.0
        elif "far" in t or "wide" in t:
            intent.distance_hint = 6.0

        # Extract height hints
        if "low angle" in t or "low shot" in t:
            intent.height_hint = -0.6
        if "high angle" in t or "rise up" in t or "ascending" in t:
            intent.height_hint = 1.5

        # Camera keywords
        if "slow motion" in t or "slowmo" in t:
            intent.camera_keywords.append("slowmo")
        if "smooth" in t:
            intent.camera_keywords.append("smooth")

        # Creativity / urgency scoring
        intent.creativity = 1.0 if _contains(t, "creative", "cinematic", "dramatic") else 0.3
        intent.urgency = 0.8 if _contains(t, "urgent", "quickly", "immediately") else 0.3

        return intent


# -----------------------------------------------------------
# Feasibility Model
# Determines whether a requested shot is safe & possible
# -----------------------------------------------------------

class ShotFeasibility:
    def __init__(self):
        self.is_feasible = True
        self.reasons = []
        self.score = 1.0  # 1.0 = fully feasible, <0.3 means unsafe/impossible

    def add_penalty(self, amount, reason):
        self.score -= amount
        self.reasons.append(reason)
        if self.score < 0.2:
            self.is_feasible = False


class ShotFeasibilityModel:
    """
    Evaluate if a requested shot is possible in the current scene
    using:
       - subject detection quality
       - obstacle density
       - drone speed limits
       - environment lighting
       - subject velocity
    """

    def evaluate(self, intent: ShotIntent, vision_context: dict):
        f = ShotFeasibility()

        selected = vision_context.get("selected")
        obstacles = vision_context.get("obstacles", [])
        light = vision_context.get("lighting", 0.6)

        # No subject found
        if intent.shot_type in ("follow", "orbit", "side_tracking", "closeup"):
            if selected is None:
                f.add_penalty(0.7, "no_subject_detected")

        # Too many obstacles → penalize complex motion
        if len(obstacles) > 5:
            f.add_penalty(0.4, "heavy_obstacle_density")

        # Poor lighting hurts complex shots
        if light < 0.2:
            f.add_penalty(0.3, "low_light")

        # If subject moving fast, orbiting becomes harder
        if intent.shot_type == "orbit" and selected is not None:
            speed = np.linalg.norm(selected.get("vel", [0,0]))
            if speed > 8.0:
                f.add_penalty(0.4, "subject_moving_too_fast_for_orbit")

        return f


# -----------------------------------------------------------
# Shot Planner Core
# Produces the planned shot description
# -----------------------------------------------------------

class ShotPlanner:
    """
    High-level cinematic shot planner.
    Part 1: basic shot selection & parameter generation.
    Part 2/3: cinematic graph, transitions, modifiers, style engine.
    """

    DEFAULT_DISTANCES = {
        "follow": 3.0,
        "orbit": 4.0,
        "reveal": 6.0,
        "push_in": 2.0,
        "pull_back": 5.0,
        "side_tracking": 3.5,
        "establishing": 12.0,
        "closeup": 1.5,
    }

    def __init__(self):
        self.parser = ShotIntentParser()
        self.feasibility_model = ShotFeasibilityModel()

    def plan(self, text, vision_context):
        """
        Returns:
            {
              "shot_type": ...,
              "subject_id": ...,
              "distance": ...,
              "speed": ...,
              "height_offset": ...,
              "camera_style": ...,
              "feasibility": {},
            }
        """

        # -------------------------
        # Parse intent
        # -------------------------
        intent = self.parser.parse(text)

        # -------------------------
        # Evaluate feasibility
        # -------------------------
        feas = self.feasibility_model.evaluate(intent, vision_context)

        # -------------------------
        # Extract subject
        # -------------------------
        selected = vision_context.get("selected")
        subject_id = selected["id"] if selected else None

        # -------------------------
        # Determine base distance
        # -------------------------
        base_dist = self.DEFAULT_DISTANCES.get(intent.shot_type, 3.0)

        if intent.distance_hint is not None:
            base_dist = intent.distance_hint

        # -------------------------
        # Height offset
        # -------------------------
        base_height = intent.height_hint if intent.height_hint is not None else 0.0

        # -------------------------
        # Speed / style
        # -------------------------
        speed = 1.0 * intent.speed_factor

        # Determine camera style keywords
        if "smooth" in intent.camera_keywords or intent.shot_type in ("orbit", "follow"):
            cam_style = "smooth_cinematic"
        elif "slowmo" in intent.camera_keywords:
            cam_style = "slow_motion"
        else:
            cam_style = "generic"

        # -------------------------
        # Build the shot plan
        # -------------------------
        plan = {
            "shot_type": intent.shot_type,
            "subject_id": subject_id,
            "distance": float(base_dist),
            "speed": float(speed),
            "height_offset": float(base_height),
            "camera_style": cam_style,
            "intent": intent.to_dict(),
            "feasibility": {
                "score": feas.score,
                "reasons": feas.reasons,
                "ok": feas.is_feasible
            }
        }

        return plan