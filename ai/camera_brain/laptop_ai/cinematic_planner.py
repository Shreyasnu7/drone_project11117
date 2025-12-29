# laptop_ai/cinematic_planner.py
import copy
import math

# Safety envelopes (adjust to your drone)
MAX_FOLLOW_DISTANCE = 30.0  # meters
MIN_FOLLOW_DISTANCE = 1.5
MAX_HEIGHT = 120.0  # legal limit in many places; tune down for safety
MIN_HEIGHT = 0.5
MAX_SPEED = 8.0  # m/s

def clamp(val, a, b):
    return max(a, min(b, val))

def to_safe_primitive(raw):
    """
    Input: raw dict from multimodal prompter or user.
    Output: validated primitive dict or default HOVER.
    """
    if not raw or "action" not in raw:
        return {"action":"HOVER","params":{}}

    p = copy.deepcopy(raw)
    action = p.get("action").upper()

    if action == "FOLLOW":
        params = p.get("params", {})
        d = clamp(float(params.get("distance_m", 5.0)), MIN_FOLLOW_DISTANCE, MAX_FOLLOW_DISTANCE)
        h = clamp(float(params.get("height_m", 3.0)), MIN_HEIGHT, MAX_HEIGHT)
        speed = clamp(float(params.get("speed_ms", 2.0)), 0.1, MAX_SPEED)
        style = params.get("style", "CINEMATIC")
        return {"action":"FOLLOW","params":{"distance_m":d,"height_m":h,"speed_ms":speed,"style":style}}

    if action == "ORBIT":
        params = p.get("params", {})
        radius = clamp(float(params.get("radius_m", 8.0)), 2.0, 50.0)
        height = clamp(float(params.get("height_m", 3.5)), MIN_HEIGHT, MAX_HEIGHT)
        speed = clamp(float(params.get("speed_ms", 1.0)), 0.1, MAX_SPEED)
        return {"action":"ORBIT","params":{"radius_m":radius,"height_m":height,"speed_ms":speed}}

    if action == "MOVE_TO":
        params = p.get("params", {})
        x = float(params.get("x", 0.0))
        y = float(params.get("y", 0.0))
        z = clamp(float(params.get("z", 2.5)), MIN_HEIGHT, MAX_HEIGHT)
        yaw = float(params.get("yaw", 0.0))
        return {"action":"MOVE_TO","params":{"x":x,"y":y,"z":z,"yaw":yaw}}

    if action == "RECORD":
        params = p.get("params", {})
        dur = clamp(float(params.get("duration_s", 8.0)), 1.0, 600.0)
        style = params.get("style", "CINEMATIC")
        return {"action":"RECORD","params":{"duration_s":dur, "style": style}}

    if action == "HOVER":
        return {"action":"HOVER","params":{}}

    # Fallback
    return {"action":"HOVER","params":{}}
