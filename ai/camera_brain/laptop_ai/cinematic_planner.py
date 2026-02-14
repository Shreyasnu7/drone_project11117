# laptop_ai/cinematic_planner.py
"""
INTELLIGENT SAFETY ENVELOPE - Preserves AI Autonomy
Applies safety limits WITHOUT restricting creative actions.
"""
import copy

# Physical Safety Limits (based on hardware capabilities)
MAX_HEIGHT = 120.0  # Legal limit in many regions
MIN_HEIGHT = 0.5    # Safe minimum
MAX_SPEED = 15.0    # m/s (increased from 8.0 for dynamic shots)
MAX_DISTANCE = 100.0  # Maximum range from home
MAX_DURATION = 600.0  # 10 minutes max per shot

def clamp(val, a, b):
    """Clamp value between min and max"""
    return max(a, min(b, val))


def to_safe_primitive(raw_plan: dict):
    """
    Intelligent safety envelope that PRESERVES AI AUTONOMY.
    
    PHILOSOPHY:
    - The AI cloud brain is intelligent and creative
    - This function ONLY enforces physical safety limits
    - It does NOT restrict what actions the AI can choose
    - It does NOT force the AI into predefined patterns
    
    INPUT: Raw plan from cloud AI (Gemini/GPT)
    OUTPUT: Same plan with safety-clamped numeric parameters
    """
    if not raw_plan or not isinstance(raw_plan, dict):
        return {"action": "HOVER", "params": {}}
    
    # Make a deep copy to avoid mutating the original
    safe_plan = copy.deepcopy(raw_plan)
    
    # Extract components
    action = safe_plan.get("action", "HOVER")
    if isinstance(action, str):
        action = action.upper()
    params = safe_plan.get("params", {})
    
    # === INTELLIGENT PARAMETER SAFETY ===
    # Apply safety bounds to ALL numeric parameters automatically
    safe_params = {}
    for key, value in params.items():
        try:
            # Detect parameter type by name and apply appropriate limits
            if "height" in key.lower() or "altitude" in key.lower() or ("z" == key.lower() and isinstance(value, (int, float))):
                # Altitude parameters
                safe_params[key] = clamp(float(value), MIN_HEIGHT, MAX_HEIGHT)
            
            elif "speed" in key.lower() or "velocity" in key.lower():
                # Speed parameters
                safe_params[key] = clamp(float(value), 0.1, MAX_SPEED)
            
            elif "distance" in key.lower() or "radius" in key.lower() or "range" in key.lower():
                # Distance/radius parameters
                safe_params[key] = clamp(float(value), 0.0, MAX_DISTANCE)
            
            elif "duration" in key.lower() or "time" in key.lower():
                # Duration parameters
                safe_params[key] = clamp(float(value), 0.1, MAX_DURATION)
            
            elif "pitch" in key.lower() and "gimbal" not in key.lower():
                # Drone pitch (degrees)
                safe_params[key] = clamp(float(value), -45.0, 45.0)
            
            elif "roll" in key.lower():
                # Drone roll (degrees)
                safe_params[key] = clamp(float(value), -45.0, 45.0)
            
            elif "yaw" in key.lower() or "heading" in key.lower():
                # Yaw (degrees, wrap around)
                safe_params[key] = float(value) % 360.0
            
            elif key == "waypoints" or key == "path" or key == "trajectory":
                # Waypoint lists - validate each point
                if isinstance(value, list):
                    safe_waypoints = []
                    for wp in value[:100]:  # Max 100 waypoints (safety limit)
                        if isinstance(wp, (list, tuple)) and len(wp) >= 3:
                            x, y, z = float(wp[0]), float(wp[1]), clamp(float(wp[2]), MIN_HEIGHT, MAX_HEIGHT)
                            # Clamp horizontal distance from origin
                            distance_2d = (x**2 + y**2)**0.5
                            if distance_2d > MAX_DISTANCE:
                                scale = MAX_DISTANCE / distance_2d
                                x, y = x * scale, y * scale
                            safe_waypoints.append([x, y, z])
                    safe_params[key] = safe_waypoints if safe_waypoints else params.get(key, [])
                else:
                    safe_params[key] = value

            # --- NEW CINEMATIC VALIDATIONS ---
            elif action == "FLY_TRAJECTORY" and "curvature" in key.lower():
                 # Validate trajectory curvature (0.0 to 1.0)
                 safe_params[key] = clamp(float(value), -1.0, 1.0)
            
            elif action == "CRANE_SHOT" and "vertical_range" in key.lower():
                 # Restrict crane shot height difference
                 safe_params[key] = clamp(float(value), 1.0, 30.0)

            elif action == "DOLLY_ZOOM" and "end_distance" in key.lower():
                 # Restrict dolly zoom distance
                 safe_params[key] = clamp(float(value), 2.0, MAX_DISTANCE)

            elif action == "FLY_THROUGH" and "clearance" in key.lower():
                 # Ensure safety clearance is respected (fake perception stub)
                 safe_params[key] = clamp(float(value), 0.5, 5.0)
            # -------------------------------
            
            elif key == "multi_shot_sequence":
                # Multi-shot sequences - recursively validate each shot
                if isinstance(value, list):
                    safe_params[key] = [to_safe_primitive(shot) for shot in value]
                else:
                    safe_params[key] = value
            
            else:
                # Unknown numeric parameter - pass through if reasonable
                if isinstance(value, (int, float)):
                    # Generic safety: clamp to reasonable range
                    safe_params[key] = clamp(float(value), -1000.0, 1000.0)
                else:
                    # Non-numeric - pass through unchanged
                    safe_params[key] = value
        
        except (ValueError, TypeError):
            # If conversion fails, keep original value
            safe_params[key] = value
    
    # === PRESERVE ALL AI COMPONENTS ===
    result = {
        "action": action,  # NO FILTERING - AI chooses freely
        "params": safe_params
    }
    
    # Pass through additional AI planning components
    if "thought_process" in safe_plan:
        result["thought_process"] = safe_plan["thought_process"]
    
    if "gimbal" in safe_plan:
        # Validate gimbal parameters
        gimbal = safe_plan["gimbal"]
        safe_gimbal = {}
        if "pitch" in gimbal:
            safe_gimbal["pitch"] = clamp(float(gimbal["pitch"]), -90.0, 20.0)  # Gimbal range
        if "yaw" in gimbal:
            safe_gimbal["yaw"] = float(gimbal["yaw"]) % 360.0
        if "roll" in gimbal:
            safe_gimbal["roll"] = clamp(float(gimbal["roll"]), -45.0, 45.0)
        result["gimbal"] = safe_gimbal
    
    if "camera_config" in safe_plan or "technical_config" in safe_plan:
        result["camera_config"] = safe_plan.get("camera_config") or safe_plan.get("technical_config", {})
    
    if "cinematic_style" in safe_plan:
        result["cinematic_style"] = safe_plan["cinematic_style"]
    
    if "lighting_analysis" in safe_plan:
        result["lighting_analysis"] = safe_plan["lighting_analysis"]
    
    if "multi_shot_sequence" in safe_plan and "multi_shot_sequence" not in result:
        # Already processed above in params section
        pass
    
    if "focus" in safe_plan:
        result["focus"] = safe_plan["focus"]
    
    if "framing_style" in safe_plan:
        result["framing_style"] = safe_plan["framing_style"]
    
    if "target_subject" in safe_plan:
        result["target_subject"] = safe_plan["target_subject"]
    
    # === EMERGENCY SAFETY: Battery Check ===
    # If battery-critical actions are sent, validate against current battery
    # (This would require battery telemetry access, placeholder for now)
    
    return result


def validate_action_safety(action: str, current_state: dict) -> tuple[bool, str]:
    """
    Final safety check before execution.
    
    Args:
        action: The action to execute
        current_state: Current drone state (battery, altitude, position, etc.)
    
    Returns:
        (is_safe, reason)
    """
    # Battery check
    battery = current_state.get("battery", 100)
    if battery < 20:
        return False, "Battery too low for autonomous actions"
    
    # Altitude check
    current_altitude = current_state.get("altitude", 0)
    if current_altitude > MAX_HEIGHT:
        return False, f"Current altitude {current_altitude}m exceeds maximum {MAX_HEIGHT}m"
    
    # GPS check (if required)
    gps_sats = current_state.get("gps_satellites", 0)
    if action.upper() in ["ORBIT", "FOLLOW", "TRACK_PATH"] and gps_sats < 6:
        return False, f"Insufficient GPS satellites ({gps_sats}) for position-based action"
    
    return True, "Safe to execute"
