from typing import Dict, Any
import uuid


class ExecutionPlanner:
    """
    Converts ShotIntent into a high-level execution plan.
    """

    def create_plan(
        self,
        shot_intent: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Input: "Director's" Intent (shot_type, motion_energy, target_subject)
        Output: "Pilot's" Flight Plan (DronePlan compliant)
        """
        maneuver = self._derive_maneuver(shot_intent)

        return {
            "plan_id": str(uuid.uuid4()),
            "intent_signature": shot_intent,
            "maneuver": maneuver,
            
            # SCHEMA COMPLIANCE (api_schemas.DronePlan)
            "action": maneuver["type"], 
            "target": shot_intent.get("target_subject"), 
            "target": shot_intent.get("target_subject"), 
            "reasoning": shot_intent.get("reasoning", "Command Queued (No AI Reasoning provided)"),
            "style": shot_intent.get("camera_movement", "Standard"),
            "style": shot_intent.get("camera_movement", "Standard"),
            
            "constraints": {
                 "speed_limit": self._energy_to_speed(shot_intent.get("motion_energy", 0.5)),
                 "framing": shot_intent.get("framing", "MEDIUM"),
                 "risk_tolerance": shot_intent.get("motion_energy", 0.5) 
            }
        }

    def _derive_maneuver(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates Abstract Intent -> Concrete Maneuver Primitive
        """
        # AI Output: shot_type (ORBIT, FOLLOW, DOLLY, HOVER)
        stype = intent.get("shot_type", "ERROR").upper()
        energy = intent.get("motion_energy", 0.5)
        
        if stype == "ORBIT":
            return {
                "type": "ORBIT",
                "radius": 2.0 if intent.get("framing") == "TIGHT" else 8.0, # UNLOCKED: 2m is very close
                "speed": energy * 10.0 # UNLOCKED: Up to 10m/s orbit speed
            }
        elif stype == "FOLLOW" or stype == "TRACK":
            return {
                "type": "FOLLOW",
                "offset": {"x": -2.0, "y": 0, "z": 1.0}, # UNLOCKED: Lower/Closer default
                "aggressiveness": energy * 1.5 # UNLOCKED: Overclock PIDs if needed (>1.0)
            }
        elif stype == "DOLLY" or stype == "REVEAL":
             return {
                 "type": "DOLLY",
                 "vector": {"x": 1.0, "y": 0.0, "z": 0.2}, 
                 "duration": 10.0
             }
        elif stype == "HOVER":
             return {
                "type": "HOVER",
                "look_at_subject": True
             }
        else:
             # UNKNOWN / ERROR
             return {
                 "type": "ERROR",
                 "reason": "Unknown shot type or AI Failure"
             }
             
    def _energy_to_speed(self, energy: float) -> float:
        # Map 0.0-1.0 to 0.5m/s - 20.0m/s (UNCONSTRAINED)
        return 0.5 + (energy * 19.5)