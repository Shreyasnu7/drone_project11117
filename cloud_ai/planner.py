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
        Output is descriptive, not prescriptive.
        """

        return {
            "plan_id": str(uuid.uuid4()),
            "intent_signature": shot_intent,
            "stages": [
                {
                    "stage": "approach",
                    "constraints": {
                        "motion_energy": shot_intent["motion_energy"],
                        "risk_tolerance": shot_intent["risk_tolerance"],
                    },
                },
                {
                    "stage": "primary_capture",
                    "constraints": {
                        "camera_presence": shot_intent["camera_presence"],
                        "imperfection_tolerance": shot_intent["imperfection_tolerance"],
                    },
                },
                {
                    "stage": "release",
                    "constraints": {
                        "pacing": shot_intent["pacing"],
                    },
                },
            ],
        }