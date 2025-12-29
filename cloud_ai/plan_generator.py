# server/cloud_ai/plan_generator.py

from api_schemas import DronePlan


class PlanGenerator:
    """
    Converts AI reasoning output into a DronePlan.
    """

    def generate(self, ai_result: dict) -> DronePlan:
        return DronePlan(
            action=ai_result.get("action", "hover"),
            style=ai_result.get("style", "neutral"),
            target=ai_result.get("target"),
            constraints=ai_result.get("constraints", {}),
        )