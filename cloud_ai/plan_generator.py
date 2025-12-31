# server/cloud_ai/plan_generator.py

from api_schemas import DronePlan


class PlanGenerator:
    """
    Converts AI reasoning output into a DronePlan.
    """

    def generate(self, ai_result: dict) -> DronePlan:
        # ROBUST MAPPING: Try 'action' first, then 'intent_type'
        raw_action = ai_result.get("action") or ai_result.get("intent_type") or "hover"
        
        return DronePlan(
            action=raw_action,
            thought_process=ai_result.get("thought_process", "Processing..."),
            reasoning=ai_result.get("reasoning", "AI decision"),
            style=ai_result.get("style", "neutral"),
            target=ai_result.get("target"),
            constraints=ai_result.get("constraints", {}),
            confidence=ai_result.get("confidence", 1.0)
        )