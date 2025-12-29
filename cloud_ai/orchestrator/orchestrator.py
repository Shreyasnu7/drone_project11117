from .session_state import SessionState
from .authority import AuthorityManager
from .planner import ExecutionPlanner
from .dispatcher import OrchestrationDispatcher


class CloudOrchestrator:
    """
    Central AI conductor.
    """

    def __init__(self, drone_id: str):
        self.state = SessionState(drone_id=drone_id)
        self.authority = AuthorityManager()
        self.planner = ExecutionPlanner()
        self.dispatcher = OrchestrationDispatcher()

    def handle_intent(
        self,
        shot_intent: dict,
        human_override: bool,
    ):
        owner = self.authority.resolve(
            human_override=human_override,
            ai_request=True,
        )

        if owner != "ai":
            return {"status": "blocked_by_human"}

        plan = self.planner.create_plan(shot_intent)

        self.state.ai_control_active = True
        self.state.active_intent_id = plan["plan_id"]
        self.state.execution_phase = "executing"

        self.dispatcher.dispatch_to_laptop(plan)

        return {
            "status": "executing",
            "plan_id": plan["plan_id"],
        }

    def handle_human_override(self):
        self.state.ai_control_active = False
        self.state.execution_phase = "paused"