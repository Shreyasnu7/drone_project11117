from .intent_builder import IntentBuilder
from .cinematic_reasoner import CinematicReasoner
from .plan_generator import PlanGenerator
from ai.shot_intent.reasoning.intent_reasoner import ShotIntentReasoner
# Assuming you have a class for multi-modal acting as the entry reasoning
# If not, we will use ShotIntentReasoner as the primary entry

class CloudOrchestrator:
    """
    The Master Conductor.
    Pipeline:
    1. Multi-modal Ingest (Text/Image/Video) -> ShotIntentReasoner
    2. Cinematic Refinement -> CinematicReasoner
    3. Plan Generation -> PlanGenerator
    """
    def __init__(self, llm_client=None):
        # We need to instantiate the sub-modules
        # Note: In a real app, inject dependencies or config
        self.intent_reasoner = ShotIntentReasoner(llm_client, "prompts/shot_intent.txt") if llm_client else None
        self.cinematic_reasoner = CinematicReasoner()
        self.planner = PlanGenerator()

    async def process_request(self, payload: dict) -> dict:
        """
        Full Pipeline Execution
        """
        user_text = payload.get("text", "")
        references = payload.get("media", []) 
        
        # 0. Extract Dynamic Keys
        api_keys = payload.get("api_keys", {})

        # 1. RAW REASONING (The "Director" Brain)
        # This converts "chase car" into { "emotion": "fast", "subject": "car" ... }
        if self.intent_reasoner:
            raw_intent = self.intent_reasoner.reason(
                user_text=user_text,
                image_refs=[r for r in references if r['type'] == 'image'],
                video_refs=[r for r in references if r['type'] == 'video'],
                api_keys=api_keys
            )
        else:
            # Fallback if no LLM client passed (or mock)
            raw_intent = {"emotional_model": {"vector": {"neutral": 1.0}, "peak_allowed": True}, "camera_plan": {"shot_energy": 0.5}}

        # 2. CINEMATIC REFINEMENT (The "Cinematographer" Brain)
        # Adjusts energy, pacing, ensures we don't peak too early
        refined_intent = self.cinematic_reasoner.refine(raw_intent)

        # 3. PLAN GENERATION (The "Assistant Director" Brain)
        # Converts abstract intent into specific Drone Waypoints/Commands
        final_plan = self.planner.generate(refined_intent)

        return final_plan
