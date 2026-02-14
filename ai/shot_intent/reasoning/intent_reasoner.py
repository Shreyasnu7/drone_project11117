# /ai/shot_intent/reasoning/intent_reasoner.py

from typing import Dict, Any, Optional, List
import json


class ShotIntentReasoner:
    """
    Free-form cinematic reasoning layer.
    This is where intelligence lives.
    """

    def __init__(self, llm_client, prompt_path: str):
        self.llm = llm_client
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def reason(
        self,
        user_text: str,
        image_refs: Optional[list[str]] = None,
        video_refs: Optional[list[str]] = None,
        memory_context: Optional[Dict[str, Any]] = None,
        provider: str = "gemini",
        api_keys: Dict[str, str] = {} # NEW
    ) -> Dict[str, Any]:
        """
        Perform unrestricted cinematic reasoning.
        """

        payload = {
            "user_text": user_text,
            "image_refs": image_refs or [],
            "video_refs": video_refs or [],
            "memory_context": memory_context or {},
        }

        response = self.llm.chat(
            system=self.system_prompt,
            user=json.dumps(payload, ensure_ascii=False),
            provider=provider,
            api_keys=api_keys
        )

        if not response: # Handle None from LLM Error
            # FALLBACK INTENT (Safety Net)
            return {
                "emotional_model": {"vector": {"neutral": 1.0}, "peak_allowed": True}, 
                "camera_plan": {
                    "shot_energy": 0.5, 
                    "framing": "wide",
                    "movement_style": "smooth-stabilized"
                },
                # KEYS REQUIRED BY PLANNER:
                "motion_energy": 0.1,  # Conservative fallback
                "risk_tolerance": 0.0, # Safe fallback
                "shot_type": "ERROR",  # Expose failure
                "action": "ERROR",     # Expose failure
                "reasoning": "Fallback: AI Offline (No Response)"
            }

        # Parse Response
        try:
             data = json.loads(response)
             # If "reasoning" starts with "System:", it's likely a mock error from llm.py
             # We should trust it even if it was a mock-up
             return data
        except:
             return {
                "action": "ERROR",
                "reasoning": "Fallback: JSON Parse Error from AI"
             }