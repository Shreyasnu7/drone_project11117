# /ai/shot_intent/reasoning/intent_reasoner.py

from typing import Dict, Any
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
        image_refs: list[str] | None = None,
        video_refs: list[str] | None = None,
        memory_context: Dict[str, Any] | None = None,
        api_keys: Dict[str, str] = {}, # Added
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
            api_keys=api_keys # Pass Through
        )

        return json.loads(response)