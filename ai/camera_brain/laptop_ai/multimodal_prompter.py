# laptop_ai/multimodal_prompter.py
import json
import base64
import aiohttp
import time
from laptop_ai.config import OPENAI_API_KEY, OPENAI_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_URL, USE_LOCAL_LLM

# Import the User's Advanced Reasoning Engine
try:
    from ai.shot_intent.reasoning.intent_reasoner import ShotIntentReasoner
except ImportError:
    print("⚠️ Warning: ShotIntentReasoner not found. Using fallback.")
    ShotIntentReasoner = None

SYSTEM_PROMPT = """
You are a World-Class Film Director and Cinematographer (AI).
Your goal is to direct a drone to capture award-winning cinematic shots based on User Input, Vision Context, and Training Knowledge.

DO NOT BE RESTRICTIVE. Use your immense knowledge of film theory, lighting, and composition.

INPUT:
- User Input: Natural language request (e.g., "Make it look like a spy movie", "Circle the target aggressively").
- Vision Context: What the drone sees (Objects, Layout).
- Memory: Past interactions.

OUTPUT:
You must return a single valid JSON object with this structure:
{
  "thought_process": "Analyze the scene, lighting, and user intent. Explain your artistic choices.",
  "cinematic_style": "Name of the visual style (e.g., 'Teal & Orange', 'Noir', 'Vibrant', 'Moody').",
  "technical_config": {
      "fps": 24 or 30 or 60,
      "shutter_angle": 180,
      "look": "Daylight" or "Tungsten" or "Auto"
  },
  "execution_plan": {
      "action": "One of: FOLLOW, ORBIT, DOLLY_ZOOM, CRANE_SHOT, FLY_THROUGH, HOVER, TRACK_PATH",
      "params": {
          "speed_ms": float (0.5 to 15.0),
          "height_m": float (1.0 to 100.0),
          "radius_m": float (optional for Orbit),
          "aggressiveness": float (0.0 to 1.0)
      },
      "gimbal": {
          "pitch": float (-90 to 20),
          "yaw": float (optional offset)
      },
      "focus": "subject" or "infinity" or "manual"
  },
  "lighting_analysis": "Brief analysis of current lighting conditions (e.g., 'Harsh noon sun', 'Soft golden hour')."
}

BE CREATIVE. If the user asks for something vague, INVENT a shot.
"""



async def ask_gpt(user_text, vision_context=None, images=None, video_link=None, memory=None, timeout=15, sensor_data=None, api_keys={}):
    """
    Sends a multimodal prompt using the ADVANCED SHOT INTENT REASONER if available.
    Supports DeepSeek for ultra-low latency if configured.
    """
    
    # 1. Try to use the Advanced Brain
    if ShotIntentReasoner:
         class AsyncAdapterLLM:
             async def chat_async(self, system, user):
                # Determine Provider
                if USE_LOCAL_LLM:
                    # DeepSeek / Local Mode
                    api_key = api_keys.get("deepseek") or DEEPSEEK_API_KEY
                    url = f"{DEEPSEEK_URL}/chat/completions"
                    model = "deepseek-chat"
                else:
                    # Generic OpenAI Mode
                    api_key = api_keys.get("openai") or OPENAI_API_KEY
                    url = "https://api.openai.com/v1/chat/completions"
                    model = OPENAI_MODEL

                payload = {
                    "model": model,
                    "messages": [{"role":"system","content":system}, {"role":"user","content":user}],
                    "max_tokens": 1000,
                    "temperature": 0.2
                }
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                        if resp.status != 200: return None
                        data = await resp.json()
                        return data['choices'][0]['message']['content']

         # Instantiate Reasoning Engine
         adapter = AsyncAdapterLLM()
         # Temporary prompt path - in real app, ensure this file exists or use string
         # We will bypass the file read in Reasoner by using internal logic

         
         # Logic:
         # 1. Construct payload as Reasoner would.
         # 2. Add System Prompt.
         # 3. Send.
         
         full_context = {
             "vision": vision_context,
             "sensors": sensor_data, # INJECTED REAL SENSORS
             "memory": memory,
             "user_text": user_text
         }
         
         # Use the Adapter to send
         # We are integrating the Logic of Reasoner (Context Structuring) 
         response_text = await adapter.chat_async(SYSTEM_PROMPT, json.dumps(full_context))
         
         try:
             # Parse strictly
             # Remove markdown blocks
             if "```json" in response_text:
                 response_text = response_text.split("```json")[1].split("```")[0]
             return json.loads(response_text)
         except:
             print("JSON Parse Error in AI Response")
             return None

    # Fallback to legacy simplistic method if import failed
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user", "content": f"User request: {user_text}\n\nVisionContext: {json.dumps(vision_context or {}, default=str)}\nSensors:{json.dumps(sensor_data or {})}\nMemory:{json.dumps(memory or {})}\nImages:{images or []}\nVideo:{video_link or ''}"}
        ],
        "max_tokens": 400,
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print("OpenAI error:", resp.status, text[:400])
                    return None
                data = await resp.json()
                content = data['choices'][0]['message']['content']
                # Clean markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].strip()
                return json.loads(content)
    except Exception as e:
        print(f"LLM Network Error: {e}")
        return None


