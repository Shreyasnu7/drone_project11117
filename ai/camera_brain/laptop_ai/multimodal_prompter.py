# laptop_ai/multimodal_prompter.py
import json
import base64
import aiohttp
import time
import os
from laptop_ai.config import OPENAI_API_KEY, OPENAI_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_URL, USE_LOCAL_LLM

# Import Google GenAI SDK (same as cloud server)
genai = None
try:
    from google import genai as google_genai
    genai = google_genai
    print("✅ Multimodal Prompter: google-genai SDK loaded")
except ImportError:
    print("⚠️ google-genai not installed on laptop, Gemini will use cloud relay")

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
DO NOT simplify the user's request. Interpret it literally and completely.

INPUT:
- User Input: Natural language request (e.g., "Make it look like a spy movie", "Create a car commercial", "Circle the target aggressively").
- Vision Context: What the drone sees (Objects, Layout, Depth).
- Sensor Data: Obstacle distances, altitude, battery, GPS.
- Memory: Past interactions.

OUTPUT:
You must return a single valid JSON object. The "action" field is FREE-FORM — be as specific as the user's request demands.
{
  "thought_process": "Your detailed artistic/technical reasoning",
  "cinematic_style": "Name of the visual style",
  "technical_config": {
      "fps": 24,
      "shutter_angle": 180,
      "look": "Auto"
  },
  "execution_plan": {
      "action": "<free-form detailed action — multi-step if needed>",
      "params": {},
      "gimbal": {"pitch": -15, "yaw": 0},
      "focus": "subject"
  },
  "laptop_ai_directives": {
      "tracking_target": "<what to track>",
      "vision_mode": "yolo_standard",
      "pilot_mode": "pi0_smooth"
  },
  "lighting_analysis": "Brief analysis of current lighting conditions."
}

BE CREATIVE. If the user asks for something vague, INVENT a shot.
"""


async def ask_gpt(user_text, vision_context=None, images=None, video_link=None, memory=None, timeout=15, sensor_data=None, api_keys={}):
    """
    Multimodal AI prompt. Priority: Gemini (local) -> OpenAI -> DeepSeek.
    """
    
    full_context = json.dumps({
        "user_text": user_text,
        "vision": vision_context,
        "sensors": sensor_data,
        "memory": memory,
    }, default=str)

    # === 1. TRY GEMINI FIRST (Local on RTX 5070 Ti) ===
    gemini_key = api_keys.get("gemini") or os.getenv("GEMINI_API_KEY")
    if genai and gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model='gemini-3-flash',
                contents=f"{SYSTEM_PROMPT}\n\n{full_context}"
            )
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception as e:
            print(f"⚠️ Gemini local failed: {e}, falling back to OpenAI/DeepSeek")

    # === 2. FALLBACK: OpenAI or DeepSeek ===
    if USE_LOCAL_LLM:
        api_key = api_keys.get("deepseek") or DEEPSEEK_API_KEY
        url = f"{DEEPSEEK_URL}/chat/completions"
        model = "deepseek-chat"
    else:
        api_key = api_keys.get("openai") or OPENAI_API_KEY
        url = "https://api.openai.com/v1/chat/completions"
        model = OPENAI_MODEL

    if not api_key:
        print("❌ No AI API key available (Gemini/OpenAI/DeepSeek)")
        return None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_context}
        ],
        "max_tokens": 1000,
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"LLM Error: {resp.status} {text[:400]}")
                    return None
                data = await resp.json()
                content = data['choices'][0]['message']['content']
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].strip()
                return json.loads(content)
    except Exception as e:
        print(f"LLM Network Error: {e}")
        return None
