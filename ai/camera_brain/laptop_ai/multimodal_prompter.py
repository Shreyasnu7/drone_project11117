# laptop_ai/multimodal_prompter.py
import json
import base64
import aiohttp
import time
from laptop_ai.config import OPENAI_API_KEY, OPENAI_MODEL

# NOTE: using OpenAI official python package is ideal, but here we do aiohttp to keep async control.
# If you prefer openai library, we can switch to that pattern.

OPENAI_API = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = """
You are a Drone Cinematic Planner AI.
Input: user_text, vision_context, images (URLs), video_link, memory.
Return: A strict JSON object with one of:
- {"action":"FOLLOW","params":{"distance_m":5,"height_m":3,"style":"CINEMATIC"}}
- {"action":"ORBIT","params":{"radius_m":8,"height_m":4,"speed_ms":1.5}}
- {"action":"HOVER","params":{}}
- {"action":"MOVE_TO","params":{"x":1.3,"y":0.0,"z":2.5,"yaw":0.0}}
- {"action":"RECORD","params":{"duration_s":8,"style":"SPORTS"}}
Always keep numbers reasonable. Do NOT include raw motor commands.
"""

async def ask_gpt(user_text, vision_context=None, images=None, video_link=None, memory=None, timeout=15):
    """
    Sends a multimodal prompt (text + vision_context + optional image URLs / video link).
    Returns parsed JSON (or None).
    """
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user", "content": f"User request: {user_text}\n\nVisionContext: {json.dumps(vision_context or {}, default=str)}\nMemory:{json.dumps(memory or {})}\nImages:{images or []}\nVideo:{video_link or ''}"}
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
            async with sess.post(OPENAI_API, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print("OpenAI error:", resp.status, text[:400])
                    return None
                data = await resp.json()
                # The assistant message should contain the JSON primitive. Try to parse.
                msg = data["choices"][0]["message"]["content"]
                # Try to find JSON within message
                # If assistant returns inline text + json, extract first {...}
                start = msg.find("{")
                end = msg.rfind("}")
                if start >= 0 and end >= 0:
                    jtxt = msg[start:end+1]
                    try:
                        return json.loads(jtxt)
                    except Exception:
                        print("Failed to parse assistant JSON. raw:", msg[:400])
                        return None
                else:
                    print("No JSON in assistant response:", msg[:200])
                    return None
    except Exception as e:
        print("ask_gpt exception:", e)
        return None
