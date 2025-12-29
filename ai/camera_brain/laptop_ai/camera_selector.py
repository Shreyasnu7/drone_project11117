# File: laptop_ai/camera_selector.py
"""
camera_selector
===============

Chooses the correct camera (GoPro or Internal AI camera)
depending on:
    • user request text
    • scene classification
    • motion level
    • available stabilization
    • need for HDR / low light / high fps
"""

def choose_camera_for_request(user_text: str, primitive, vision_context):
    text = user_text.lower()

    # If user SPECIFICALLY requests a camera -> override
    if "use gopro" in text or "gopro only" in text:
        return "gopro"
    if "use internal" in text or "use drone camera" in text:
        return "internal"

    # AI-driven decision:
    scene = primitive.get("camera_plan", {}).get("scene", {})

    # High motion → GoPro preferred
    if scene.get("action", 0) > 0.55:
        return "gopro"

    # Low light → Internal (if IMX678 or PiCam v3)
    if scene.get("night", 0) > 0.6:
        return "internal"

    # Cinematic request → GoPro
    if "cinematic" in text:
        return "gopro"

    # Default:
    return "gopro"