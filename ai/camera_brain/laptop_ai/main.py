import time, uuid
from fastapi import FastAPI
from pydantic import BaseModel
from contracts.shot_intent import ShotIntent

app = FastAPI(title="Laptop Camera AI")


class TextCommand(BaseModel):
    text: str
    camera: str = "external"


@app.post("/analyze/text")
def analyze_text(cmd: TextCommand):
    """
    TEMPORARY implementation:
    Later this will call real LLM + vision models.
    """

    intent = ShotIntent(
        intent_id=str(uuid.uuid4()),
        source="laptop_ai",
        timestamp=time.time(),

        shot_type="tracking",
        emotion="awe",
        camera=cmd.camera,
        priority="subject",

        pan=0.0,
        tilt=-0.1,
        forward_motion=0.3,
        altitude_change=0.05,
        yaw_rate=0.2,

        framing={
            "rule_of_thirds": True,
            "lead_room": True
        },

        zoom=1.2,
        exposure_bias=0.1,

        max_speed=5.0,
        allow_aggressive_motion=False,

        debug={
            "input_text": cmd.text
        }
    )

    return intent