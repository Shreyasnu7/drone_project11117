# plan_router.py
from fastapi import APIRouter
from api_schemas import DronePlan
import logging

# Single Router Definition
router = APIRouter(tags=["Plans"])

# Memory Queue for simple Plan Handover
_PLAN_QUEUE = []

@router.post("/submit")
async def submit_plan(plan: DronePlan):
    """
    Submits a plan to the queue. 
    Used by: ai_command_router.py
    """
    logging.info(f"📝 PLAN RECEIVED: {plan.action}")
    _PLAN_QUEUE.append(plan)
    return {"status": "queued", "queue_length": len(_PLAN_QUEUE)}

@router.get("/next")
async def get_next_plan():
    """
    Fetches and consumes the next plan.
    Used by: DirectorCore (via polling)
    """
    if not _PLAN_QUEUE:
        return {"plan": None}
    
    # FIFO Pop
    next_plan = _PLAN_QUEUE.pop(0)
    logging.info(f"📤 DISPATCHING PLAN: {next_plan.action}")
    return {"plan": next_plan.dict()}
