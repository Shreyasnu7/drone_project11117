from fastapi import APIRouter, HTTPException
from cloud_ai.orchestrator import CloudOrchestrator
from cloud_ai.llm import RealLLMClient
from plan_router import submit_plan
from api_schemas import DronePlan
import logging

router = APIRouter(prefix="/director", tags=["AI-Command"])

# Initialize Orchestrator with REAL AI Client
# This ensures no fake "Mock" responses are ever generated.
orchestrator = CloudOrchestrator(llm_client=RealLLMClient())

@router.post("/ai/command")
async def ai_command(payload: dict):
    """
    Structured Pipeline: 
    Input -> Orchestrator -> Plan -> Queue
    """
    try:
        # 0. SYSTEM COMMAND BYPASS (For Config/Direct Control)
        if payload.get("provider") == "system":
            print(f"⚙️ SYSTEM COMMAND: {payload.get('text')}")
            # Direct Plan Creation (Skip LLM)
            # We wrap the text (e.g., "SET_CONFIG: ...") into a Plan Action
            # The Brain's director_core.py knows how to parse this string.
            plan = DronePlan(
                thought_process="System Override",
                reasoning="Direct User Configuration",
                action=payload.get("text"), # "SET_CONFIG: source=external"
                emotional_context={"override": 1.0},
                confidence=1.0
            )
            await submit_plan(plan)
            return {"status": "queued", "plan": plan.dict(), "type": "system"}

        # 1. Delegate to Orchestrator (Real AI)
        plan_result = await orchestrator.process_request(payload)
        
        # 2. Ensure it matches DronePlan schema
        if isinstance(plan_result, dict):
            plan = DronePlan(**plan_result)
        else:
            plan = plan_result

        # 3. Push to Execution Queue (Legacy/Laptop Polling)
        await submit_plan(plan)

        # 4. DIRECT DISPATCH (Cloud Mode)
        # Immediately send to connected drones via WebSocket
        from ws_router import connected_clients, WebSocket
        import json
        
        # Convert Plan -> MAVLink/Radxa Command
        # Plan Action: "GIMBAL_TRACK target=car" -> {"type": "gimbal", "payload": ...}
        # Plan Action: "TAKEOFF" -> {"type": "ai", "payload": {"action": "TAKEOFF"}}
        
        cmd_payload = {}
        cmd_type = "ai" # default
        
        if "GIMBAL" in plan.action.upper():
            cmd_type = "gimbal"
            # Extract Pitch/Yaw (Simplified - In real system, reasoning would give numbers)
            # Fore now, let's assume raw_intent usually has numerical data, but here we just pass intent
            # Actually, let's just send the whole plan and let Radxa decide? 
            # Radxa only understands specific JSON.
            
            # Parsing "GIMBAL_LOOK: 45, 0"
            try:
                parts = plan.action.split(":")
                if len(parts) > 1:
                    coords = parts[1].split(",")
                    cmd_payload = {"pitch": int(coords[0]), "yaw": int(coords[1])}
            except:
                cmd_payload = {"pitch": 0, "yaw": 0}

        elif "TAKEOFF" in plan.action.upper():
             cmd_type = "ai"
             cmd_payload = {"action": "TAKEOFF"}
        elif "LAND" in plan.action.upper():
             cmd_type = "ai"
             cmd_payload = {"action": "LAND"}
        
        # Broadcast to Drones
        if cmd_type and connected_clients:
            print(f"📡 DISPATCHING TO DRONES: {cmd_type} {cmd_payload}")
            msg = json.dumps({"type": cmd_type, "payload": cmd_payload})
            for cid, sock in connected_clients.items():
                if "drone" in cid.lower() or "brain" in cid.lower() or "radxa" in cid.lower():
                     try:
                         await sock.send_text(msg)
                     except:
                         pass

        return {
            "status": "queued_and_dispatched",
            "plan": plan.dict()
        }
    except Exception as e:
        print(f"❌ ORCHESTRATOR ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
