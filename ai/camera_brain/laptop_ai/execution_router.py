# File: laptop_ai/execution_router.py

class ExecutionRouter:
    """
    High-level router that decides what to do with a completed cinematic plan.
    It does NOT generate motor commands. It only forwards safe validated plans.
    """

    def __init__(self, messaging_client):
        self.ws = messaging_client

    async def send_plan(self, job_id, user_id, drone_id, primitive):
        packet = {
            "type": "ai_plan",
            "target": "server",
            "job_id": job_id,
            "user_id": user_id,
            "drone_id": drone_id,
            "primitive": primitive,
            "meta": {
                "source": "laptop",
                "safe": True
            }
        }

        await self.ws.send(packet)
        print(f"[ExecutionRouter] forwarded plan (safe) job={job_id}")
