import asyncio
import json
import websockets

SERVER_URL = "wss://drone-server-r0qe.onrender.com/ws/TEST_RADXA"

async def main():
    async with websockets.connect(SERVER_URL) as ws:
        print("✅ Connected to cloud server")

        # Receive system message
        msg = await ws.recv()
        print("⬇ From server:", msg)

        # 1️⃣ Send ping
        await ws.send(json.dumps({"type": "ping"}))
        print("⬆ Sent ping")

        print("⬇", await ws.recv())

        # 2️⃣ Send manual control
        await ws.send(json.dumps({
            "type": "control",
            "payload": {
                "throttle": 0.6,
                "yaw": 0.1,
                "pitch": 0.0,
                "roll": 0.0
            }
        }))
        print("⬆ Sent control")

        print("⬇", await ws.recv())

        # 3️⃣ Send AI command
        await ws.send(json.dumps({
            "type": "ai",
            "payload": {
                "prompt": "smooth cinematic follow"
            }
        }))
        print("⬆ Sent AI command")

        print("⬇", await ws.recv())

        # 4️⃣ Send telemetry
        await ws.send(json.dumps({
            "type": "telemetry",
            "payload": {
                "battery": 87,
                "altitude": 12.4,
                "speed": 3.9
            }
        }))
        print("⬆ Sent telemetry")

        await asyncio.sleep(2)

asyncio.run(main())