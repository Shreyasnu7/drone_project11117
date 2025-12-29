# laptop_ai/messaging_client.py
import asyncio
import json
import websockets
from laptop_ai.config import VPS_WS, AUTH_TOKEN
import time

class MessagingClient:
    """
    Robust websocket client that auto-reconnects and does a handshake:
    sends {"id": "<client_id>", "token": AUTH_TOKEN} immediately after connect.
    """

    def __init__(self, client_id="laptop"):
        self.client_id = client_id
        self.ws = None
        self.connected = False
        self._recv_handlers = []
        self._send_lock = asyncio.Lock()
        self._closing = False

    async def connect(self):
        backoff = 1.0
        while not self._closing:
            try:
                print("CONNECTING TO VPS_WS =", VPS_WS)
                self.ws = await websockets.connect(VPS_WS, ping_interval=10, max_size=None)
                # handshake
                await self.ws.send(json.dumps({"id": self.client_id, "token": AUTH_TOKEN}))
                self.connected = True
                print("MessagingClient connected")
                asyncio.create_task(self._recv_loop())
                return
            except Exception as e:
                print("WS connect err:", e)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)

    async def _recv_loop(self):
        try:
            while True:
                raw = await self.ws.recv()
                try:
                    packet = json.loads(raw)
                except Exception:
                    packet = {"raw": raw}
                # call handlers
                for h in list(self._recv_handlers):
                    try:
                        await h(packet)
                    except Exception as e:
                        print("Handler error:", e)
        except Exception as e:
            print("WS recv failed:", e)
            self.connected = False
            # reconnect loop
            await asyncio.sleep(1.0)
            await self.connect()

    def add_recv_handler(self, coro):
        self._recv_handlers.append(coro)

    async def send(self, packet: dict):
        async with self._send_lock:
            if not self.connected:
                await self.connect()
            tries = 0
            while tries < 3:
                try:
                    await self.ws.send(json.dumps(packet))
                    return
                except Exception as e:
                    print("WS send failed:", e)
                    self.connected = False
                    tries += 1
                    await asyncio.sleep(0.5)
                    await self.connect()

    async def close(self):
        self._closing = True
        if self.ws:
            await self.ws.close()
