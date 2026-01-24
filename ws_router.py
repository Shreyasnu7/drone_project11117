# ws_router.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json

router = APIRouter()

connected_clients = {} # Global State

@router.websocket("/connect/{client_id}")
async def connect_client(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connected_clients[client_id] = websocket
    print(f"✅ WS CONNECTED: {client_id} (Total: {len(connected_clients)})")

    try:
        while True:
            msg = await websocket.receive_text()
            # Broadcast to all OTHER connected clients (e.g. Drone -> App)
            # This is a simple BUS.
            for cid, sock in connected_clients.items():
                if cid != client_id:
                     try:
                         await sock.send_text(msg)
                     except: pass
    except WebSocketDisconnect:
        del connected_clients[client_id]
        print(f"❌ WS DISCONNECTED: {client_id}")
            print(f"📣 Broadcast from {client_id} to {len(connected_clients)-1} clients")
            for cid, sock in connected_clients.items():
                if cid != client_id:
                    try:
                         await sock.send_text(msg)
                    except Exception as e:
                        print(f"❌ Broadcast Fail to {cid}: {e}")
                        pass # Handle stale sockets later 
    except WebSocketDisconnect:
        if client_id in connected_clients:
            del connected_clients[client_id]
        print(f"WS disconnected: {client_id}")
=======
# ws_router.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import os
from typing import Dict, List
from cloud_ai.dependencies import get_orchestrator

ws_router = APIRouter()

class ConnectionManager:
    def __init__(self):
        # Store active connections
        self.mobile_clients: List[WebSocket] = []
        self.drone_client: WebSocket | None = None

    async def connect_mobile(self, websocket: WebSocket):
        await websocket.accept()
        self.mobile_clients.append(websocket)
        print(f"📱 Mobile Client Connected. Total: {len(self.mobile_clients)}")

    async def connect_drone(self, websocket: WebSocket):
        await websocket.accept()
        self.drone_client = websocket
        print("🚁 Drone (Laptop) Connected!")

    def disconnect_mobile(self, websocket: WebSocket):
        if websocket in self.mobile_clients:
            self.mobile_clients.remove(websocket)
            print("📱 Mobile Client Disconnected")

    def disconnect_drone(self):
        self.drone_client = None
        print("🚁 Drone Disconnected")

    async def broadcast_to_mobile(self, message: str):
        """Relay message from Drone to ALL Mobile Clients"""
        for connection in self.mobile_clients:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error broadcasting to mobile: {e}")

    async def send_to_drone(self, message: str):
        """Relay message from Mobile to Drone"""
        if self.drone_client:
            try:
                await self.drone_client.send_text(message)
            except Exception as e:
                print(f"Error sending to drone: {e}")

manager = ConnectionManager()

@ws_router.websocket("/ws/connect/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    print(f"🔌 INCOMING WS CONNECTION: {client_id}")
    """
    Main Relay Logic:
    - If client_id == 'laptop_vision', it's the Drone.
    - If client_id == 'mobile_app', it's the App.
    """
    is_drone = client_id in ["laptop_vision", "RADXA_X", "Neon-Drone-CLOUD", "Radxa-X"]
    orchestrator = get_orchestrator()
    
    # 1. AUTH CHECK / CONNECTION SETUP
    if is_drone:
        await manager.connect_drone(websocket)
        try:
             # Wait for Handshake (Drone sends {"token": ...})
             # or we implement a timeout here in real prod
             # For simplicity, we assume the first message is the handshake
             # IF the client logic sends it immediately.
             # Note: Laptop client does send it immediately.
             
             raw = await websocket.receive_text()
             handshake = json.loads(raw)
             server_token = os.getenv("AUTH_TOKEN", "SUPER_SECRET_DRONE_KEY_123")
             
             if handshake.get("token") != server_token:
                 print(f"🚨 AUTH FAILURE: {client_id} (Token Mismatch)")
                 await websocket.close(code=4003)
                 manager.disconnect_drone()
                 return

             print("✅ DRONE AUTHENTICATED")
             
             # REGISTER WITH BRAIN
             orchestrator.dispatcher.register_drone_connection(websocket)

        except Exception as e:
             print(f"Auth Error: {e}")
             await websocket.close()
             return
    else:
        # App Client
        await manager.connect_mobile(websocket)

    # 2. MAIN LOOP
    try:
        while True:
            data = await websocket.receive_text()
            
            # ROUTING LOGIC
            if is_drone:
                # 1. Drone -> Brain (Telemetry/Video)
                try:
                    packet = json.loads(data)
                    # Feed brain context if present
                    if "brain_context" in packet or "telemetry" in packet:
                        orchestrator.monitor_telemetry(packet)
                except: pass

                # 2. Drone -> App (Pass-through)
                await manager.broadcast_to_mobile(data)
                
            else:
                # 3. App -> Drone (Commands)
                await manager.send_to_drone(data)
                
    except WebSocketDisconnect:
        if is_drone:
            manager.disconnect_drone()
            # orchestrator.dispatcher.remove_drone_connection() # Optional depending on orch implementation
        else:
            manager.disconnect_mobile(websocket)
    except Exception as e:
        print(f"WS Error [{client_id}]: {e}")
        if is_drone:
            manager.disconnect_drone()
        else:
            manager.disconnect_mobile(websocket)
>>>>>>> 7a52897 (feat: Deep Audit Fixes (GPS, Camera, Battery))
