# ai/camera_brain/outputs/agent_sender.py

import json
import socket
from ai.camera_brain.outputs.camera_command import CameraCommand


class RadxaAgentSender:
    """
    Sends CameraCommand to radxa_agent.
    """

    def __init__(self, agent_ip: str, agent_port: int):
        self.addr = (agent_ip, agent_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, cmd: CameraCommand):
        payload = {
            "type": "camera_command",
            "version": 1,
            "data": cmd.__dict__,
        }
        self.sock.sendto(json.dumps(payload).encode(), self.addr)