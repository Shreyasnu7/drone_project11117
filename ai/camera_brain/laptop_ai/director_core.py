# File: laptop_ai/director_core.py
"""
Main orchestrator for the Laptop AI Director.

Responsibilities:
- Listen for new AI jobs from the VPS (via MessagingClient).
- Capture context frame(s) from the drone RTSP / local camera.
- Run VisionTracker to produce vision_context (smoothed tracks).
- Send multimodal request to the cloud prompter (text + optional images/video link).
- Convert cloud response into a validated cinematic primitive via cinematic_planner.
- Use UltraDirector for curve planning (Bezier + obstacle warping) when the primitive needs a trajectory.
- Send validated plan back to the VPS (server) using MessagingClient for Radxa to pick up.
- Strict safety-first behavior, simulation-friendly.
- Extensive logging + retry + backoff.

Important safety notes (READ BEFORE USING ON REAL DRONE):
- This module does NOT actuate motors directly. It emits *high-level safe primitives*.
- Always test in SITL / simulation (PX4 SITL, Gazebo, or indoors with props off).
- Keep human-in-loop override ready (joystick / RC switch).
"""

import asyncio
import time
import os
import json
import traceback
import cv2
import numpy as np
from typing import Optional

# Local modules (ensure these files exist or produce them next)
# Local modules (ensure these files exist or produce them next)
from camera_fusion import CameraFusion
from gopro_driver import GoProDriver
from ai_camera_brain import AICameraBrain
from camera_selector import choose_camera_for_request
from messaging_client import MessagingClient
from vision_tracker import VisionTracker
from multimodal_prompter import ask_gpt
from cinematic_planner import to_safe_primitive
from ultra_director import UltraDirector
from memory_client import read_memory, write_memory
from config import RTSP_URL, AI_CALL_INTERVAL, FRAME_SKIP, TEMP_ARTIFACT_DIR, OPENAI_API_KEY

# Safety configuration (tweak to your hardware / rules)
MAX_FRAME_WAIT = 2.0            # seconds to wait to get a frame
JOB_PROCESS_TIMEOUT = 60.0      # max seconds to spend processing a job
DEBUG_SAVE_FRAME = True

os.makedirs(TEMP_ARTIFACT_DIR, exist_ok=True)

class Director:
    def __init__(self, client_id="laptop", simulate: bool = False):
        """
        simulate: if True, never send plans to real VPS target "drone"; instead log them.
        """
        self.ws = MessagingClient(client_id=client_id)
        self.tracker = VisionTracker()            # vision module (YOLO + tracker)
        self.ultra = UltraDirector()              # curve planner + mode chooser
        self.fusion = CameraFusion()
        self.gopro = GoProDriver()
        self.ai_cam = AICameraBrain()
        self.simulate = simulate
        self.processing = False
        self.last_job_time = 0
        self.frame_skip = FRAME_SKIP or 1
    
    def predict_collision(self, future_points, obstacles):
        for p in future_points:
            for ob in obstacles:
                if np.linalg.norm(p - ob.position) < ob.radius:
                    return True
        return False

    async def start(self):
        await self.ws.connect()
        self.ws.add_recv_handler(self._handle_packet)
        print("Director: connected to messaging service. Waiting for jobs...")

    async def _handle_packet(self, packet):
        """
        Messaging client will call this when any new packet arrives.
        We expect packets like:
          {"type":"ai_job", "job_id": "123", "user_id":"u", "drone_id":"d", "text":"Follow the white car", "images":[url], "video": url}
        """
        try:
            t = packet.get("type")
            if t == "ai_job":
                # dispatch job to background worker
                asyncio.create_task(self.process_job(packet))
            else:
                # other packet types (telemetry ack etc.)
                print(f"Director: received non-job packet type={t}")
        except Exception as e:
            print("Error in _handle_packet:", e)
            traceback.print_exc()

    async def process_job(self, job: dict):
        """
        Top-level job processing pipeline. This is where the high-level AI -> plan flow happens.
        """
        if self.processing:
            print("Director busy. requeuing job", job.get("job_id"))
            # Optionally send back a busy response so server requeues with delay
            await asyncio.sleep(1.0)
            return

        self.processing = True
        job_id = job.get("job_id", f"job_{int(time.time())}")
        user_id = job.get("user_id")
        drone_id = job.get("drone_id")
        user_text = job.get("text", "")
        images = job.get("images", []) or []
        video_link = job.get("video", None)
        # --- CAMERA BRAIN DECISION (only valid AFTER primitive exists) ---
        fusion_state = self.fusion.get_fusion_state()
        camera_plan = self.ai_cam.decide(user_text, fusion_state)

        # Embed camera plan into primitive metadata
        if "meta" not in primitive:
           primitive["meta"] = {}

        primitive["meta"]["camera_plan"] = camera_plan

        print(f"\n=== Starting job {job_id} user={user_id} drone={drone_id} text='{user_text[:80]}' ===")
        self.last_job_time = time.time()
        try:
            # 1) Grab a context frame from RTSP (non-blocking with timeout)
            frame = await asyncio.to_thread(self._grab_frame, RTSP_URL, MAX_FRAME_WAIT)
            if frame is None:
                print("No frame available — returning a conservative HOVER primitive.")
                primitive = {"action": "HOVER", "params": {}}
                await self._send_plan(job_id, user_id, drone_id, primitive, reason="no_frame")
                return

            # Save debug artifact
            if DEBUG_SAVE_FRAME:
                fname = os.path.join(TEMP_ARTIFACT_DIR, f"job_{job_id}_ctx.jpg")
                cv2.imwrite(fname, frame)

            # 2) Vision context (tracks, selected, velocities)
            vision_context, annotated = await asyncio.to_thread(self.tracker.process_frame, frame)

            # optional save annotated frame
            if DEBUG_SAVE_FRAME:
                fname = os.path.join(TEMP_ARTIFACT_DIR, f"job_{job_id}_annot.jpg")
                cv2.imwrite(fname, annotated)

            # 3) Attach user/drone memory
            memory = read_memory(user_id, drone_id) or {}

            # 4) Call multimodal prompter (cloud) — this returns a raw primitive or plan structure
            # ask_gpt MUST be implemented to accept text + vision_context + image urls + video link
            print("Director: calling multimodal prompter (cloud)...")
            start = time.time()
            raw = await asyncio.to_thread(ask_gpt, user_text, vision_context, images, video_link, memory)
            print(f"Prompter returned in {time.time() - start:.2f}s. Raw response: {str(raw)[:200]}")

            # 5) Convert raw response into a safe validated primitive
            primitive = to_safe_primitive(raw or {"action": "HOVER"})
            if primitive is None:
                print("Cinematic planner returned None -> HOVER")
                primitive = {"action": "HOVER", "params": {}}
                camera_choice = choose_camera_for_request(user_text, primitive, vision_context)
                primitive['meta']['camera_choice'] = camera_choice
             # send to server — server/radxa will forward camera command


            # 6) If primitive requires trajectory planning (e.g. FOLLOW/ORBIT with target position)
            # we sample a start and target position. In real production you will feed real lat/lon/alt or body-frame offsets.
            # Here we use vision_context.selected + an estimated relative target position (placeholder).
            if primitive.get("action") in ("FOLLOW", "ORBIT", "TRACK_PATH"):
                # estimate positions (placeholder) — replace with your localization pipeline
                start_pos = primitive.get("params", {}).get("start_pos") or [0.0, 0.0, 2.5]
                target_pos = primitive.get("params", {}).get("target_pos") or [1.5, 0.0, 2.5]

                # Plan shot with UltraDirector (warp for obstacles)
                curve, mode = self.ultra.plan_shot(primitive.get("params", {}), vision_context, start_pos, target_pos)
                if curve:
                    primitive["plan_curve"] = {
                        "duration": self.ultra.duration,
                        "control_points": [list(map(float, p)) for p in [curve.p0, curve.p1, curve.p2, curve.p3]]
                    }
                    primitive["meta"] = {"mode": mode}
                
                if curve is None and mode == "unsafe_hover":
                  primitive = {"action": "HOVER", "params": {}}
                  await self._send_plan(job_id, user_id, drone_id, primitive, reason="unsafe_curve")
                  return

                else:
                    print("UltraDirector could not build curve; falling back to HOVER primitive.")
                    primitive = {"action": "HOVER", "params": {}}

            # 7) Final safety validation (server and radxa will also revalidate)
            primitive = to_safe_primitive(primitive)

            # 8) Send validated plan back to VPS for Radxa to execute (or simulation if set)
            await self._send_plan(job_id, user_id, drone_id, primitive, reason="ok")

            # Optionally update memory (learning user preferences)
            # Example: save last style chosen
            try:
                last_style = primitive.get("params", {}).get("style")
                if last_style:
                    write_memory(user_id, drone_id, {"last_style": last_style})
            except Exception:
                pass

        except Exception as e:
            print("Director: Unhandled exception while processing job:", e)
            traceback.print_exc()
            # send a failure ack to server
            await self._send_plan(job_id, user_id, drone_id, {"action": "HOVER"}, reason="error")

        finally:
            self.processing = False
            print(f"=== Finished job {job_id} ===\n")

    def set_frame_source(self, source_func):
        """Allows injecting a frame source (e.g. from HTTP Video Router)"""
        self._frame_source = source_func

    def _grab_frame(self, rtsp_url: str, timeout_s: float) -> Optional[any]:
        """
        Attempts to open RTSP/local camera OR read from shared memory buffer.
        """
        # 1. PRIORITY: In-Memory Buffer (from Radxa Push)
        if hasattr(self, '_frame_source') and self._frame_source:
             f = self._frame_source()
             if f is not None:
                 # Decode if bytes
                 if isinstance(f, bytes):
                     arr = np.frombuffer(f, np.uint8)
                     return cv2.imdecode(arr, cv2.IMREAD_COLOR)
                 return f

        # 2. Fallback: RTSP / Local
        cap = cv2.VideoCapture(rtsp_url if rtsp_url else 0)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return frame
            time.sleep(0.05)
        try:
            cap.release()
        except Exception:
            pass
        return None

    async def _send_plan(self, job_id, user_id, drone_id, primitive, reason="ok"):
        """
        Send plan to VPS. Packet format:
          {
            "target":"server",
            "type":"ai_plan",
            "job_id": ...,
            "user_id": ...,
            "drone_id": ...,
            "primitive": { ... },
            "meta": { "source":"laptop", "reason": "ok" }
          }
        """
        packet = {
            "target": "server",
            "type": "ai_plan",
            "job_id": job_id,
            "user_id": user_id,
            "drone_id": drone_id,
            "primitive": primitive,
            "meta": {"source": "laptop", "reason": reason, "ts": time.time()}
        }

        # simulate mode: save to disk instead of sending
        if self.simulate:
            fname = os.path.join(TEMP_ARTIFACT_DIR, f"sim_plan_{job_id}.json")
            with open(fname, "w") as f:
                json.dump(packet, f, indent=2)
            print(f"[SIM] saved plan to {fname}")
            return

        # Production: send via MessagingClient (with retries)
        max_tries = 4
        backoff = 1.0
        for attempt in range(1, max_tries + 1):
            try:
                await self.ws.send(packet)
                print(f"Plan sent to server (job={job_id}) attempt={attempt}")
                return
            except Exception as e:
                print(f"Failed to send plan (attempt {attempt}): {e}")
                await asyncio.sleep(backoff)
                backoff = min(8.0, backoff * 2)

        print("Director: failed to send plan after retries — saved locally.")
        # fallback: persist locally for operator inspection
        fname = os.path.join(TEMP_ARTIFACT_DIR, f"unsent_plan_{job_id}.json")
        with open(fname, "w") as f:
            json.dump(packet, f, indent=2)

# --- CLI / main entrypoint ---
async def main_loop(simulate=False):
    d = Director(simulate=simulate)
    await d.start()

    # keep process alive
    try:
        while True:
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        print("Director main loop cancelled. Exiting.")

if __name__ == "__main__":
    # CLI: pass simulate=True for development testing
    sim = False
    asyncio.run(main_loop(simulate=sim))
