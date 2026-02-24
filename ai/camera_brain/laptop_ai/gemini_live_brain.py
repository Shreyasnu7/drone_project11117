"""
Gemini Live Brain — Autonomous Real-Time AI for Drone Control
=============================================================
Opens a persistent Gemini Live session. Sends video frames + sensor data
every ~2 seconds. Receives continuous JSON decisions for movement, gimbal,
obstacle avoidance, and cinematic filming.

Runs as a background async task. Director core reads the latest decision
from a thread-safe queue.
"""

import asyncio
import base64
import json
import os
import io
import time
import threading
import logging
from collections import deque

try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# System prompt for the continuous Cloud Director
DRONE_BRAIN_PROMPT = """You are the CONTINUOUS MASTER DIRECTOR of an advanced AI cinematic drone.
You think deeply every 6 seconds to update the "Master Plan" for the local reflex brain.

== YOUR CAPABILITIES ==
You receive a HIGH-RESOLUTION snapshot of the scene, plus depth data (VL53L1X ToF + YDLidar X1).
Your job is to understand the deep cinematic intent and context, and output exactly the plan you want the drone to execute for the next 6 seconds. The local brain will execute the split-second physics of your plan.

== OUTPUT FORMAT ==
Output ONLY valid JSON representing the Director's Master Plan.
{
    "er_intent": "<A dense, commanding string explaining EXACTLY what the drone should do for the next 6 seconds. Example: 'Track the red car smoothly from the left side, keeping it in the lower third, pitch gimbal down slightly, apply ACES cinematic color grading. Avoid the tree on the left.'>",
    "basic_camera_settings": {
        "exposure_compensation": "<e.g., -0.5 for moody>",
        "color_profile": "<e.g., Flat, ACES, Vibrant>",
        "sharpness": "<Low, Medium, High>"
    },
    "reasoning": "<Your deep context analysis of what is happening in the scene>"
}
"""


class GeminiLiveBrain:
    """
    Persistent Gemini Live session for continuous autonomous drone control.
    Sends video frames + sensor data, receives JSON decisions.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self.session = None
        self.running = False
        self._thread = None
        self._loop = None

        # Thread-safe output: latest decision from Gemini
        self._latest_decision = None
        self._decision_lock = threading.Lock()
        self._decision_count = 0

        # Input queue: frames + sensor data to send
        self._input_queue = deque(maxlen=3)  # Keep only latest 3

        # User mission context (set from app commands)
        self._mission = ""  # e.g. "Film a sports car ad" or "Cinematic sunset shot"

        # Stats
        self.total_decisions = 0
        self.total_errors = 0
        self.last_decision_time = 0
        self.session_start_time = 0
        self.connected = False

        # Config
        self.frame_interval = 6.0  # Time to sleep between checks
        self.max_retries = 5
        self.model = "gemini-2.0-flash"  # Max limits
        
        # Event-driven optimization: Ensure we exactly hit 10 Requests Per Minute
        self.last_api_call_time = 0
        self.min_call_interval = 6.0  # Max 1 call every 6 seconds (10 RPM)
        self.last_mission = ""

        if not GENAI_AVAILABLE:
            logger.warning("google-genai SDK not available. GeminiLiveBrain disabled.")
            return

        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found. GeminiLiveBrain disabled.")
            return

        # We now instantiate the client per-request in _call_gemini_sync 
        # to prevent httpx "Unclosed connection" spam
        logger.info("✅ GeminiLiveBrain initialized")

    def set_mission(self, mission_text: str):
        """Set the current user mission/intent for the brain to consider."""
        self._mission = mission_text
        logger.info(f"🧠 Mission updated: {mission_text[:100]}")

    def start(self):
        """Start the background brain thread."""
        if not self.client:
            logger.warning("GeminiLiveBrain: No client, cannot start.")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="GeminiLiveBrain")
        self._thread.start()
        logger.info("🧠 GeminiLiveBrain thread started")

    def connect(self):
        """Alias for start() used by DirectorCore wiring."""
        self.start()

    def stop(self):
        """Stop the brain."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("🧠 GeminiLiveBrain stopped")

    def feed(self, frame: np.ndarray, sensor_data: dict = None, detections: list = None):
        """
        Feed a new video frame + sensor data to the brain.
        Called from director_core's vision loop.
        Non-blocking — just queues the data.
        """
        self._input_queue.append({
            "frame": frame,
            "sensors": sensor_data or {},
            "detections": detections or [],
            "timestamp": time.time()
        })

    def get_latest_decision(self) -> dict:
        """
        Get the latest autonomous decision from Gemini.
        Thread-safe. Returns None if no decision yet.
        """
        with self._decision_lock:
            return self._latest_decision

    def _set_decision(self, decision: dict):
        """Thread-safe set decision."""
        with self._decision_lock:
            self._latest_decision = decision
            self._decision_count += 1

    def _run_loop(self):
        """Background thread: runs the async event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._brain_loop())
        except Exception as e:
            logger.error(f"GeminiLiveBrain loop crashed: {e}")
        finally:
            self._loop.close()

    async def _brain_loop(self):
        """Main brain loop — connect, send frames, receive decisions."""
        retry_count = 0

        while self.running and retry_count < self.max_retries:
            try:
                logger.info(f"🧠 Connecting to Gemini Live (attempt {retry_count + 1})...")
                await self._polling_brain()

            except Exception as e:
                retry_count += 1
                self.connected = False
                logger.error(f"🧠 GeminiLiveBrain error (retry {retry_count}): {e}")
                await asyncio.sleep(min(2 ** retry_count, 30))

        if retry_count >= self.max_retries:
            logger.error("🧠 GeminiLiveBrain: Max retries reached. Brain offline.")

    async def _polling_brain(self):
        """
        Polling-based brain: sends frames via generateContent every N seconds.
        More reliable than WebSocket Live API for long-running drone sessions.
        """
        self.connected = True
        self.session_start_time = time.time()
        logger.info("🧠 Gemini Brain ONLINE — polling mode")

        while self.running:
            if not self._input_queue:
                await asyncio.sleep(0.1)
                continue

            data = self._input_queue[-1]
            self._input_queue.clear()
            
            # EVENT-DRIVEN THROTTLE: Ensure we don't spam the API and exhaust 1500 limit.
            # Max 1 request every 5 seconds (12/min) to stay well within free tier.
            time_since_last = time.time() - self.last_api_call_time
            if time_since_last < self.min_call_interval and self._mission == self.last_mission:
                await asyncio.sleep(0.5)
                continue

            try:
                frame = data["frame"]
                if frame is None:
                    continue

                # High-Res Frame (1024px) for maximum spatial understanding
                h, w = frame.shape[:2]
                MAX_DIM = 1024
                if max(h, w) > MAX_DIM:
                    scale = MAX_DIM / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                _, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_b64 = base64.b64encode(jpeg_buf.tobytes()).decode('utf-8')

                context = self._build_context(data["sensors"], data["detections"])

                # Mark call time
                self.last_api_call_time = time.time()
                self.last_mission = self._mission

                response = await asyncio.to_thread(
                    self._call_gemini_sync, frame_b64, context
                )

                if response:
                    self._set_decision(response)
                    self.total_decisions += 1
                    self.last_decision_time = time.time()
                    logger.debug(f"🧠 Decision #{self.total_decisions}: {response.get('reasoning', '?')[:60]}")

            except Exception as e:
                self.total_errors += 1
                logger.error(f"🧠 Brain processing error: {e}")

            await asyncio.sleep(self.frame_interval)

    def _call_gemini_sync(self, frame_b64: str, context: str) -> dict:
        """Synchronous Gemini call with image + text. Run in thread."""
        try:
            image_part = genai_types.Part.from_bytes(
                data=base64.b64decode(frame_b64),
                mime_type="image/jpeg"
            )

            prompt = f"{DRONE_BRAIN_PROMPT}\n\n{context}\n\nAnalyze the image and sensor data. Think deeply. Output JSON only."

            with google_genai.Client(api_key=self.api_key) as client:
                response = client.models.generate_content(
                    model=self.model,
                    contents=[prompt, image_part]
                )

            text = response.text.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            data = json.loads(text)
            
            # Robust unwrap: recursively dig until we find a dictionary
            while isinstance(data, list) and len(data) > 0:
                data = data[0]
                
            if not isinstance(data, dict):
                logger.warning(f"🧠 Gemini returned invalid JSON structure: {type(data)}")
                return {"flight": {"hover": True}, "reasoning": "Invalid JSON structure", "confidence": 0.0}
                
            return data

        except json.JSONDecodeError as e:
            logger.warning(f"🧠 Gemini returned non-JSON: {e}")
            return {"flight": {"hover": True}, "reasoning": "JSON parse error", "confidence": 0.0}
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "503" in error_str:
                logger.warning(f"🧠 Gemini rate limited: {e}")
                return None
            raise

    def _build_context(self, sensors: dict, detections: list) -> str:
        """Build rich text context from ALL sensor data + YOLO detections."""
        lines = ["== LIVE ENVIRONMENT STATE =="]

        if sensors:
            # VL53L1X ToF Array
            lines.append("\n[VL53L1X ToF PROXIMITY (cm)]")
            tof_keys = ["t1", "t2", "t3", "t4"]
            tof_labels = ["FRONT", "RIGHT", "BACK", "LEFT"]
            for key, label in zip(tof_keys, tof_labels):
                val = sensors.get(key, "no_data")
                warning = " ⚠️ CLOSE!" if isinstance(val, (int, float)) and val < 80 else ""
                lines.append(f"  {label}: {val}cm{warning}")

            # YDLidar X1
            lidar = sensors.get("lidar_scan") or sensors.get("lidar") or sensors.get("ydlidar")
            if lidar:
                lines.append("\n[YDLidar X1 360° SCAN]")
                if isinstance(lidar, list):
                    # Array of (angle, distance) pairs
                    min_dist = min((d for _, d in lidar if d > 0), default=9999)
                    lines.append(f"  Closest point: {min_dist:.0f}cm")
                    lines.append(f"  Scan points: {len(lidar)}")
                elif isinstance(lidar, dict):
                    for k, v in lidar.items():
                        lines.append(f"  {k}: {v}")

            # IMU
            imu = sensors.get("imu", {})
            if imu:
                lines.append(f"\n[IMU] ax={imu.get('ax','?')} ay={imu.get('ay','?')} az={imu.get('az','?')}")

            # Telemetry
            lines.append(f"\n[TELEMETRY]")
            lines.append(f"  Altitude: {sensors.get('altitude', '?')}m")
            lines.append(f"  Battery: {sensors.get('battery', '?')}%")
            lines.append(f"  Heading: {sensors.get('heading', '?')}°")
            lines.append(f"  Speed: {sensors.get('speed', '?')}m/s")
            lines.append(f"  Flight Mode: {sensors.get('mode', '?')}")
            lines.append(f"  Armed: {sensors.get('armed', '?')}")
            gps = sensors.get("gps", {})
            if gps:
                lines.append(f"  GPS: ({gps.get('lat', '?')}, {gps.get('lng', '?')})")

            # Depth stats from MiDaS
            depth = sensors.get("depth_stats", {})
            if depth:
                lines.append(f"\n[MiDaS DEPTH MAP]")
                lines.append(f"  Min: {depth.get('min_m', '?')}m, Max: {depth.get('max_m', '?')}m, Mean: {depth.get('mean_m', '?')}m")

        # YOLO detections
        if detections:
            lines.append(f"\n[VISION — YOLO DETECTIONS] ({len(detections)} objects)")
            for det in detections[:15]:
                if isinstance(det, dict):
                    name = det.get("class", det.get("name", "unknown"))
                    conf = det.get("confidence", det.get("conf", 0))
                    bbox = det.get("bbox", det.get("box", []))
                    depth_m = det.get("depth_m", "?")
                    lines.append(f"  - {name} (conf={float(conf):.2f}, depth≈{depth_m}m) at {bbox}")
                else:
                    lines.append(f"  - {det}")

        # User mission
        if self._mission:
            lines.append(f"\n[USER MISSION]")
            lines.append(f"  {self._mission}")
            lines.append(f"  Execute this cinematically using your knowledge of professional filmmaking.")

        if not sensors and not detections:
            lines.append("No sensor data available yet. Analyze camera image only.")

        return "\n".join(lines)

    @property
    def status(self) -> dict:
        """Get brain status."""
        return {
            "connected": self.connected,
            "total_decisions": self.total_decisions,
            "total_errors": self.total_errors,
            "last_decision_age": time.time() - self.last_decision_time if self.last_decision_time else None,
            "uptime": time.time() - self.session_start_time if self.session_start_time else 0,
            "model": self.model,
        }
