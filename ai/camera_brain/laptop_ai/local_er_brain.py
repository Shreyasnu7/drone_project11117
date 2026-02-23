import cv2
import time
import json
import base64
import torch
import numpy as np
import threading
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging

logger = logging.getLogger('LocalERBrain')

# ER Brain Prompt - Fast, strictly spatial, 0.1s execution
ER_BRAIN_PROMPT = """You are the Local Embodied Reasoning (ER) Brain of an advanced cinematic drone. 
You run locally at high speed to execute the High-Level Intent provided by the Cloud Director.

== YOUR CAPABILITIES ==
You have direct control over:
1. Flight (velocity x, y, z, yaw)
2. Gimbal (pitch, yaw)
3. 150+ Cinematic AI Modules (ACES color, stabilizers, bloom, etc.)

== CRITICAL NAVIGATION RULES ==
- < 1 METER: STRICTLY use VL53L1X ToF Array and YDLidar X1 for cm-level precision.
- 1 TO 10 METERS: Use Optical Flow + Pi0-FAST.

== OUTPUT FORMAT ==
Output ONLY valid JSON.
{
    "flight": {"vx": 0, "vy": 0, "vz": 0, "yaw_rate": 0, "hover": true/false},
    "gimbal": {"pitch": 0, "yaw": 0},
    "camera_ai": {<activate cinematic modules based on Director's Intent>},
    "obstacle_alert": <true/false>,
    "reasoning": "<brief spatial/intent reasoning>"
}
"""

class LocalERBrain:
    def __init__(self):
        self.connected = False
        self.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        # We process extremely small frames to keep latency at ~0.1 - 0.2s
        self.max_frame_dim = 256
        self.min_interval = 0.5  # Max 2 FPS to allow GPU breathing room
        
        # State
        self.last_api_call_time = 0
        self.current_director_intent = "Hover and wait for instructions."
        self._input_queue = []
        self._latest_decision = None
        self._lock = threading.Lock()
        
        self.worker_thread = threading.Thread(target=self._er_loop, daemon=True)
        
    def connect(self):
        logger.info(f"Loading {self.model_id} into VRAM... (Using approx 5GB)")
        try:
            # Load model and processor to RTX 5070 Ti (cuda)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.connected = True
            self.worker_thread.start()
            logger.info("Local ER Brain Online: Zero-latency spatial reasoning active.")
            return True
        except Exception as e:
            logger.error(f"Failed to load Local ER Brain: {e}")
            return False

    def set_director_intent(self, intent_text):
        """Called when Gemini 3 Flash (Cloud) sends a new master plan."""
        with self._lock:
            self.current_director_intent = intent_text
            logger.info(f"Local ER received new Director Intent: {intent_text[:50]}...")

    def update_state(self, raw_frame, sensors, detections):
        if not self.connected:
            return
            
        with self._lock:
            # Keep only the freshest frame to prevent lag queueing
            self._input_queue = [{
                "frame": raw_frame.copy() if raw_frame is not None else None,
                "sensors": sensors,
                "detections": detections
            }]

    def get_latest_decision(self):
        with self._lock:
            return self._latest_decision
            
    def _build_sensor_text(self, sensors, detections):
        lines = []
        # Add high-level intent from Gemini
        lines.append(f"DIRECTOR'S TARGET INTENT: {self.current_director_intent}")
        lines.append("---")
        
        if sensors.get('tof'):
            t = sensors['tof']
            lines.append(f"ToF (cm): F={t.get('t1', -1)}, R={t.get('t2', -1)}, B={t.get('t3', -1)}, L={t.get('t4', -1)}")
        if sensors.get('lidar_min_dist'):
            lines.append(f"Lidar Min Dist: {sensors['lidar_min_dist']}cm at {sensors.get('lidar_min_angle')}deg")
        if sensors.get('altitude'):
            lines.append(f"Altitude: {sensors['altitude']}m")
            
        if detections:
            dets = [f"{d['class']} ({d.get('confidence', 0):.2f})" for d in detections[:3]]
            lines.append(f"YOLO Specs: {', '.join(dets)}")
            
        return "\n".join(lines)

    def _er_loop(self):
        """Continuous background loop running VLM inference."""
        while self.connected:
            time.sleep(0.01)
            
            with self._lock:
                if not self._input_queue:
                    continue
                
                time_since_last = time.time() - self.last_api_call_time
                if time_since_last < self.min_interval:
                    continue
                    
                data = self._input_queue[-1]
                self._input_queue.clear()
                intent = self.current_director_intent
                
            try:
                frame = data["frame"]
                if frame is None:
                    continue
                    
                # Downsize frame for ~0.1s inference time
                h, w = frame.shape[:2]
                if max(h, w) > self.max_frame_dim:
                    scale = self.max_frame_dim / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    
                # Convert BGR to RGB for PiL/Qwen
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                sensor_text = self._build_sensor_text(data["sensors"], data["detections"])
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            # We pass the image as a tensor/numpy array via the processor
                            {"type": "image", "image": rgb_frame},
                            {"type": "text", "text": f"{ER_BRAIN_PROMPT}\n\nSENSOR DATA:\n{sensor_text}\n\nDecide immediately based on the image and sensors. Output JSON only."}
                        ]
                    }
                ]

                # Preparation for inference
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to("cuda")

                self.last_api_call_time = time.time()
                
                # Inference: Fast generation, minimal tokens
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # Parse JSON
                try:
                    # Clean markdown block if present
                    if "```json" in output_text:
                        output_text = output_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in output_text:
                        output_text = output_text.split("```")[1].strip()
                        
                    decision = json.loads(output_text)
                    with self._lock:
                        self._latest_decision = decision
                except json.JSONDecodeError:
                    logger.error(f"ER Brain output invalid JSON: {output_text[:100]}...")
                    
            except Exception as e:
                logger.error(f"ER Brain inference error: {e}")
                time.sleep(2)
