# File: laptop_ai/ai_camera_brain.py
"""
AICameraBrain
=============

The *central intelligence* for all camera-related decisions in the Drone AI System.

This module does **NOT** control the drone or motors.
It only decides:
    • autofocus parameters
    • exposure (AE) parameters
    • stabilization transforms
    • color grading presets
    • scene classification
    • tone mapping grid
    • camera prioritization (GoPro vs internal)
    • encoder preferences (fps, bitrate, resolution)
    • cinematic mode adjustments based on user text

Everything returned is SAFE: metadata only.
Actuation is performed by camera drivers (GoProDriver, pi_camera_driver)
and drone motion is controlled strictly by Autopilot.

This file is designed for large-scale expansion (60000+ project lines).
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional

import numpy as np
import cv2

# Import AI modules implemented previously
# Import AI modules implemented previously
from laptop_ai.ai_autofocus import AIAutofocus
from laptop_ai.ai_exposure_engine import AIExposureEngine
from laptop_ai.ai_stabilizer import AIStabilizer
from laptop_ai.ai_scene_classifier import SceneClassifier
from laptop_ai.ai_color_engine import AIColorEngine
# RE-WIRED (Audit Fix)
from laptop_ai.ai_camera_pipeline import AICameraPipeline 
from laptop_ai.ai_deblur import AIDeblur 
from laptop_ai.ai_motion_blur_controller import AIMotionBlurController
from laptop_ai.ai_pipeline.tone_curve.gpu_aces import GPUACESToneCurve

# Farming & Core Brain Wiring (Audit Fix)
try:
    try:
        from farming.farming_engine import FarmingEngine
        from core.camera_brain import CameraBrain as CameraBrainCore
    except ImportError:
        from camera_brain.farming.farming_engine import FarmingEngine
        from camera_brain.core.camera_brain import CameraBrain as CameraBrainCore
except ImportError:
    FarmingEngine = None
    CameraBrainCore = None


class AICameraBrain:
    """
    Central orchestrator combining all camera-related AI modules.

    Public method:
        decide(user_text, fusion_state, frame, vision_context) -> camera_plan dict

    This result gets attached to primitive["camera_plan"] inside director_core.py
    and sent to Radxa → camera drivers.
    """

    def __init__(self):
        # Core AI modules
        self.autofocus = AIAutofocus()
        self.exposure = AIExposureEngine()
        self.stabilizer = AIStabilizer()
        self.scene = SceneClassifier()
        self.stabilizer = AIStabilizer()
        self.gpu_aces = GPUACESToneCurve()
        self.color = AIColorEngine()
        
        # Wired Sub-systems
        self.farming = FarmingEngine() if FarmingEngine else None
        self.core_brain = CameraBrainCore() if CameraBrainCore else None
        
        # RE-WIRED (Audit Fix)
        # Import classes locally to avoid circulars if needed, or rely on top-level
        from laptop_ai.ai_depth_estimator import AIDepthEstimator
        from laptop_ai.ai_hdr_engine import AIHDREngine
        
        self.pipeline = AICameraPipeline()
        self.deblur = AIDeblur()
        self.shutter_ctrl = AIMotionBlurController()
        self.depth_estimator = AIDepthEstimator()
        self.hdr_engine = AIHDREngine()

        # History buffers
        self.last_scene = None
        self.last_plan_ts = 0

        # Default recording preferences
        self.default_specs = {
            "fps": 60,
            "resolution": "4K",
            "bitrate": "high",
            "encoding": "h264",     # safe default; user can request log/h265 later
        }

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------

    def _compute_subject_metadata(self, vision_context):
        """
        Extracts selected subject info for autofocus + exposure weighting.
        vision_context expected format:
            {
                "tracks": [...],
                "selected": {"id", "bbox", "distance", ...} or None,
                ...
            }
        """
        selected = vision_context.get("selected")
        if selected:
            subj = {
                "id": selected.get("id"),
                "bbox": selected.get("bbox"),
                "distance_m": selected.get("distance_m") or None
            }
            return subj
        return None

    def _infer_recording_specs(self, user_text: str, scene_labels: Dict[str,float]) -> Dict[str,Any]:
        """
        Determines fps + resolution based on user request and scene.
        Examples:
            “ultra slow motion” → 120-240fps
            “cinematic” → 24/30fps + log color
            “sports / action” → 60-120fps
            “night recording” → lower fps, longer shutter
        """
        text = user_text.lower()

        fps = self.default_specs["fps"]
        res = self.default_specs["resolution"]
        bit = self.default_specs["bitrate"]
        enc = self.default_specs["encoding"]

        if "slow" in text or "slowmo" in text or "slow motion" in text:
            fps = 120
            enc = "h265"

        if "cinematic" in text or "film" in text:
            fps = 24
            enc = "log"

        if scene_labels.get("action", 0) > 0.55:
            fps = 60 if fps < 60 else fps
            bit = "very_high"

        if scene_labels.get("night", 0) > 0.6:
            fps = 24
            bit = "high"

        return {
            "fps": fps,
            "resolution": res,
            "bitrate": bit,
            "encoding": enc
        }

    # -------------------------------------------------------------------------
    # PUBLIC MAIN METHOD
    # -------------------------------------------------------------------------

    def decide(
        self,
        user_text: str,
        fusion_state: Dict[str,Any],
        frame: Optional[np.ndarray],
        vision_context: Optional[Dict[str,Any]] = None
    ) -> Dict[str,Any]:
        """
        Main entry point.
        Returns a fully formed camera_plan that director_core.py stores inside primitive["camera_plan"].

        Steps:
            1. Scene classification
            2. Autofocus
            3. Exposure
            4. Stabilization
            5. Color grading
            6. Recording specs
            7. Fuse with fusion_state (GoPro + internal)
        """

        ts = time.time()
        if frame is None:
            # If no frame, return conservative safe plan
            return {
                "timestamp": ts,
                "mode": "fail_safe",
                "error": "no_frame",
                "recording": self.default_specs
            }

        # ---------------------------------------------------------------------
        # 1) Scene classification
        # ---------------------------------------------------------------------
        scene_labels = self.scene.predict(frame)
        self.last_scene = scene_labels

        # ---------------------------------------------------------------------
        # 2) Subject metadata (for focus + exposure weighting)
        # ---------------------------------------------------------------------
        subject = None
        if vision_context:
            subject = self._compute_subject_metadata(vision_context)

        # ---------------------------------------------------------------------
        # 3) Autofocus
        # ---------------------------------------------------------------------
        af_info = self.autofocus.analyze_frame(frame, detections=[subject] if subject else None)
        af_cmd = self.autofocus.decide_focus_command(af_info)
        af_cmd = self.autofocus.smooth_update(af_cmd)

        # ---------------------------------------------------------------------
        # 4) Exposure + tone map
        # ---------------------------------------------------------------------
        stats = self.exposure.analyze_scene(frame)
        ae_cmd = self.exposure.propose_exposure(stats)
        tone_map = self.exposure.local_tone_map(frame)

        # ---------------------------------------------------------------------
        # 5) Stabilization (gyro+flow)
        # ---------------------------------------------------------------------
        # fusion_state may contain “gyro”: {"gx","gy","gz"}
        gyro = fusion_state.get("gyro") if fusion_state else None
        stab_cmd = self.stabilizer.update(frame, gyro)

        # ---------------------------------------------------------------------
        # 6) Color grading
        # ---------------------------------------------------------------------
        grade = self.color.propose_grade(stats)
        # NOTE: We do not apply full grading here; we send instructions for camera driver OR offline pipeline.

        # ---------------------------------------------------------------------
        # 7) Recording specs
        # ---------------------------------------------------------------------
        rec_specs = self._infer_recording_specs(user_text, scene_labels)

        # ---------------------------------------------------------------------
        # 8) Camera fusion (GoPro + internal)
        # ---------------------------------------------------------------------
        camera_choice = fusion_state.get("camera_choice", "auto")
        # In future steps:
        # camera_selector.py will override this dynamically

        # ---------------------------------------------------------------------
        # 9) FINAL CAMERA PLAN JSON
        # ---------------------------------------------------------------------
        plan = {
            "timestamp": ts,
            "scene": scene_labels,
            "subject": subject,

            "autofocus": af_cmd,
            "exposure": ae_cmd,
            "tone_map": tone_map.tolist(),
            "stabilization": stab_cmd,
            "color_grade": grade,

            "recording": rec_specs,
            "camera_choice": camera_choice,

            "meta": {
                "latency_ms": int((time.time() - ts) * 1000),
                "version": "1.0.0"
            }
        }

        self.last_plan_ts = ts
        return plan


# -----------------------------------------------------------------------------
# Example stand-alone test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    brain = AICameraBrain()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        plan = brain.decide(
            user_text="cinematic follow shot of the person",
            fusion_state={"gyro": {"gz": 0.02}, "camera_choice": "gopro"},
            frame=frame,
            vision_context={"selected": {"id": 1, "bbox": [100,100,100,150], "distance_m": 3.2}}
        )

        print("\n--- CAMERA PLAN ---")
        print(plan)

        cv2.imshow("input", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
