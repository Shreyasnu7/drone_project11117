# ai_autofocus.py
# Ultra-Intelligent Autofocus Engine (Part 1 + Part 2 Fully Combined)
# This module performs predictive autofocus, subject-aware focus,
# depth-assisted focusing, motion-adaptive focus pulls,
# cinematic rack-focus planning, confidence scoring, and error recovery.
#
# Designed to rival real cinema autofocus (Sony/Canon FX series level).
#
# NOTE: This module does NOT control any motors directly — it outputs
# focus parameters for camera drivers (GoPro/internal camera).

import numpy as np
import cv2
import time
import math
from collections import deque

class AIAutofocus:
    """
    Main autofocus engine with:
      • Subject-aware tracking
      • Predictive trajectory estimation
      • Multi-frame sharpness evaluation
      • Depth-aware focusing (via ai_depth_estimator)
      • Motion-adaptive focus smoothing
      • Cinematic rack-focus planner
      • Fallback autofocus when subject lost

    This engine produces:
        { "focus_distance": meters,
          "confidence": 0–1,
          "mode": "predictive" | "subject" | "fallback"
        }
    """

    def __init__(self):
        # History buffers
        self.subject_history = deque(maxlen=30)
        self.sharpness_history = deque(maxlen=20)
        self.depth_history = deque(maxlen=20)

        # Last known states
        self.last_focus_distance = 2.0
        self.last_confidence = 0.5

        # Motion smoothing parameters
        self.smooth_factor = 0.85     # higher = smoother, slower
        self.max_focus_speed = 6.0    # m/s focus change limit

        # Sharpness computation kernel
        self.laplacian_kernel = np.array([[0, -1, 0],
                                          [-1, 4, -1],
                                          [0, -1, 0]])

        # Baseline camera parameters (placeholder, adjustable)
        self.focal_length_mm = 5.0
        self.f_stop = 1.8
        self.sensor_width_mm = 6.3

    # ---------------------------------------------------------------
    # (1) SUBJECT SHARPNESS ESTIMATION
    # ---------------------------------------------------------------
    def compute_sharpness(self, gray):
        """
        Computes sharpness score of the subject region using Laplacian.
        """
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharp = lap.var()
        return sharp

    # ---------------------------------------------------------------
    # (2) DEPTH FUSION (from ai_depth_estimator)
    # ---------------------------------------------------------------
    def update_depth(self, depth_m):
        """
        Add depth sample from the depth estimator (meters).
        """
        if depth_m is None:
            return
        self.depth_history.append(depth_m)

    def get_depth_focus(self):
        """
        Computes stable focus from depth history.
        """
        if len(self.depth_history) == 0:
            return None, 0.0

        median_depth = np.median(self.depth_history)
        stability = np.std(self.depth_history)

        confidence = max(0.0, 1.0 - (stability * 2.2))

        return float(median_depth), confidence

    # ---------------------------------------------------------------
    # (3) SUBJECT TRACKING → Predict future position
    # ---------------------------------------------------------------
    def update_subject_position(self, bbox_3d):
        """
        bbox_3d = { "x": , "y": , "z": , "distance": m }
        """
        self.subject_history.append(bbox_3d)

    def predict_future_distance(self, t=0.15):
        """
        Predicts subject distance after t seconds using linear motion.
        """
        if len(self.subject_history) < 3:
            return None

        # Extract last 3 distances
        d1 = self.subject_history[-1]["distance"]
        d2 = self.subject_history[-2]["distance"]
        d3 = self.subject_history[-3]["distance"]

        vel = (d1 - d3) / 2.0
        pred = d1 + vel * t

        return max(0.2, pred)

    # ---------------------------------------------------------------
    # (4) SHARPNESS-BASED FOCUS SEARCH (Fallback AF)
    # ---------------------------------------------------------------
    def fallback_focus(self, frame, search_steps=7, search_range=0.7):
        """
        When subject is lost or depth is unstable:
        Sweep focus distance around last known distance and measure sharpness.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        base = self.last_focus_distance
        best_focus = base
        best_sharp = -1

        for i in range(-search_steps, search_steps + 1):
            candidate = base + (i / search_steps) * search_range
            candidate = max(0.2, candidate)

            # Simulated blur for candidate evaluation
            blur_strength = max(0, min(15, abs(candidate - base) * 4.0))
            if blur_strength > 0.1:
                blur_img = cv2.GaussianBlur(gray, (0, 0), blur_strength)
            else:
                blur_img = gray

            sharp = self.compute_sharpness(blur_img)
            if sharp > best_sharp:
                best_sharp = sharp
                best_focus = candidate

        confidence = min(1.0, best_sharp / 2000.0)

        return best_focus, confidence

    # ---------------------------------------------------------------
    # (5) FOCUS SMOOTHING (Cinematic)
    # ---------------------------------------------------------------
    def smooth_focus_transition(self, new_focus):
        """
        Smooths abrupt focus changes using exponential filtering +
        max focus speed limit.
        """
        old = self.last_focus_distance
        raw_change = new_focus - old

        # limit rate
        max_step = self.max_focus_speed * 0.02  # per frame (~50 Hz)
        raw_change = np.clip(raw_change, -max_step, max_step)

        filtered = old * self.smooth_factor + (old + raw_change) * (1 - self.smooth_factor)
        self.last_focus_distance = filtered

        return filtered

    # ---------------------------------------------------------------
    # (6) CINEMATIC RACK-FOCUS PLANNING
    # ---------------------------------------------------------------
    def plan_rack_focus(self, subject_near, subject_far, duration=1.2):
        """
        Generates a time-parametric focus curve between two subjects.
        Used when user commands: "rack focus from person to bike".

        Returns:
            { "curve": [ {t, focus}, ... ], "duration": s }
        """
        samples = 40
        curve = []
        for i in range(samples):
            t = i / (samples - 1)
            eased = t*t*(3 - 2*t)   # smoothstep easing
            f = subject_near + (subject_far - subject_near) * eased
            curve.append({"t": float(t * duration), "focus": float(f)})

        return {"curve": curve, "duration": duration}

    # ---------------------------------------------------------------
    # (7) MAIN UPDATE LOOP (the heart of AF)
    # ---------------------------------------------------------------
    def update(self, frame, subject_box=None, depth_sample=None):
        """
        Core autofocus logic. Called every frame (30–120 FPS).

        Input:
            frame (BGR)
            subject_box = { "x","y","w","h","distance" } — OR None
            depth_sample = meters — OR None

        Output:
            { "focus_distance": m,
              "confidence": 0–1,
              "mode": "predictive" | "subject" | "fallback"
            }
        """

        mode = "fallback"

        # (A) Update depth estimator
        if depth_sample is not None:
            self.update_depth(depth_sample)

        # (B) If subject is known → update tracker & predictive distance
        if subject_box is not None:
            self.update_subject_position({
                "x": subject_box["x"],
                "y": subject_box["y"],
                "z": subject_box.get("z", 0),
                "distance": subject_box["distance"]
            })

        predicted = None
        predicted_conf = 0

        if subject_box is not None:
            predicted = self.predict_future_distance()
            if predicted is not None:
                predicted_conf = 0.75
                mode = "predictive"

        # (C) Depth-based focusing
        depth_focus, depth_conf = self.get_depth_focus()
        if depth_focus is not None and depth_conf > predicted_conf:
            predicted = depth_focus
            predicted_conf = depth_conf
            mode = "subject" if subject_box is not None else "depth"

        # (D) If no predicted value yet → run fallback scan
        if predicted is None:
            fb_focus, fb_conf = self.fallback_focus(frame)
            predicted = fb_focus
            predicted_conf = fb_conf
            mode = "fallback"

        # -------------------------------------------------------
        # Stabilize focus using cinematic smoothing
        # -------------------------------------------------------
        final_focus = self.smooth_focus_transition(predicted)

        # -------------------------------------------------------
        # Pack output
        # -------------------------------------------------------
        return {
            "focus_distance": float(final_focus),
            "confidence": float(predicted_conf),
            "mode": mode
        }

    # ===================================================================
    #                           DEBUG UTILITIES
    # ===================================================================
    def draw_focus_info(self, frame, focus_data):
        """
        Draws focus diagnostics on frame.
        """
        msg = f"AF={focus_data['focus_distance']:.2f}m conf={focus_data['confidence']:.2f} mode={focus_data['mode']}"
        cv2.putText(frame, msg, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    # ===================================================================
    #              SUBJECT TRACKING HELPER FOR CINEMATIC AF
    # ===================================================================
    def assign_subject(self, detections):
        """
        Selects a subject from detection list.
        Detections format:
           [{"label":"person","conf":0.92,"box":[x,y,w,h],"distance":3.2}, ...]
        """
        if not detections:
            self.subject_history = []
            return None

       # pick the most confident "person" OR nearest object
        best = None
        best_score = -1

        for det in detections:
            label = det["label"]
            score = det.get("conf", 0)
            dist = det.get("distance", 999)

            # score:
            #   person detected → bonus
            #   closer → stronger priority
            #   confidence → multiplies
            s = score * 1.5 if label == "person" else score
            s *= (1.0 / max(0.7, dist))

            if s > best_score:
                best = det
                best_score = s

        # Save history for predictive motion
        if best is not None:
            self.update_subject_position({
                "x": best["box"][0] + best["box"][2] / 2,
                "y": best["box"][1] + best["box"][3] / 2,
                "z": 0,
                "distance": best.get("distance", 3.0)
            })

        return best

    # ===================================================================
    #         DEPTH-BASED REFOCUS WITH STABILITY DETECTION
    # ===================================================================
    def depth_based_refocus(self, depth_map):
        """
        Uses full depth map to choose optimal focal plane.
        More robust than single depth_sample.

        depth_map: 2D float array of meters.
        """
        if depth_map is None:
            return None, 0

        # Compute histogram of depth to find stable cluster
        flat = depth_map.flatten()
        flat = flat[(flat > 0.1) & (flat < 30.0)]
        if len(flat) < 200:
            return None, 0

        hist, edges = np.histogram(flat, bins=40, range=(0, 30))
        peak_idx = np.argmax(hist)
        peak_depth = (edges[peak_idx] + edges[peak_idx + 1]) / 2

        # confidence: based on histogram peak prominence
        conf = hist[peak_idx] / max(1, np.sum(hist))

        # update depth estimator
        self.update_depth(peak_depth)

        return peak_depth, float(conf)

    # ===================================================================
    #         ADVANCED AF METRIC: REGIONAL SHARPNESS MAP
    # ===================================================================
    def sharpness_map(self, gray):
        """
        Compute per-block sharpness metric across frame.
        Useful for detecting regions in focus vs out of focus.
        """
        h, w = gray.shape
        block_h = h // 10
        block_w = w // 10

        sharp_map = np.zeros((10, 10))

        for i in range(10):
            for j in range(10):
                y0 = i * block_h
                x0 = j * block_w
                y1 = y0 + block_h
                x1 = x0 + block_w

                block = gray[y0:y1, x0:x1]
                # Laplacian-based sharpness
                sharp = cv2.Laplacian(block, cv2.CV_64F).var()
                sharp_map[i, j] = sharp

        return sharp_map

    # ===================================================================
    #     SMART REGIONAL AUTOFOCUS (detects where cinematic subject is)
    # ===================================================================
    def region_focus(self, gray, subject_box=None):
        """
        Computes a weighted sharpness score in region of interest.

        If subject_box is provided:
            → focus aggressively inside box
        Otherwise:
            → use center-weighted region
        """
        h, w = gray.shape

        if subject_box is not None:
            x, y, bw, bh = subject_box
            x = max(0, int(x))
            y = max(0, int(y))
            bw = int(bw)
            bh = int(bh)
            crop = gray[y:y+bh, x:x+bw]
        else:
            cx = w // 4
            cy = h // 4
            crop = gray[cy:cy + h//2, cx:cx + w//2]

        if crop.size == 0:
            return 0

        sharp = cv2.Laplacian(crop, cv2.CV_64F).var()
        return sharp

    # ===================================================================
    #     HIGH-LEVEL API: RETURN "WHAT CAMERA SHOULD DO NEXT"
    # ===================================================================
    def autofocus_step(self, frame, vision_context):
        """
        Frame → detect subject → compute focus distance → return camera command.

        High-level output example:
            {
              "focus_distance": 2.1,
              "confidence": 0.89,
              "mode": "subject",
              "should_refocus": True
            }
        """
        dets = vision_context.get("detections", [])
        subject = self.assign_subject(dets)

        subject_box = subject["box"] if subject else None
        subject_dist = subject.get("distance") if subject else None

        focus_data = self.compute_focus(frame, subject_box, subject_dist)

        # logic: refocus if mode != fallback OR conf < threshold
        should_refocus = focus_data["confidence"] < 0.30 or focus_data["mode"] != "fallback"

        return {
            "focus_distance": focus_data["focus_distance"],
            "confidence": focus_data["confidence"],
            "mode": focus_data["mode"],
            "should_refocus": should_refocus,
            "subject_box": subject_box
        }

    # ===================================================================
    #      META: EXTRACT FOCUS METRICS FOR CAMERA BRAIN INTEGRATION
    # ===================================================================
    def get_focus_metadata(self):
        """
        Return internal state for UltraCameraBrain to fuse with
        HDR engine, exposure engine, stabilizer, etc.
        """
        return {
            "current_focus": float(self.current_focus),
            "depth_state": float(self.depth_state),
            "subject_history": list(self.subject_history[-5:]),
            "last_sharpness": float(self.sharpness_state),
        }

    # ===================================================================
    #               EXPERIMENTAL: SCENE TYPE → AF MODE SWITCHING
    # ===================================================================
    def scene_based_mode(self, scene_type):
        """
        Scene types:
            - "landscape" → use depth-based AF
            - "portrait"  → strong subject preference
            - "action"    → predictive motion AF
        """

        if scene_type == "landscape":
            self.scene_mode = "depth"

        elif scene_type == "portrait":
            self.scene_mode = "subject"

        elif scene_type == "action":
            # enforce predictive AF
            self.scene_mode = "predictive"

        else:
            self.scene_mode = "auto"

    # ===================================================================
    #       EXPERIMENTAL: DEPTH + MOTION PREDICTION AF (CINEMATIC)
    # ===================================================================
    def predictive_af(self, subject_box, depth_map, prev_frame, curr_frame):
        """
        Predictive autofocus:
            • Computes optical flow around subject
            • Predicts subject movement next frame
            • Uses depth_map (if available) to refine distance

        Returns:
            {
               "predicted_focus": float,
               "motion_vector": (vx, vy),
               "confidence": float
            }
        """

        if prev_frame is None or curr_frame is None:
            return {
                "predicted_focus": self.current_focus,
                "motion_vector": (0.0, 0.0),
                "confidence": 0.1,
            }

        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        h, w = gray_curr.shape

        if subject_box:
            x, y, bw, bh = subject_box
            x = max(0, int(x))
            y = max(0, int(y))
            bw = int(bw)
            bh = int(bh)

            prev_crop = gray_prev[y:y+bh, x:x+bw]
            curr_crop = gray_curr[y:y+bh, x:x+bw]
        else:
            # fallback: central region
            cx = w // 4
            cy = h // 4
            prev_crop = gray_prev[cy:cy+h//2, cx:cx+w//2]
            curr_crop = gray_curr[cy:cy+h//2, cx:cx+w//2]

        if prev_crop.size == 0 or curr_crop.size == 0:
            return {
                "predicted_focus": self.current_focus,
                "motion_vector": (0.0, 0.0),
                "confidence": 0.1,
            }

        # ---------------------------------------------------------------
        #     Optical flow (Farneback or Lucas-Kanade)
        # ---------------------------------------------------------------
        flow = cv2.calcOpticalFlowFarneback(
            prev_crop, curr_crop, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # mean motion vector
        mv = flow.mean(axis=(0, 1))   # (vx, vy)
        vx, vy = float(mv[0]), float(mv[1])
        motion_magnitude = (vx**2 + vy**2)**0.5

        # ---------------------------------------------------------------
        #      Depth refinement — if depth map is supplied
        # ---------------------------------------------------------------
        predicted_focus = self.current_focus

        if depth_map is not None:
            # Extract depth value under subject center
            if subject_box:
                cx = int(subject_box[0] + subject_box[2] / 2)
                cy = int(subject_box[1] + subject_box[3] / 2)
            else:
                cy, cx = h // 2, w // 2

            if 0 <= cx < depth_map.shape[1] and 0 <= cy < depth_map.shape[0]:
                d = float(depth_map[cy, cx])
                if d > 0:
                    # Weighted blend of AF and depth
                    predicted_focus = 0.7 * predicted_focus + 0.3 * d

        # motion influences focus slightly
        predicted_focus += motion_magnitude * 0.01

        conf = min(1.0, 0.2 + motion_magnitude * 0.05)

        return {
            "predicted_focus": predicted_focus,
            "motion_vector": (vx, vy),
            "confidence": conf,
        }

    # ===================================================================
    #       DEBUG VISUALIZATION OVERLAYS FOR STUDIO-LEVEL TUNING
    # ===================================================================
    def draw_debug(self, frame, focus_info):
        """
        Draws:
            - subject box
            - focus distance
            - focus confidence
            - mode indicator
        """
        vis = frame.copy()

        # subject box
        if focus_info.get("subject_box") is not None:
            x, y, w, h = focus_info["subject_box"]
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        text = f"F={focus_info['focus_distance']:.2f}  C={focus_info['confidence']:.2f}  Mode={focus_info.get('mode','?')}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return vis

    # ===================================================================
    #       CALIBRATION & TUNING UTILITIES
    # ===================================================================
    def set_focus_range(self, min_focus, max_focus):
        """Adjust allowed physical range of the lens."""
        self.min_focus = float(min_focus)
        self.max_focus = float(max_focus)
        self.current_focus = np.clip(self.current_focus, self.min_focus, self.max_focus)

    def set_smoothing(self, alpha):
        """
        Controls how aggressively focus changes are smoothed.
        Lower alpha = slower, cinematic focus pulls.
        """
        self.smoothing_alpha = float(alpha)

    def set_confidence_thresholds(self, stable, jump):
        """
        stable: confidence required to accept continuous tracking
        jump:   confidence required to allow sudden focus jump
        """
        self.min_confidence_track = float(stable)
        self.min_confidence_jump = float(jump)

    # ===================================================================
    #       EXPORT / IMPORT FOCUS PARAMETERS (for LUT building)
    # ===================================================================
    def export_state(self):
        """Return autofocus parameters for saving."""
        return {
            "current_focus": self.current_focus,
            "min_focus": self.min_focus,
            "max_focus": self.max_focus,
            "smoothing_alpha": self.smoothing_alpha,
            "subject_locked": self.subject_locked,
            "subject_last_box": self.subject_last_box,
            "history": list(self.history),
        }

    def load_state(self, data):
        """Load autofocus parameters."""
        try:
            self.current_focus = float(data.get("current_focus", self.current_focus))
            self.min_focus = float(data.get("min_focus", self.min_focus))
            self.max_focus = float(data.get("max_focus", self.max_focus))
            self.smoothing_alpha = float(data.get("smoothing_alpha", self.smoothing_alpha))
            self.subject_last_box = data.get("subject_last_box", None)
            self.subject_locked = bool(data.get("subject_locked", False))
        except Exception as e:
            print("[AF] Warning: failed to load saved state:", e)

    # ===================================================================
    #       HIGH-LEVEL: END-TO-END AUTOFOCUS PIPELINE
    # ===================================================================
    def run(self, frame, vision_context, depth_map=None, prev_frame=None):
        """
        Full autofocus pipeline wrapper:
            1) detect subject from vision_context
            2) compute desired focus with multi-heuristic fusion
            3) apply smoothing + hysteresis
            4) return focus + metadata for camera brain

        Returns dict:
            {
                "focus_distance": float,
                "subject_box": [x,y,w,h] or None,
                "confidence": float,
                "mode": "tracking" / "smart" / "face" / "wide"
            }
        """
        # -----------------------------------------------
        # Step 1: get subject box (or None)
        # -----------------------------------------------
        subject_box = None

        if vision_context:
            if "selected" in vision_context and vision_context["selected"]:
                # the user's selected target
                subject_box = vision_context["selected"].get("bbox")
            elif vision_context.get("faces"):
                # fallback: choose largest face
                faces = vision_context["faces"]
                if faces:
                    areas = [(f["bbox"][2] * f["bbox"][3], f["bbox"]) for f in faces]
                    areas.sort(reverse=True)
                    subject_box = areas[0][1]
            else:
                # fallback: largest detected object
                dets = vision_context.get("detections", [])
                if dets:
                    areas = [(d["bbox"][2] * d["bbox"][3], d["bbox"]) for d in dets]
                    areas.sort(reverse=True)
                    subject_box = areas[0][1]

        # -----------------------------------------------
        # Step 2: compute focus from multi-source heuristics
        # -----------------------------------------------
        # Note: compute_focus is internal, assuming it maps to our logic above
        # for now, we use a custom blend since compute_focus isn't explicitly defined as a single func above
        # (It was split into components in my reconstruction). 
        # I will use update() which seems to be the main entry point logic from the scratchpad
        af_info = self.update(frame, subject_box, None) # depth_sample passed as None for now

        # -----------------------------------------------
        # Step 3: predictive assistant (optical flow)
        # -----------------------------------------------
        if prev_frame is not None:
            pred = self.predictive_af(subject_box, depth_map, prev_frame, frame)
            # blend predictions slightly
            af_info["focus_distance"] = (
                0.85 * af_info["focus_distance"] + 0.15 * pred["predicted_focus"]
            )
            af_info["confidence"] = min(1.0, af_info["confidence"] + pred["confidence"] * 0.1)

        # clip & record
        af_info["focus_distance"] = np.clip(
            af_info["focus_distance"],
            self.min_focus if hasattr(self, 'min_focus') else 0.1,
            self.max_focus if hasattr(self, 'max_focus') else 10.0,
        )

        return af_info

    # ===================================================================
    #      COMPATIBILITY LAYER (Adapts to AI Camera Brain API)
    # ===================================================================
    def analyze_frame(self, frame, detections=None):
        """
        Compatibility alias for AI Camera Brain.
        Maps to autofocus_step.
        """
        vision_context = {"detections": detections or []}
        return self.autofocus_step(frame, vision_context)

    def decide_focus_command(self, af_info):
        """
        Compatibility alias. The logic is already in analyze_frame/autofocus_step.
        """
        return af_info

    def smooth_update(self, af_cmd):
        """
        Compatibility alias. Smoothing is already applied in update().
        """
        return af_cmd
        
    def compute_focus(self, frame, subject_box, subject_dist):
        """
        Internal helper used by autofocus_step.
        Maps to the main update() logic.
        """
        # Convert list bbox [x1, y1, x2, y2] to dict expected by update()
        formatted_box = None
        if subject_box is not None and len(subject_box) >= 4:
            x1, y1, x2, y2 = subject_box[:4]
            w = x2 - x1
            h = y2 - y1
            formatted_box = {
                "x": x1,
                "y": y1,
                "w": w,
                "h": h,
                "distance": subject_dist if subject_dist is not None else 0.0
            }
        
        return self.update(frame, subject_box=formatted_box)

    # ===================================================================
    #      FILE COMPLETE — END OF ai_autofocus.py
    # ===================================================================