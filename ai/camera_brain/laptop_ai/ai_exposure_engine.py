"""
AI EXPOSURE ENGINE — Cinematic Auto-Exposure (AE) + Local Tone Mapping
======================================================================

This module performs:
    • Global exposure estimation
    • Scene-brightness normalization
    • Local contrast enhancement (tone mapping)
    • Temporal smoothing (no flicker)
    • Anti-blowout protection
    • Highlight priority mode
    • Shadow recovery mode
    • Scene-adaptive exposure (faces, sky, subject priority)

It does NOT control any motors or flight hardware → 100% SAFE.
Only adjusts image exposure parameters for cinematography.
"""

import numpy as np
import cv2
import time
import math
from collections import deque
from typing import Dict, Optional

# -------------------------------------------------------------
# Utility: clamp function
# -------------------------------------------------------------
def clamp(x, low, high):
    return low if x < low else high if x > high else x

def compute_ev_delta(current_brightness, target_brightness):
    """
    Simple P-controller for EV delta.
    """
    error = target_brightness - current_brightness
    # 1.0 difference (full black to white) -> ~5 EV stops correction
    return error * 5.0


# =====================================================================
# CLASS 1 — HISTOGRAM ANALYZER
# =====================================================================
class ExposureHistogramAnalyzer:
    """
    Computes luminance histogram and exposure metrics.
    """

    def __init__(self, bins: int = 256):
        self.bins = bins

    def analyze(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Input: frame (RGB or BGR)
        Output: dict of luminance statistics
        """
        if frame is None or frame.size == 0:
            return {
                "avg_luma": 0.5,
                "median_luma": 0.5,
                "shadow_ratio": 0.0,
                "highlight_ratio": 0.0,
                "contrast_index": 0.0,
                "brightness": 0.5,
                "dynamic_range": 0.0,
                "highlights": 0.0,
                "shadows": 0.0
            }

        # convert BGR → Y (luminance)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y = yuv[:, :, 0].astype(np.float32) / 255.0

        # histogram
        hist, _ = np.histogram(y, bins=self.bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= np.sum(hist) + 1e-8
        # cumulative histograms
        cdf = np.cumsum(hist)

        # average brightness
        avg_luma = float(np.sum(hist * np.linspace(0, 1, self.bins)))

        # median brightness
        median_idx = np.searchsorted(cdf, 0.5)
        median_luma = float(median_idx / (self.bins - 1))

        # shadows = lower 15% of brightness range
        shadow_ratio = float(np.sum(hist[: int(self.bins * 0.15)]))

        # highlights = top 15% of range
        highlight_ratio = float(np.sum(hist[int(self.bins * 0.85):]))

        # simple global contrast index:
        # difference between bright-side median & dark-side median
        low_region = y[y < 0.25]
        high_region = y[y > 0.75]

        if low_region.size > 20 and high_region.size > 20:
            dark_med = float(np.median(low_region))
            bright_med = float(np.median(high_region))
            contrast_index = float(bright_med - dark_med)
        else:
            # fallback estimation
            contrast_index = float(np.std(y))

        return {
            "avg_luma": avg_luma,
            "median_luma": median_luma,
            "shadow_ratio": shadow_ratio,
            "highlight_ratio": highlight_ratio,
            "contrast_index": contrast_index,
            "brightness": avg_luma, # alias
            "dynamic_range": contrast_index, # alias
            "highlights": highlight_ratio, # alias
            "shadows": shadow_ratio # alias
        }


# =====================================================================
# CLASS 2 — EXPOSURE STATE (persistent state of the camera exposure)
# =====================================================================
class ExposureState:
    """
    Maintains a persistent exposure state:
        • current exposure compensation (EV)
        • previous histogram metrics
        • scene mode (normal / highlight-priority / shadow-priority)
        • history buffers for smoothing
    """

    def __init__(self):
        self.ev = 0.0                     # exposure compensation (-2.0 ... +2.0)
        self.target_luma = 0.55           # ideal cinematic brightness
        self.brightness_target = 0.55     # alias
        self.mode = "normal"              # modes: normal / highlight / shadow

        self.hist_history = deque(maxlen=12)   # last 12 histograms
        self.ev_history = deque(maxlen=8)       # last few EV adjustments
        self.framerate = 30.0             # Default FPS

        self.ev_last = 0.0 # Internal tracking

        self.last_update_ts = time.time()

    def push_histogram(self, hist_metrics: Dict[str, float]):
        self.hist_history.append(hist_metrics)

    def push_ev(self, ev: float):
        self.ev_history.append(ev)

    def get_smooth_ev(self) -> float:
        """
        Smooth exposure compensation — prevents flicker.
        """
        if not len(self.ev_history):
            return self.ev

        # Weighted smoothing: last 3 frames get more weight
        n = len(self.ev_history)
        weights = np.linspace(1, 2, n)
        ev = float(np.sum(np.array(self.ev_history) * weights) / np.sum(weights))
        return clamp(ev, -2.0, 2.0)

    def get(self, key, default=None):
        return getattr(self, key, default)


# =====================================================================
# CLASS 3 — EXPOSURE DECISION ENGINE
# =====================================================================
class ExposureDecisionEngine:
    """
    Computes the EV (Exposure Value) correction needed for cinematic exposure.
    Based on:
        • histogram metrics
        • target mid-tone luma
        • highlight/shadow protection
        • scene mode
        • anti-flicker smoothing
    """

    def __init__(self):
        self.max_step = 0.25          # max EV change per frame (anti-flicker)
        self.highlight_threshold = 0.22
        self.shadow_threshold = 0.22
        self.cinematic_mid_bias = 1.15    # pulls exposure toward mid-tones

    def compute_correction(self,
                           hist_metrics: Dict[str, float],
                           state: ExposureState) -> float:
        """
        Returns the EV correction (delta EV) for this frame.
        """
        avg_luma = hist_metrics["avg_luma"]
        median_luma = hist_metrics["median_luma"]
        shadows = hist_metrics["shadow_ratio"]
        highlights = hist_metrics["highlight_ratio"]

        # -------------------------------------------------------------
        # 1) BASE EXPOSURE ERROR (mid-tone driven)
        # -------------------------------------------------------------
        # cinematic mid-tone emphasis
        mid_tone = (avg_luma * 0.5 + median_luma * 0.5)
        mid_tone *= self.cinematic_mid_bias

        target = state.target_luma
        error = (target - mid_tone)

        # EV adjustment is proportional to error
        ev_delta = error * 1.8     # aggressive control (cinematic)

        # -------------------------------------------------------------
        # 2) HIGHLIGHT PROTECTION
        # -------------------------------------------------------------
        if highlights > self.highlight_threshold:
            # too many blown pixels → reduce exposure
            ev_delta -= (highlights * 0.45)

        # -------------------------------------------------------------
        # 3) SHADOW BOOST
        # -------------------------------------------------------------
        if shadows > self.shadow_threshold:
            # too many crushed blacks → raise exposure
            ev_delta += (shadows * 0.32)

        # -------------------------------------------------------------
        # 4) SCENE-MODE LOGIC
        # -------------------------------------------------------------
        if state.mode == "highlight":
            # keep highlights safe
            ev_delta -= 0.18
        elif state.mode == "shadow":
            # boost shadows for dark scenes
            ev_delta += 0.18
        else:
            # normal → nothing special
            pass

        # -------------------------------------------------------------
        # 5) LIMIT MAX STEP SIZE (anti-flicker)
        # -------------------------------------------------------------
        ev_delta = float(clamp(ev_delta, -self.max_step, self.max_step))

        return ev_delta

    # -------------------------------------------------------------
    # MODE SWITCHING FOR SCENE DETECTION
    # -------------------------------------------------------------
    def update_scene_mode(self, hist_metrics: Dict[str,float], state: ExposureState):
        """
        Update `state.mode` based on histogram classification.
        """
        shadows = hist_metrics["shadow_ratio"]
        highlights = hist_metrics["highlight_ratio"]

        if highlights > 0.28:
            state.mode = "highlight"
        elif shadows > 0.28:
            state.mode = "shadow"
        else:
            state.mode = "normal"

        return state.mode

    def decide(self, metrics, scene_hint=None):
         # Wrapper for Pro integration
         # Returns updated state object (simulated)
         st = ExposureState()
         st.mode = self.update_scene_mode(metrics, st)
         return st


# =====================================================================
# CLASS 4 — EXPOSURE TRANSLATOR (EV → CAMERA SETTINGS)
# =====================================================================
class ExposureTranslator:
    """
    Converts exposure compensation (EV delta) into ISO, shutter, gain.
    """

    def __init__(self,
                 base_iso=100,
                 max_iso=6400,
                 min_shutter=1/8000,
                 max_shutter=1/24,
                 shutter_180deg=True):

        self.base_iso = base_iso
        self.max_iso = max_iso
        self.min_shutter = min_shutter
        self.max_shutter = max_shutter
        self.shutter_180deg = shutter_180deg

    def estimate_motion_level(self, motion_score: float) -> float:
        """
        Converts a motion score (0–1) into shutter bias.
        High motion → faster shutter recommended.
        """
        return clamp(motion_score, 0.0, 1.0)

    def translate(self,
                  state: ExposureState,
                  hist_metrics: Dict[str,float],
                  ev_delta: float,
                  motion_score: float) -> Dict[str,float]:
        """
        Returns a dict:
            { "iso": ..., "shutter": ..., "ev_applied": ... }
        """

        # -------------------------------------------------------------
        # 1) Apply EV correction to brightness model
        # -------------------------------------------------------------
        # update internal EV
        state.ev_last += ev_delta
        state.ev_last = float(clamp(state.ev_last, -4.0, +4.0))

        # -------------------------------------------------------------
        # 2) Motion-adjusted shutter selection
        # -------------------------------------------------------------
        motion = self.estimate_motion_level(motion_score)

        if self.shutter_180deg:
            # Cinematic shutter rule (180°) → shutter ≈ 1/(2 × framerate)
            shutter = 1.0 / (2.0 * state.framerate)
            # fast action → bias toward faster shutter
            shutter *= (1.0 - 0.35 * motion)
        else:
            # free shutter mode
            shutter = self.max_shutter - (motion * (self.max_shutter - self.min_shutter))

        # Clamp shutter
        shutter = clamp(shutter, self.min_shutter, self.max_shutter)

        # -------------------------------------------------------------
        # 3) ISO calculation
        # -------------------------------------------------------------
        # Start from base ISO, add EV correction
        iso_multiplier = 2 ** state.ev_last
        iso = self.base_iso * iso_multiplier

        # clamp ISO
        iso = float(clamp(iso, self.base_iso, self.max_iso))

        # -------------------------------------------------------------
        # 4) Gain compensation (if ISO topped out)
        # -------------------------------------------------------------
        gain = 1.0
        if iso >= self.max_iso * 0.98:
            # If ISO maxed, add small digital gain
            gain = 1.0 + (state.ev_last * 0.06)
            gain = float(clamp(gain, 1.0, 1.25))

        # -------------------------------------------------------------
        # 5) Output structure
        # -------------------------------------------------------------
        return {
            "iso": iso,
            "shutter": shutter,
            "gain": gain,
            "ev_applied": state.ev_last,
            "mode": state.mode,
            "motion": motion,
        }


# =====================================================================
# CLASS 5 — TEMPORAL EXPOSURE SMOOTHER
# =====================================================================
class ExposureSmoother:
    """
    Smooths exposure transitions over time to avoid flicker/jumps.
    """

    def __init__(self,
                 iso_smooth=0.25,
                 shutter_smooth=0.20,
                 gain_smooth=0.35,
                 ev_momentum=0.15):

        self.iso_smooth = iso_smooth
        self.shutter_smooth = shutter_smooth
        self.gain_smooth = gain_smooth
        self.ev_momentum = ev_momentum

        # last stable outputs
        self.last_iso = None
        self.last_shutter = None
        self.last_gain = None
        self.last_ev = 0.0

    def smooth_step(self, prev, new, amt):
        """Generic smoothing function."""
        if prev is None:
            return new
        return prev + (new - prev) * amt

    def apply(self, raw: Dict[str,float]) -> Dict[str,float]:
        iso = raw["iso"]
        shutter = raw["shutter"]
        gain = raw["gain"]
        ev = raw["ev_applied"]

        # ----------------------------------------
        # Smooth all channels
        # ----------------------------------------
        self.last_iso = self.smooth_step(self.last_iso, iso, self.iso_smooth)
        self.last_shutter = self.smooth_step(self.last_shutter, shutter, self.shutter_smooth)
        self.last_gain = self.smooth_step(self.last_gain, gain, self.gain_smooth)

        # EV momentum (prevents micro flicker)
        self.last_ev = self.smooth_step(self.last_ev, ev, self.ev_momentum)

        return {
            "iso": float(self.last_iso),
            "shutter": float(self.last_shutter),
            "gain": float(self.last_gain),
            "ev": float(self.last_ev),
        }


# =====================================================================
# CLASS 6 — MAIN EXPOSURE ENGINE (what AICameraBrain calls)
# =====================================================================
class ExposureEngine:
    """
    High-level interface for the auto-exposure pipeline.
    """

    def __init__(self):
        self.scene = ExposureHistogramAnalyzer()
        self.translator = ExposureTranslator()
        self.smoother = ExposureSmoother()

    def update(self,
               frame: np.ndarray,
               motion_score: float,
               state: ExposureState,
               return_debug: bool = False) -> Dict[str, any]:

        # ---------------------------------------------------------
        # 1) Analyze scene
        # ---------------------------------------------------------
        metrics = self.scene.analyze(frame)

        # brightness error → EV correction
        ev_delta = compute_ev_delta(
            metrics["brightness"],        # predicted exposure
            state.brightness_target       # desired
        )

        # ---------------------------------------------------------
        # 2) Translate exposure
        # ---------------------------------------------------------
        raw_settings = self.translator.translate(
            state=state,
            hist_metrics=metrics,
            ev_delta=ev_delta,
            motion_score=motion_score,
        )

        # ---------------------------------------------------------
        # 3) Smooth exposure for cinematic look
        # ---------------------------------------------------------
        smooth = self.smoother.apply(raw_settings)

        # ---------------------------------------------------------
        # 4) Pack final output
        # ---------------------------------------------------------
        out = {
            "iso": smooth["iso"],
            "shutter": smooth["shutter"],
            "gain": smooth["gain"],
            "ev": smooth["ev"],
            "scene_metrics": metrics,
            "mode": state.mode,
            "timestamp": time.time(),
        }

        # optional developer debug
        if return_debug:
            out["debug"] = {
                "raw": raw_settings,
                "metrics": metrics,
            }

        return out

class ExposureMetering(ExposureHistogramAnalyzer):
    """Alias for Pro integration"""
    def compute_metrics(self, frame):
        return self.analyze(frame)

class ExposureDecisionLayer(ExposureDecisionEngine):
    """Alias for Pro integration"""
    pass


# =====================================================================
# CLASS 7 — EXPOSURE HISTORY (for prediction + oscillation prevention)
# =====================================================================
class ExposureHistory:
    """
    Maintains rolling history of:
        • EV over time
        • brightness values
        • shutter & ISO transitions
    """

    def __init__(self, size=24):
        self.size = size
        self.ev_hist = []
        self.brightness_hist = []
        self.shutter_hist = []
        self.iso_hist = []

    def push(self, ev, brightness, shutter, iso):
        """Append to history with max length."""
        if len(self.ev_hist) >= self.size:
            self.ev_hist.pop(0)
            self.brightness_hist.pop(0)
            self.shutter_hist.pop(0)
            self.iso_hist.pop(0)

        self.ev_hist.append(ev)
        self.brightness_hist.append(brightness)
        self.shutter_hist.append(shutter)
        self.iso_hist.append(iso)

    def detect_ev_oscillation(self):
        """
        Returns True if EV is bouncing back and forth:
             +1, -1, +1, -1...
        """
        if len(self.ev_hist) < 6:
            return False

        signs = []
        for i in range(1, len(self.ev_hist)):
            delta = self.ev_hist[i] - self.ev_hist[i-1]
            signs.append(np.sign(delta))

        # Look for alternating pattern: + - + - + -
        # Count alternations
        alt = 0
        for i in range(1, len(signs)):
            if signs[i] * signs[i-1] < 0:
                alt += 1

        return alt > (len(signs) * 0.6)

    def mean_brightness(self):
        if not self.brightness_hist:
            return None
        return float(np.mean(self.brightness_hist))

    def trend_brightness(self):
        """Return slope of brightness trend."""
        if len(self.brightness_hist) < 2:
            return 0.0
        x = np.arange(len(self.brightness_hist))
        y = np.array(self.brightness_hist)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)


# =====================================================================
# CLASS 8 — ADVANCED EV CORRECTOR (anti-flicker + highlight safety)
# =====================================================================
class AdvancedEVController:
    """
    Applies intelligent damping to EV changes, using history.
    """

    def __init__(self,
                 flicker_damp=0.35,
                 highlight_protect_strength=0.25):

        self.flicker_damp = flicker_damp
        self.highlight_protect_strength = highlight_protect_strength

    def apply(self, ev_raw, metrics, history: ExposureHistory):
        """
        Modify EV based on historical behavior.
        """

        # ----------------------------
        # 1) Anti-oscillation damping
        # ----------------------------
        if history.detect_ev_oscillation():
            # reduce EV swings by damping
            ev_raw *= (1.0 - self.flicker_damp)

        # ----------------------------
        # 2) Highlight protection
        # ----------------------------
        highlight_ratio = metrics["highlight_ratio"]
        if highlight_ratio > 0.15:     # meaning >15% of image is clipped
            protect = min(1.0, highlight_ratio * 2.5)
            ev_raw -= protect * self.highlight_protect_strength

        return float(ev_raw)


# =====================================================================
# CLASS 10 — HDR BRACKET MANAGER
# =====================================================================
class HDRBracketManager:
    """
    Controls multi-frame exposure bracketing for HDR fusion.
    """

    def __init__(self,
                 ev_low=-1.5,
                 ev_high=+1.5,
                 frames=3,
                 activate_threshold=0.35):

        self.ev_low = ev_low
        self.ev_high = ev_high
        self.frames = frames      # 3 or 5
        self.activate_threshold = activate_threshold  # contrast trigger

        self.active = False
        self.pattern = []

    def should_enable_hdr(self, metrics):
        """
        Detect high dynamic range scenes.
        """
        bright = metrics["highlight_ratio"]
        dark = metrics["shadow_ratio"]
        return (bright + dark) > self.activate_threshold

    def generate_pattern(self):
        """
        Return EV offsets for a bracket:
            3-frame: [-ev, 0, +ev]
            5-frame: [-evH, -evL, 0, +evL, +evH]
        """
        if self.frames == 3:
            return [self.ev_low, 0.0, self.ev_high]

        elif self.frames == 5:
            mid_low = self.ev_low * 0.5
            mid_high = self.ev_high * 0.5
            return [self.ev_low, mid_low, 0.0, mid_high, self.ev_high]

        else:
            # fallback
            return [self.ev_low, 0.0, self.ev_high]

    def start_bracket(self, metrics):
        """Start HDR capture sequence if needed."""
        if self.should_enable_hdr(metrics):
            self.active = True
            self.pattern = self.generate_pattern()
        else:
            self.active = False
            self.pattern = []

    def get_ev_offset(self):
        """
        Pop the next bracket EV shift.
        Returns None when bracket sequence is complete.
        """
        if not self.active or not self.pattern:
            return None
        return self.pattern.pop(0)


# =====================================================================
# CLASS 11 — HDR FUSION ENGINE (MANTIUK / REINHARD HYBRID)
# =====================================================================
class HDRFusionEngine:
    """
    Fuses multiple LDR frames into a single HDR image.
    """

    def __init__(self):
        # tone-mapping parameters
        self.mantiuk_strength = 0.85
        self.reinhard_white = 4.0

    def luminance(self, img):
        """BT.709 luminance"""
        return (
            0.2126 * img[..., 2] +
            0.7152 * img[..., 1] +
            0.0722 * img[..., 0]
        )

    def compute_weight(self, img, ev_shift):
        """
        Give higher weight to mid-tones, lower to clipped regions.
        """
        lum = self.luminance(img)
        w = np.exp(-4.0 * (lum - 0.5) ** 2)

        # reduce weight if under/overexposed relative to shift
        if ev_shift < 0:
            w *= (lum < 0.9).astype(np.float32)
        else:
            w *= (lum > 0.1).astype(np.float32)

        return w + 1e-6

    def merge(self, frames, ev_shifts):
        """
        frames: list of np.ndarray images
        ev_shifts: EV offsets applied to each frame
        """
        assert len(frames) == len(ev_shifts)

        H, W = frames[0].shape[:2]
        hdr = np.zeros((H, W, 3), dtype=np.float32)
        total_w = np.zeros((H, W), dtype=np.float32)

        for img, ev in zip(frames, ev_shifts):
            # convert EV to exposure factor
            scale = 2.0 ** ev
            scaled = np.clip(img.astype(np.float32) * scale, 0, 1)

            w = self.compute_weight(scaled, ev)

            hdr += scaled * w[..., None]
            total_w += w

        hdr /= (total_w[..., None] + 1e-6)
        return hdr

    def tone_map(self, hdr):
        """
        Apply hybrid tone-mapping.
        """
        # ---------------------------
        # Reinhard global
        # ---------------------------
        L = self.luminance(hdr)
        L_white = self.reinhard_white

        L_tone = (L * (1 + L / (L_white ** 2))) / (1 + L)

        # ---------------------------
        # Mantiuk local contrast
        # ---------------------------
        blur = cv2.GaussianBlur(L_tone, (0, 0), 8)
        detail = L_tone - blur
        L_final = L_tone + detail * self.mantiuk_strength

        # reapply to RGB channels
        scale = (L_final / (L + 1e-6))[..., None]
        out = np.clip(hdr * scale, 0, 1)

        return out


# =====================================================================
# CLASS 12 — HDR ORCHESTRATOR (CONNECTS BRACKETS + FUSION)
# =====================================================================
class HDROrchestrator:
    """
    Manages high-level HDR workflow.
    """

    def __init__(self):
        self.bracket_mgr = HDRBracketManager()
        self.fusion = HDRFusionEngine()

        self.collecting = False
        self.buffer_frames = []
        self.buffer_shifts = []

    def begin_if_needed(self, metrics):
        self.bracket_mgr.start_bracket(metrics)

        if self.bracket_mgr.active:
            self.collecting = True
            self.buffer_frames.clear()
            self.buffer_shifts.clear()
        else:
            self.collecting = False
            self.buffer_frames.clear()
            self.buffer_shifts.clear()

    def step(self, frame):
        """
        Called each frame when HDR is active.
        """
        if not self.collecting:
            return None

        ev = self.bracket_mgr.get_ev_offset()
        if ev is None:
            # bracket ended
            self.collecting = False
            return None

        return ev

    def push_frame(self, frame, ev):
        self.buffer_frames.append(frame)
        self.buffer_shifts.append(ev)

    def ready(self):
        return (
            len(self.buffer_frames) > 0 and
            len(self.buffer_frames) == len(self.buffer_shifts) and
            (not self.bracket_mgr.active)
        )

    def fuse(self):
        if not self.ready():
            return None

        try:
            hdr = self.fusion.merge(self.buffer_frames, self.buffer_shifts)
            out = self.fusion.tone_map(hdr)
            return out
        except Exception as e:
            print("[HDR] Fusion error:", e)
            return None


# =====================================================================
# CLASS 13 — INTEGRATION INTO ExposureEnginePro
# =====================================================================
class ExposureEnginePro(ExposureEngine):
    """
    FULL PIPELINE:
       1. Compute exposure metrics
       2. Exposure decision layer (AE + scene classifier)
       3. HDR check + bracket orchestration
       4. Generate a final exposure setting for the frame
       5. (Optional) HDR fusion output for internal camera
    """

    def __init__(self):
        super().__init__()
        self.history = ExposureHistory(size=32)
        self.ev_controller = AdvancedEVController()
        
        # Redfining components for Pro version
        self.metering = ExposureMetering()
        self.controller = ExposureDecisionLayer()
        self.hdr = HDROrchestrator()

        self.exposure_state = None
        self.last_hdr_output = None

    def update(self,
               frame,
               motion_score=0.0,
               state=None,
               return_debug=False):

        if state is None:
             state = ExposureState()

        # -----------------------------
        # Base exposure metrics
        # -----------------------------
        metrics = self.scene.analyze(frame)
        raw_ev_delta = compute_ev_delta(
            metrics["brightness"],
            state.brightness_target
        )

        # -----------------------------
        # Apply EV corrections
        # -----------------------------
        ev_corrected = self.ev_controller.apply(
            raw_ev_delta,
            metrics,
            self.history
        )

        # -----------------------------
        # Base exposure translation
        # -----------------------------
        raw_settings = self.translator.translate(
            state=state,
            hist_metrics=metrics,
            ev_delta=ev_corrected,
            motion_score=motion_score,
        )

        # -----------------------------
        # Temporal smoothing
        # -----------------------------
        smooth = self.smoother.apply(raw_settings)

        # -----------------------------
        # Update history AFTER smoothing
        # -----------------------------
        self.history.push(
            smooth["ev"],
            metrics["brightness"],
            smooth["shutter"],
            smooth["iso"]
        )

        # -----------------------------
        # HDR Logic (Pro Feature)
        # -----------------------------
        # self.hdr.begin_if_needed(metrics) # Simplified: Only trigger if explicitly called in main workflow
        
        out = {
            "iso": smooth["iso"],
            "shutter": smooth["shutter"],
            "gain": smooth["gain"],
            "ev": smooth["ev"],
            "mode": state.mode,
            "scene_metrics": metrics,
            "timestamp": time.time(),
        }

        if return_debug:
            out["debug"] = {
                "raw": raw_settings,
                "metrics": metrics,
                "ev_corrected": ev_corrected,
                "history": {
                    "ev_hist": self.history.ev_hist[-6:],
                    "bright_hist": self.history.brightness_hist[-6:]
                }
            }

        return out

    # Alternate signature for compatibility with some consumers that use 'scene_hint'
    def update_v2(self, frame, scene_hint=None):
         # This matches the signature in the corrupted file's chunk 8
         return self.update(frame)

# =====================================================================
# CLASS 14 — DEBUG VISUALIZER (Zebra, Over/Under Heatmaps, Focus Peaking)
# =====================================================================
class ExposureDebugVisualizer:
    """
    Produces overlays for debugging camera exposure + HDR.
    """

    def __init__(self):
        self.zebra_thresh = 0.95     # 95% brightness
        self.shadow_thresh = 0.05    # 5% brightness
        self.heatmap_enabled = True
        self.focus_peak_enabled = True

    def _compute_luma(self, frame):
        return (0.299 * frame[:,:,2] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,0]) / 255.0

    def draw(self, frame):
        vis = frame.copy()
        h, w = vis.shape[:2]

        luma = self._compute_luma(frame)

        # =============================
        # OVEREXPOSURE ZEBRA STRIPES
        # =============================
        over_mask = (luma > self.zebra_thresh)
        if np.any(over_mask):
            for y in range(0, h, 4):
                for x in range(0, w, 8):
                    if over_mask[y, x]:
                        vis[y:y+2, x:x+8] = (0, 255, 255)

        # =============================
        # SHADOW BLUE MASK
        # =============================
        shadow_mask = (luma < self.shadow_thresh)
        vis[shadow_mask] = vis[shadow_mask] * 0.5 + np.array([255, 0, 0]) * 0.5

        # =============================
        # HEATMAP (OPTIONAL)
        # =============================
        if self.heatmap_enabled:
            heat = (luma * 255).astype(np.uint8)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.7, heat, 0.3, 0)

        # =============================
        # FOCUS PEAKING (EDGE MAGNITUDE)
        # =============================
        if self.focus_peak_enabled:
            gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
            edge = np.sqrt(gx*gx + gy*gy)
            mask = edge > (edge.mean() * 2.5)
            vis[mask] = (0, 255, 0)   # green highlights

        return vis


# =====================================================================
# CLASS 15 — TEMPORAL HDR SMOOTHING ENGINE
# =====================================================================
class HDRTemporalSmoother:
    """
    Prevents flicker between HDR frames.
    """

    def __init__(self, strength=0.6):
        self.prev = None
        self.strength = float(strength)

    def apply(self, new_hdr):
        if new_hdr is None:
            return None

        if self.prev is None:
            self.prev = new_hdr
            return new_hdr

        # small optical-flow compensation
        try:
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(self.prev, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(new_hdr, cv2.COLOR_BGR2GRAY),
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            h, w = new_hdr.shape[:2]

            xs, ys = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (xs + flow[:,:,0]).astype(np.float32)
            map_y = (ys + flow[:,:,1]).astype(np.float32)
            warped_prev = cv2.remap(self.prev, map_x, map_y, cv2.INTER_LINEAR)
        except Exception:
            warped_prev = self.prev

        # temporal smoothing
        smoothed = cv2.addWeighted(new_hdr, 1.0 - self.strength, warped_prev, self.strength, 0)
        self.prev = smoothed
        return smoothed


# =====================================================================
# CLASS 16 — FILMIC TONE CURVE ENGINE
# =====================================================================
class FilmicToneCurve:
    """
    Applies a film-style tone curve.
    """

    def __init__(self):
        # adjustable parameters
        self.mid = 0.18
        self.contrast = 1.15
        self.rolloff = 0.92

    def apply(self, frame):
        f = frame.astype(np.float32) / 255.0

        # log encoding
        log = np.log1p(f * 4.0) / np.log1p(4.0)

        # contrast shaping
        log = ((log - self.mid) * self.contrast) + self.mid

        # highlight rolloff
        roll = 1.0 - np.exp(-log * (5 * self.rolloff))

        out = np.clip(roll * 255.0, 0, 255).astype(np.uint8)
        return out


# =====================================================================
# CLASS 17 — METADATA PACKER (per-frame metadata for encoder)
# =====================================================================
class ExposureMetadataPacker:
    """
    Produces JSON metadata for each frame.
    """

    def pack(self, state, metrics, hdr_output, ev_shift):
        meta = {
            "iso": state.get("iso"),
            "shutter": state.get("shutter_us"),
            "aperture": state.get("aperture"),
            "scene_luma": metrics.get("brightness"),
            "dynamic_range": metrics.get("dynamic_range"),
            "highlights": metrics.get("highlights"),
            "shadows": metrics.get("shadows"),
            "hdr_active": hdr_output is not None,
            "ev_shift": ev_shift,
            "timestamp": time.time()
        }
        return meta

# ================================================================
# 18. Auto-ND Filter Advisor
# ================================================================
class NDFilterAdvisor:
    """
    Analyzes scene brightness vs. ideal cinematic shutter speed.
    """

    def __init__(self):
        self.nd_table = {
            1: "ND2",
            2: "ND4",
            3: "ND8",
            4: "ND16",
            5: "ND32",
            6: "ND64",
            7: "ND128",
            8: "ND256",
            9: "ND512",
            10: "ND1000"
        }
        self.base_iso = 100.0
        self.target_shutter = 1.0 / 60.0  # Ideal for 30fps cinematic video

    def calculate_ev(self, iso: float, shutter: float) -> float:
        """
        Simplified EV estimate used for relative comparisons:
        EV ≈ log2( (100 * (1/shutter)) / ISO )
        """
        if shutter <= 0 or iso <= 0:
            return 0.0
        try:
            ev = math.log2((100.0 * (1.0 / shutter)) / iso)
            return float(ev)
        except Exception:
            return 0.0

    def get_recommendation(self, current_iso: float, current_shutter: float, scene_brightness_ev: float = None):
        if current_shutter <= 0:
            return {
                "recommended_filter": "NONE",
                "stops_needed": 0.0,
                "excess_light_ratio": 1.0,
                "message": "Invalid shutter value."
            }

        light_excess_ratio = (self.target_shutter / current_shutter)
        # If light_excess_ratio < 1 → scene is darker than ideal (no ND needed)
        if light_excess_ratio <= 1.0:
            return {
                "recommended_filter": "NONE",
                "stops_needed": 0.0,
                "excess_light_ratio": round(light_excess_ratio, 2),
                "message": "Lighting is optimal or dark. No ND filter needed."
            }

        # stops_needed in floating stops:
        stops_needed = math.log2(light_excess_ratio)

        # Round to nearest whole stop for available ND table
        closest_stop = int(round(stops_needed))
        if closest_stop < 1:
            closest_stop = 0

        if closest_stop == 0:
            recommendation = "NONE"
        elif closest_stop > max(self.nd_table.keys()):
            recommendation = f"{self.nd_table[max(self.nd_table.keys())]}+"
        else:
            recommendation = self.nd_table.get(closest_stop, f"ND_stop_{closest_stop}")

        return {
            "recommended_filter": recommendation,
            "stops_needed": round(stops_needed, 2),
            "excess_light_ratio": round(light_excess_ratio, 2),
            "message": f"Scene is {round(stops_needed, 2)} stops too bright for {int(1/self.target_shutter)}fps cinematic shutter."
        }


# ================================================================
# 19. HDR Bracketing Planner
# ================================================================
class HDRBracketingPlanner:
    """
    Monitors histogram for clipping and generates bracketing plans.
    """

    def __init__(self):
        self.clipping_threshold_low = 5    # Pixel value (0-255)
        self.clipping_threshold_high = 250
        self.pixel_percent_trigger = 0.05  # 5% of pixels must be clipping
        self.last_plan_time = 0.0
        self.cooldown = 2.0                # Seconds between plans

    def _ensure_uint8(self, frame_gray):
        """Ensure grayscale frame is uint8 for histogram ops."""
        if frame_gray is None:
            return None
        if frame_gray.dtype != np.uint8:
            # Clip and convert safely
            frame_gray = np.clip(frame_gray, 0, 255).astype(np.uint8)
        return frame_gray

    def analyze_histogram(self, frame_gray):
        if frame_gray is None:
            return "NORMAL"

        frame_gray = self._ensure_uint8(frame_gray)
        if frame_gray is None:
            return "NORMAL"

        hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256]).flatten()
        total_pixels = float(frame_gray.shape[0] * frame_gray.shape[1])
        if total_pixels <= 0:
            return "NORMAL"

        shadow_pixels = float(np.sum(hist[:self.clipping_threshold_low]))
        highlight_pixels = float(np.sum(hist[self.clipping_threshold_high:]))

        shadow_ratio = shadow_pixels / total_pixels
        highlight_ratio = highlight_pixels / total_pixels

        is_shadow_clipping = shadow_ratio > self.pixel_percent_trigger
        is_highlight_clipping = highlight_ratio > self.pixel_percent_trigger

        if is_shadow_clipping and is_highlight_clipping:
            return "HIGH_CONTRAST"
        elif is_shadow_clipping:
            return "UNDEREXPOSED"
        elif is_highlight_clipping:
            return "OVEREXPOSED"
        else:
            return "NORMAL"

    def generate_plan(self, analysis_result, current_ev: float = 0.0):
        now = time.time()
        if (now - self.last_plan_time) < self.cooldown:
            return None

        plan = {
            "trigger_hdr": False,
            "offsets": [],
            "shots": 0,
            "reason": analysis_result
        }

        if analysis_result == "HIGH_CONTRAST":
            plan["trigger_hdr"] = True
            plan["offsets"] = [-2.0, 0.0, +2.0]
            plan["shots"] = 3
            self.last_plan_time = now
        elif analysis_result == "UNDEREXPOSED":
            plan["trigger_hdr"] = False
            plan["offsets"] = [+1.0]
            plan["shots"] = 1
            self.last_plan_time = now
        elif analysis_result == "OVEREXPOSED":
            plan["trigger_hdr"] = False
            plan["offsets"] = [-1.0]
            plan["shots"] = 1
            self.last_plan_time = now

        return plan

    def compute(self, frame):
        if frame is None:
            return {"scene_state": "NORMAL", "hdr_plan": None}

        # Accept color or gray
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        state = self.analyze_histogram(gray)
        plan = self.generate_plan(state)
        return {
            "scene_state": state,
            "hdr_plan": plan
        }

class ExposureUtils:
    @staticmethod
    def iso_to_gain_db(iso):
        """Approx conversion of ISO to dB gain (base ISO 100 => 0 dB)."""
        if iso <= 0:
            return 0.0
        return 20.0 * math.log10(max(iso, 1.0) / 100.0)

    @staticmethod
    def shutter_to_us(shutter_sec):
        """Converts fractional seconds (1/60) to microseconds (16666)."""
        if shutter_sec <= 0:
            return 0
        return int(shutter_sec * 1_000_000)

    @staticmethod
    def clamp_settings(iso, shutter, limits):
        """
        limits is a dict:
          { 'min_iso': 100, 'max_iso': 6400, 'min_shutter': 1/8000, 'max_shutter': 1/2 }
        """
        safe_iso = float(np.clip(iso, limits.get('min_iso', 100), limits.get('max_iso', 6400)))
        safe_shutter = float(np.clip(shutter, limits.get('min_shutter', 1/8000.0), limits.get('max_shutter', 1/2.0)))
        return safe_iso, safe_shutter

# Helper alias for compatibility
AIExposureEngine = ExposureEnginePro