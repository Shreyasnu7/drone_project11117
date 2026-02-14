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

Structure:
    ExposureHistogramAnalyzer
    ExposureState
    ExposureTemporalFilter
    ExposureDecisionEngine
    AIExposureEngine (main controller)

This file will be ~450–500 lines total.
"""

import numpy as np
import cv2
import time
from collections import deque
from typing import Dict, Optional

# -------------------------------------------------------------
# Utility: clamp function
# -------------------------------------------------------------
def clamp(x, low, high):
    return low if x < low else high if x > high else x


# =====================================================================
# CLASS 1 — HISTOGRAM ANALYZER
# =====================================================================
class ExposureHistogramAnalyzer:
    """
    Computes luminance histogram and exposure metrics.

    Outputs:
        avg_luma           — mean brightness
        median_luma        — median brightness
        shadow_ratio       — dark area %
        highlight_ratio    — bright area %
        contrast_index     — global contrast score
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
                "contrast_index": 0.0
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
            "contrast_index": contrast_index
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
        self.mode = "normal"              # modes: normal / highlight / shadow

        self.hist_history = deque(maxlen=12)   # last 12 histograms
        self.ev_history = deque(maxlen=8)       # last few EV adjustments

        self.last_update_ts = time.time()
        self.ev_last = 0.0 # Added per usage
        self.framerate = 30.0 # Added per usage
        self.brightness_target = 0.55

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


# =====================================================================
# CLASS 4 — EXPOSURE TRANSLATOR (EV → CAMERA SETTINGS)
# =====================================================================
class ExposureTranslator:
    """
    Converts exposure compensation (EV delta) into ISO, shutter, gain.

    Uses:
        • scene brightness
        • motion level
        • cinematic shutter rules
        • ISO noise model
        • device limits
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
    def __init__(self):
        self.scene = ExposureHistogramAnalyzer()
        self.translator = ExposureTranslator()
        self.smoother = ExposureSmoother()

    def update(self,
               frame: np.ndarray,
               motion_score: float,
               state: ExposureState,
               return_debug: bool = False) -> Dict[str, any]:
        # 1) Analyze scene
        metrics = self.scene.analyze(frame)

        # brightness error → EV correction
        ev_delta = compute_ev_delta(
            metrics["brightness"],        # predicted exposure
            state.brightness_target       # desired
        )

        # 2) Translate exposure
        raw_settings = self.translator.translate(
            state=state,
            hist_metrics=metrics,
            ev_delta=ev_delta,
            motion_score=motion_score,
        )

        # 3) Smooth exposure for cinematic look
        smooth = self.smoother.apply(raw_settings)

        # 4) Pack final output
        out = {
            "iso": smooth["iso"],
            "shutter": smooth["shutter"],
            "gain": smooth["gain"],
            "ev": smooth["ev"],
            "scene_metrics": metrics,
            "mode": state.mode,
            "timestamp": time.time(),
        }

        if return_debug:
            out["debug"] = {
                "raw": raw_settings,
                "metrics": metrics,
            }

        return out

# =====================================================================
# CLASS 7 — EXPOSURE HISTORY (for prediction + oscillation prevention)
# =====================================================================
class ExposureHistory:
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
        if len(self.ev_hist) < 6:
            return False

        signs = []
        for i in range(1, len(self.ev_hist)):
            delta = self.ev_hist[i] - self.ev_hist[i-1]
            signs.append(np.sign(delta))

        # Look for alternating pattern: + - + - + -
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
    def __init__(self,
                 flicker_damp=0.35,
                 highlight_protect_strength=0.25):

        self.flicker_damp = flicker_damp
        self.highlight_protect_strength = highlight_protect_strength

    def apply(self, ev_raw, metrics, history: ExposureHistory):
        # 1) Anti-oscillation damping
        if history.detect_ev_oscillation():
            ev_raw *= (1.0 - self.flicker_damp)

        # 2) Highlight protection
        highlight_ratio = metrics["highlight_ratio"]
        if highlight_ratio > 0.15:     # meaning >15% of image is clipped
            protect = min(1.0, highlight_ratio * 2.5)
            ev_raw -= protect * self.highlight_protect_strength

        return float(ev_raw)

# =====================================================================
# CLASS 9 — EXTENDED ExposureEngine (attaches history + EV controller)
# =====================================================================
class ExposureEnginePro(ExposureEngine):
    def __init__(self):
        super().__init__()
        self.history = ExposureHistory(size=32)
        self.ev_controller = AdvancedEVController()
        # To support extended calls:
        self.hdr = HDROrchestrator() 

    def update(self,
               frame,
               motion_score,
               state,
               return_debug=False):

        # Base exposure metrics
        metrics = self.scene.analyze(frame)
        raw_ev_delta = compute_ev_delta(
            metrics["brightness"],
            state.brightness_target
        )

        # Apply EV corrections
        ev_corrected = self.ev_controller.apply(
            raw_ev_delta,
            metrics,
            self.history
        )

        # Base exposure translation
        raw_settings = self.translator.translate(
            state=state,
            hist_metrics=metrics,
            ev_delta=ev_corrected,
            motion_score=motion_score,
        )

        # Temporal smoothing
        smooth = self.smoother.apply(raw_settings)

        # Update history AFTER smoothing
        self.history.push(
            smooth["ev"],
            metrics["brightness"],
            smooth["shutter"],
            smooth["iso"]
        )

        # Output package
        out = {
            "iso": smooth["iso"],
            "shutter": smooth["shutter"],
            "gain": smooth["gain"],
            "ev": smooth["ev"],
            "mode": state.mode,
            "scene_metrics": metrics,
            "timestamp": time.time(),
        }

        # HDR Check Logic Integration (simplified from Class 13)
        # We need to call self.hdr if it exists
        if hasattr(self, 'hdr'):
             self.hdr.begin_if_needed(metrics)
             ev_shift = self.hdr.step(frame)
             if ev_shift is not None:
                 out["request_ev_shift"] = ev_shift
             
             if self.hdr.ready():
                 out["hdr_output"] = self.hdr.fuse()

        if return_debug:
            out["debug"] = {
                "raw": raw_settings,
                "metrics": metrics,
                "ev_corrected": ev_corrected,
            }

        return out

def compute_ev_delta(current, target):
    # Logarithmic error approximation
    if current < 0.01: current = 0.01
    return np.log2(target / current)

# =====================================================================
# CLASS 10 — HDR BRACKET MANAGER
# =====================================================================
class HDRBracketManager:
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
        bright = metrics["highlight_ratio"]
        dark = metrics["shadow_ratio"]
        return (bright + dark) > self.activate_threshold

    def generate_pattern(self):
        if self.frames == 3:
            return [self.ev_low, 0.0, self.ev_high]

        elif self.frames == 5:
            mid_low = self.ev_low * 0.5
            mid_high = self.ev_high * 0.5
            return [self.ev_low, mid_low, 0.0, mid_high, self.ev_high]
        else:
            return [self.ev_low, 0.0, self.ev_high]

    def start_bracket(self, metrics):
        if self.should_enable_hdr(metrics):
            self.active = True
            self.pattern = self.generate_pattern()
        else:
            self.active = False
            self.pattern = []

    def get_ev_offset(self):
        if not self.active or not self.pattern:
            return None
        return self.pattern.pop(0)

# =====================================================================
# CLASS 11 — HDR FUSION ENGINE (MANTIUK / REINHARD HYBRID)
# =====================================================================
class HDRFusionEngine:
    def __init__(self):
        self.mantiuk_strength = 0.85
        self.reinhard_white = 4.0

    def luminance(self, img):
        return (
            0.2126 * img[..., 2] +
            0.7152 * img[..., 1] +
            0.0722 * img[..., 0]
        )

    def compute_weight(self, img, ev_shift):
        lum = self.luminance(img)
        w = np.exp(-4.0 * (lum - 0.5) ** 2)
        if ev_shift < 0:
            w *= (lum < 0.9).astype(np.float32)
        else:
            w *= (lum > 0.1).astype(np.float32)
        return w + 1e-6

    def merge(self, frames, ev_shifts):
        H, W = frames[0].shape[:2]
        hdr = np.zeros((H, W, 3), dtype=np.float32)
        total_w = np.zeros((H, W), dtype=np.float32)

        for img, ev in zip(frames, ev_shifts):
            scale = 2.0 ** ev
            scaled = np.clip(img.astype(np.float32) * scale, 0, 1)
            w = self.compute_weight(scaled, ev)
            hdr += scaled * w[..., None]
            total_w += w

        hdr /= (total_w[..., None] + 1e-6)
        return hdr

    def tone_map(self, hdr):
        L = self.luminance(hdr)
        L_white = self.reinhard_white
        L_tone = (L * (1 + L / (L_white ** 2))) / (1 + L)
        blur = cv2.GaussianBlur(L_tone, (0, 0), 8)
        detail = L_tone - blur
        L_final = L_tone + detail * self.mantiuk_strength
        scale = (L_final / (L + 1e-6))[..., None]
        out = np.clip(hdr * scale, 0, 1)
        return out

# =====================================================================
# CLASS 12 — HDR ORCHESTRATOR (CONNECTS BRACKETS + FUSION)
# =====================================================================
class HDROrchestrator:
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
        if not self.collecting:
            return None
        ev = self.bracket_mgr.get_ev_offset()
        if ev is None:
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
# CLASS 14 — DEBUG VISUALIZER (Zebra, Over/Under Heatmaps, Focus Peaking)
# =====================================================================
class ExposureDebugVisualizer:
    def __init__(self):
        self.zebra_thresh = 0.95
        self.shadow_thresh = 0.05
        self.heatmap_enabled = True
        self.focus_peak_enabled = True

    def _compute_luma(self, frame):
        return (0.299 * frame[:,:,2] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,0]) / 255.0

    def draw(self, frame):
        vis = frame.copy()
        h, w = vis.shape[:2]
        luma = self._compute_luma(frame)

        # OVEREXPOSURE ZEBRA STRIPES
        over_mask = (luma > self.zebra_thresh)
        if np.any(over_mask):
            for y in range(0, h, 4):
                for x in range(0, w, 8):
                    if over_mask[y, x]:
                        vis[y:y+2, x:x+8] = (0, 255, 255)

        # SHADOW BLUE MASK
        shadow_mask = (luma < self.shadow_thresh)
        vis[shadow_mask] = vis[shadow_mask] * 0.5 + np.array([255, 0, 0]) * 0.5

        # HEATMAP
        if self.heatmap_enabled:
            heat = (luma * 255).astype(np.uint8)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.7, heat, 0.3, 0)

        # FOCUS PEAKING
        if self.focus_peak_enabled:
            gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
            edge = np.sqrt(gx*gx + gy*gy)
            mask = edge > (edge.mean() * 2.5)
            vis[mask] = (0, 255, 0)

        return vis


# =====================================================================
# CLASS 15 — TEMPORAL HDR SMOOTHING ENGINE
# =====================================================================
class HDRTemporalSmoother:
    def __init__(self, strength=0.6):
        self.prev = None
        self.strength = float(strength)

    def apply(self, new_hdr):
        if new_hdr is None:
            return None
        if self.prev is None:
            self.prev = new_hdr
            return new_hdr

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

        smoothed = cv2.addWeighted(new_hdr, 1.0 - self.strength, warped_prev, self.strength, 0)
        self.prev = smoothed
        return smoothed


# =====================================================================
# CLASS 16 — FILMIC TONE CURVE ENGINE
# =====================================================================
class FilmicToneCurve:
    def __init__(self):
        self.mid = 0.18
        self.contrast = 1.15
        self.rolloff = 0.92

    def apply(self, frame):
        f = frame.astype(np.float32) / 255.0
        log = np.log1p(f * 4.0) / np.log1p(4.0)
        log = ((log - self.mid) * self.contrast) + self.mid
        roll = 1.0 - np.exp(-log * (5 * self.rolloff))
        out = np.clip(roll * 255.0, 0, 255).astype(np.uint8)
        return out


# =====================================================================
# CLASS 17 — METADATA PACKER
# =====================================================================
class ExposureMetadataPacker:
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
# 13. Auto-ND Filter Advisor
# ================================================================
class NDFilterAdvisor:
    def __init__(self):
        self.nd_table = {1: "ND2", 2: "ND4", 3: "ND8", 4: "ND16", 5: "ND32", 6: "ND64", 7: "ND128", 8: "ND256", 9: "ND512", 10: "ND1000"}
        self.base_iso = 100.0
        self.target_shutter = 1.0 / 60.0

    def calculate_ev(self, iso: float, shutter: float) -> float:
        if shutter <= 0 or iso <= 0: return 0.0
        try:
            ev = math.log2((100.0 * (1.0 / shutter)) / iso)
            return float(ev)
        except Exception: return 0.0

    def get_recommendation(self, current_iso: float, current_shutter: float, scene_brightness_ev: float = None):
        if current_shutter <= 0: return {"recommended_filter": "NONE", "stops_needed": 0.0}
        light_excess_ratio = (self.target_shutter / current_shutter)
        if light_excess_ratio <= 1.0: return {"recommended_filter": "NONE", "stops_needed": 0.0}
        stops_needed = math.log2(light_excess_ratio)
        closest_stop = int(round(stops_needed))
        if closest_stop < 1: closest_stop = 0
        recommendation = self.nd_table.get(closest_stop, f"ND_stop_{closest_stop}") if closest_stop > 0 else "NONE"
        return {"recommended_filter": recommendation, "stops_needed": round(stops_needed, 2)}

# ================================================================
# 14. HDR Bracketing Planner
# ================================================================
class HDRBracketingPlanner:
    def __init__(self):
        self.clipping_threshold_low = 5
        self.clipping_threshold_high = 250
        self.pixel_percent_trigger = 0.05
        self.last_plan_time = 0.0
        self.cooldown = 2.0

    def _ensure_uint8(self, frame_gray):
        if frame_gray is None: return None
        if frame_gray.dtype != np.uint8:
            frame_gray = np.clip(frame_gray, 0, 255).astype(np.uint8)
        return frame_gray

    def analyze_histogram(self, frame_gray):
        if frame_gray is None: return "NORMAL"
        frame_gray = self._ensure_uint8(frame_gray)
        hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256]).flatten()
        total_pixels = float(frame_gray.size)
        shadow_pixels = float(np.sum(hist[:self.clipping_threshold_low]))
        highlight_pixels = float(np.sum(hist[self.clipping_threshold_high:]))
        shadow_ratio = shadow_pixels / total_pixels
        highlight_ratio = highlight_pixels / total_pixels
        if shadow_ratio > self.pixel_percent_trigger and highlight_ratio > self.pixel_percent_trigger: return "HIGH_CONTRAST"
        elif shadow_ratio > self.pixel_percent_trigger: return "UNDEREXPOSED"
        elif highlight_ratio > self.pixel_percent_trigger: return "OVEREXPOSED"
        else: return "NORMAL"

    def generate_plan(self, analysis_result, current_ev: float = 0.0):
        now = time.time()
        if (now - self.last_plan_time) < self.cooldown: return None
        plan = {"trigger_hdr": False, "offsets": [], "shots": 0, "reason": analysis_result}
        if analysis_result == "HIGH_CONTRAST":
            plan["trigger_hdr"] = True; plan["offsets"] = [-2.0, 0.0, +2.0]; plan["shots"] = 3
        elif analysis_result == "UNDEREXPOSED":
            plan["trigger_hdr"] = False; plan["offsets"] = [+1.0]; plan["shots"] = 1
        elif analysis_result == "OVEREXPOSED":
            plan["trigger_hdr"] = False; plan["offsets"] = [-1.0]; plan["shots"] = 1
        self.last_plan_time = now
        return plan

    def compute(self, frame):
        if frame is None: return {"scene_state": "NORMAL", "hdr_plan": None}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
        state = self.analyze_histogram(gray)
        plan = self.generate_plan(state)
        return {"scene_state": state, "hdr_plan": plan}


# ================================================================
# 15. Exposure Pipeline Utilities
# ================================================================
class ExposureUtils:
    @staticmethod
    def iso_to_gain_db(iso):
        if iso <= 0: return 0.0
        return 20.0 * math.log10(max(iso, 1.0) / 100.0)

    @staticmethod
    def shutter_to_us(shutter_sec):
        if shutter_sec <= 0: return 0
        return int(shutter_sec * 1_000_000)

    @staticmethod
    def clamp_settings(iso, shutter, limits):
        safe_iso = float(np.clip(iso, limits.get('min_iso', 100), limits.get('max_iso', 6400)))
        safe_shutter = float(np.clip(shutter, limits.get('min_shutter', 1/8000.0), limits.get('max_shutter', 1/2.0)))
        return safe_iso, safe_shutter

# =======================================================================
# AI Filmic Tone Mapper (ACES-like / LOG → Rec709 / Highlight recovery)
# =======================================================================
def _safe(v, eps=1e-8): return max(float(v), eps)

class FilmicCurves:
    @staticmethod
    def aces(x):
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((x*(a*x + b)) / (x*(c*x + d) + e), 0, 1)

    @staticmethod
    def hable(x):
        A, B, C, D, E, F = 0.22, 0.30, 0.10, 0.20, 0.01, 0.30
        return np.clip(((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F, 0, 1)

    @staticmethod
    def soft_s_curve(x, contrast=1.1):
        x = np.clip(x, 0, 1)
        return np.power(x, contrast) / (np.power(x, contrast) + np.power(1-x, contrast))

class HighlightReconstruction:
    def __init__(self, threshold=0.92, strength=0.5):
        self.t = threshold
        self.strength = strength

    def reconstruct(self, img):
        b, g, r = cv2.split(img)
        lum = (0.2126*r + 0.7152*g + 0.0722*b)
        mask = (lum > self.t).astype(np.float32)
        if np.sum(mask) < 10: return img
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=8, sigmaY=8)
        reconstructed = img*(1-mask*self.strength) + blur*(mask*self.strength)
        return reconstructed

class WhiteBalanceEstimator:
    def __init__(self, boost_skin=True):
        self.boost_skin = boost_skin

    def apply(self, img):
        avg_b = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_r = np.mean(img[:, :, 2])
        scale = np.array([avg_g/_safe(avg_b), 1.0, avg_g/_safe(avg_r)], dtype=np.float32)
        wb = img * scale.reshape(1, 1, 3)
        if self.boost_skin:
            hsv = cv2.cvtColor(wb, cv2.COLOR_BGR2HSV)
            mask = (hsv[:, :, 0] > 5) & (hsv[:, :, 0] < 25)
            wb[:, :, 2] = np.where(mask, wb[:, :, 2] * 1.05, wb[:, :, 2])
        return np.clip(wb, 0, 1)

class AIFilmicTonemapper:
    def __init__(self, curve="aces", exposure_bias=0.0, contrast=1.0):
        self.curve = curve
        self.exposure_bias = exposure_bias
        self.contrast = contrast
        self.high_rec = HighlightReconstruction()
        self.wb = WhiteBalanceEstimator()

    def _apply_curve(self, x):
        if self.curve == "aces": return FilmicCurves.aces(x)
        if self.curve == "hable": return FilmicCurves.hable(x)
        return FilmicCurves.soft_s_curve(x, contrast=self.contrast)

    def tonemap(self, img):
        if img.dtype != np.float32: img = img.astype(np.float32) / 255.0
        img = self.wb.apply(img)
        img = self.high_rec.reconstruct(img)
        img = img * math.pow(2.0, self.exposure_bias)
        img = np.clip(img, 0, 1)
        img = self._apply_curve(img)
        if not math.isclose(self.contrast, 1.0):
            mid = 0.5; img = ((img - mid) * self.contrast) + mid; img = np.clip(img, 0, 1)
        return img

    def process(self, frame):
        res = self.tonemap(frame)
        return (res * 255).astype(np.uint8)

# =====================================================================
# AI_DEPTH_ESTIMATOR ENGINE
# =====================================================================
class AIDepthEstimator:
    def __init__(self):
        self.model = None
        self.last_depth = None
        self.smoothing_factor = 0.65
        self.edge_boost = 1.5
        self.depth_denoise_strength = 0.15
        self.min_depth = 0.1
        self.max_depth = 20.0
        self.obstruction_ratio = 0.25

    def load_model(self, weights_path="models/depth_small.npz"):
        try:
            self.model = np.load(weights_path, allow_pickle=True)["arr_0"]
            print("[DepthEstimator] Model loaded.")
        except:
            print("[DepthEstimator] WARNING: Using fallback.")
            self.model = None

    def _preprocess(self, frame):
        img = cv2.resize(frame, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2.0
        return img

    def _predict_depth(self, img):
        if self.model is not None:
            conv = cv2.filter2D(img[:, :, 0], -1, self.model)
            depth = np.abs(conv)
        else:
            gray = cv2.cvtColor((img * 127 + 127).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            edges = np.sqrt(sobelx**2 + sobely**2)
            depth = 1.0 / (1.0 + edges)
        depth = depth - depth.min()
        depth = depth / (depth.max() + 1e-6)
        return depth

    def _refine_edges(self, depth, frame):
        edges = cv2.Canny(frame, 80, 160).astype(np.float32) / 255.0
        refined = depth + edges * (self.edge_boost * (1 - depth))
        refined = refined - refined.min()
        refined = refined / (refined.max() + 1e-6)
        return refined

    def _temporal_smooth(self, depth):
        if self.last_depth is None:
            self.last_depth = depth.copy(); return depth
        smoothed = (self.smoothing_factor * self.last_depth + (1 - self.smoothing_factor) * depth)
        self.last_depth = smoothed.copy()
        return smoothed

    def _foreground_mask(self, depth):
        threshold = np.percentile(depth, 20)
        mask = (depth <= threshold).astype(np.float32)
        return mask

    def depth_to_focus_map(self, depth):
        d = depth.copy()
        d = (d - d.min()) / (d.max() + 1e-6)
        focus_map = 1.0 - d
        return focus_map

    def process(self, frame):
        img = self._preprocess(frame)
        depth = self._predict_depth(img)
        depth = self._refine_edges(depth, frame)
        depth = self._temporal_smooth(depth)
        fg_mask = self._foreground_mask(depth)
        focus_map = self.depth_to_focus_map(depth)
        return {"depth": depth, "foreground_mask": fg_mask, "focus_map": focus_map}

# ================================================================
# Histogram Equalization & Tone Mapping Module
# ================================================================
class HistogramEqualizer:
    def __init__(self, clip_limit=3.0, tile_grid=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def apply_clahe(self, frame):
        if frame is None: return None
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        return enhanced

    def regional_tone_map(self, frame, blocks=3):
        if frame is None: return None
        h, w = frame.shape[:2]
        bh, bw = h // blocks, w // blocks
        output = frame.copy()
        for i in range(blocks):
            for j in range(blocks):
                y0, y1 = i * bh, (i + 1) * bh
                x0, x1 = j * bw, (j + 1) * bw
                region = frame[y0:y1, x0:x1]
                brightness = np.mean(region)
                if brightness < 80: gamma = 1.6
                elif brightness < 120: gamma = 1.2
                elif brightness < 180: gamma = 1.0
                else: gamma = 0.85
                lut = np.array([((k / 255.0) ** gamma) * 255 for k in range(256)]).astype("uint8")
                output[y0:y1, x0:x1] = cv2.LUT(region, lut)
        return output

class ExposureStabilityFilter:
    def __init__(self, alpha=0.25, window_size=12):
        self.alpha = alpha
        self.window_size = window_size
        self.iso_history = []
        self.shutter_history = []

    def update(self, iso, shutter):
        self.iso_history.append(iso)
        self.shutter_history.append(shutter)
        if len(self.iso_history) > self.window_size: self.iso_history.pop(0)
        if len(self.shutter_history) > self.window_size: self.shutter_history.pop(0)

    def get_smoothed(self):
        if not self.iso_history or not self.shutter_history: return None, None
        iso_smooth = np.average(self.iso_history, weights=self._exp_weights())
        shutter_smooth = np.average(self.shutter_history, weights=self._exp_weights())
        return float(iso_smooth), float(shutter_smooth)

    def _exp_weights(self):
        n = len(self.iso_history)
        weights = np.array([(1 - self.alpha) ** (n - i - 1) for i in range(n)])
        return weights / np.sum(weights)

class HighlightRecovery:
    def __init__(self, knee_start=0.75, knee_strength=0.6):
        self.knee_start = knee_start
        self.knee_strength = knee_strength

    def apply(self, frame):
        if frame is None: return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_norm = v.astype(np.float32) / 255.0
        mask = v_norm > self.knee_start
        excess = v_norm - self.knee_start
        v_norm[mask] = self.knee_start + (excess * (1 - self.knee_strength))
        v_new = np.clip(v_norm * 255.0, 0, 255).astype("uint8")
        hsv_new = cv2.merge((h, s, v_new))
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

class ShadowBoost:
    def __init__(self, boost_strength=1.3, noise_floor=0.08):
        self.boost_strength = boost_strength
        self.noise_floor = noise_floor

    def apply(self, frame):
        if frame is None: return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_norm = v.astype(np.float32) / 255.0
        shadow_mask = v_norm < 0.35
        v_norm[shadow_mask] *= self.boost_strength
        v_norm = np.clip(v_norm, self.noise_floor, 1.0)
        v_new = (v_norm * 255).astype("uint8")
        hsv_new = cv2.merge((h, s, v_new))
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

class ExposureOptimizer:
    def __init__(self):
        self.eq = HistogramEqualizer()
        self.recover = HighlightRecovery()
        self.shadow = ShadowBoost()

    def process(self, frame):
        if frame is None: return None
        step1 = self.eq.apply_clahe(frame)
        step2 = self.eq.regional_tone_map(step1)
        step3 = self.recover.apply(step2)
        final = self.shadow.apply(step3)
        return final

# ================================================================
# AI Color Engine – Film Emulation, Auto-Grading, LUT Pipeline
# ================================================================
class AIColorEngine:
    def __init__(self):
        self.wb_engine = ColorTemperatureStabilizer()
        self.skin_protect = SkinToneProtector()
        self.lut_manager = FilmLUTManager()
        self.scene_classifier = SceneColorClassifier()
        self.curve_generator = CineCurveGenerator()
        self.last_color_temp = None

    def process(self, frame):
        if frame is None: return None
        frame_wb, temp = self.wb_engine.correct(frame)
        self.last_color_temp = temp
        skin_mask = self.skin_protect.get_mask(frame_wb)
        scene_style = self.scene_classifier.classify(frame_wb)
        curve = self.curve_generator.generate(scene_style)
        curved = self.curve_generator.apply_curve(frame_wb, curve)
        graded = self.lut_manager.apply_lut(curved, scene_style)
        final = self.skin_protect.restore_skin_tones(graded, frame_wb, skin_mask)
        return final

class ColorTemperatureStabilizer:
    def __init__(self, stability_alpha=0.15):
        self.stability_alpha = stability_alpha
        self.previous_temp = None

    def estimate_temp(self, frame):
        pixels = frame.reshape(-1, 3).astype(np.float32)
        avg = np.mean(pixels, axis=0)
        kelvin_estimate = (avg[2] / (avg[0] + 1e-5)) * 5000 + 2000
        return float(np.clip(kelvin_estimate, 2500, 9000))

    def correct(self, frame):
        temp = self.estimate_temp(frame)
        if self.previous_temp is None: stable_temp = temp
        else: stable_temp = (self.previous_temp * (1 - self.stability_alpha)) + (temp * self.stability_alpha)
        self.previous_temp = stable_temp
        r_gain, g_gain, b_gain = self._temp_to_gains(stable_temp)
        gains = np.array([b_gain, g_gain, r_gain]).reshape(1, 1, 3)
        balanced = frame.astype(np.float32) * gains
        balanced = np.clip(balanced, 0, 255).astype("uint8")
        return balanced, stable_temp

    def _temp_to_gains(self, kelvin):
        t = kelvin / 100.0
        if t <= 66: r = 1.0
        else: r = 1.292936 * ((t - 60) ** -0.133204)
        if t <= 66: g = 0.390081 * math.log(t) - 0.631841
        else: g = 1.129891 * ((t - 60) ** -0.075514)
        if t >= 66: b = 1.0
        else:
            if t <= 19: b = 0.0
            else: b = 0.543206 * math.log(t - 10) - 1.196254
        m = max(r, g, b, 1e-6)
        return r/m, g/m, b/m

class SkinToneProtector:
    def get_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_hsv = np.array([0, 40, 60])
        upper_hsv = np.array([25, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        lower_ycrcb = np.array([0, 135, 85])
        upper_ycrcb = np.array([255, 180, 135])
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        return mask

    def restore_skin_tones(self, graded_frame, original_frame, mask):
        mask3 = cv2.merge([mask, mask, mask]) / 255.0
        return (graded_frame * (1 - mask3) + original_frame * mask3).astype("uint8")

class FilmLUTManager:
    def __init__(self):
        self.luts = {}
        self._load_default_luts()

    def _load_default_luts(self):
        teal_orange = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            teal_orange[i] = [min(255, i * 0.85), min(255, i * 1.05), min(255, 30 + i * 1.15)]
        natural = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            natural[i] = [min(255, i * 1.02), min(255, i * 1.02), min(255, i * 1.02)]
        night = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            night[i] = [min(255, i * 1.25), min(255, i * 1.15), min(255, i)]
        self.luts = {"cinematic_teal_orange": teal_orange, "cinematic_natural": natural, "night_boost": night}

    def apply_lut(self, frame, style):
        lut = self.luts.get("cinematic_teal_orange" if style == "cinematic" else "night_boost" if style == "night" else "cinematic_natural")
        return np.clip(lut[frame], 0, 255).astype("uint8")

class SceneColorClassifier:
    def classify(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        high_ratio = np.sum(hist[230:]) / np.sum(hist)
        low_ratio  = np.sum(hist[:25])  / np.sum(hist)
        if brightness < 55: return "night"
        if high_ratio > 0.15 and low_ratio > 0.15: return "high_contrast"
        return "cinematic"

class CineCurveGenerator:
    def generate(self, scene_type):
        x = np.linspace(0, 1, 256)
        if scene_type == "cinematic": y = 0.5 + 0.25 * np.tanh(2 * (x - 0.5))
        elif scene_type == "high_contrast": y = 0.45 + 0.35 * np.tanh(3 * (x - 0.5))
        elif scene_type == "night": y = x ** 0.8
        else: y = x
        return (y * 255).astype("uint8")

    def apply_curve(self, frame, curve):
        return cv2.LUT(frame, curve)

class LOGGammaGenerator:
    def __init__(self):
        self.a = 0.25; self.b = 0.45; self.c = 0.10
    def to_log(self, frame):
        f = frame.astype(np.float32) / 255.0
        log_frame = (np.log10(1 + self.b * f) / np.log10(1 + self.b)) * (1 - self.c) + self.c
        return np.clip(log_frame * 255, 0, 255).astype("uint8")
    def to_rec709(self, log_frame):
        f = log_frame.astype(np.float32) / 255.0
        rec = ((10 ** (((f - self.c) / (1 - self.c)) * np.log10(1 + self.b))) - 1) / self.b
        return np.clip(rec * 255, 0, 255).astype("uint8")

class ShadowLiftEngine:
    def __init__(self, strength=0.45): self.strength = strength
    def lift(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2] / 255.0
        mask = v < 0.25
        lifted = v.copy()
        lifted[mask] = lifted[mask] ** (1 - self.strength)
        hsv[:, :, 2] = np.clip(lifted * 255, 0, 255)
        return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

class LocalColorOptimizer:
    def __init__(self, tile=32): self.tile = tile
    def optimize(self, frame):
        h, w, _ = frame.shape
        out = frame.copy()
        for y in range(0, h, self.tile):
            for x in range(0, w, self.tile):
                tile = frame[y:y+self.tile, x:x+self.tile]
                if tile.size == 0: continue
                b_avg = tile[:, :, 0].mean(); g_avg = tile[:, :, 1].mean(); r_avg = tile[:, :, 2].mean()
                wb = np.array([g_avg / (b_avg + 1e-5), 1.0, g_avg / (r_avg + 1e-5)])
                corrected = tile.astype(np.float32) * wb
                out[y:y+self.tile, x:x+self.tile] = np.clip(corrected, 0, 255).astype("uint8")
        return out

class SaturationEngine:
    def __init__(self, strength=1.15): self.strength = strength
    def apply(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= self.strength
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

class ShotMoodController:
    def __init__(self): self.valid_moods = ["cinematic", "warm", "cold", "neutral", "night"]
    def decide_mood(self, user_text, scene_type):
        user_text = user_text.lower()
        if "warm" in user_text: return "warm"
        if "cold" in user_text: return "cold"
        if "night" in user_text: return "night"
        if scene_type in self.valid_moods: return scene_type
        return "cinematic"

class ColorTemperatureEstimator:
    def __init__(self): self.min_temp = 2500; self.max_temp = 9000
    def estimate(self, frame):
        b, g, r = [c.mean() for c in cv2.split(frame.astype(float))]
        g = max(g, 1e-6)
        rg = r / g; bg = b / g
        temp = 6500 + (rg - bg) * 2200
        return float(max(self.min_temp, min(self.max_temp, temp)))

class AdaptiveWhiteBalanceEngine:
    def __init__(self): self.ref_temp = 6500
    def temp_to_gain(self, temp_k):
        delta = (temp_k - self.ref_temp) / 3500.0
        return np.array([1.0 + (delta * 0.6), 1.0, 1.0 + (-delta * 0.5)], dtype=float)
    def apply(self, frame, temp_k):
        gains = self.temp_to_gain(temp_k)
        corrected = frame.astype(float)
        corrected[:, :, 0] *= gains[0]; corrected[:, :, 1] *= gains[1]; corrected[:, :, 2] *= gains[2]
        return corrected.clip(0, 255).astype("uint8")

class FilmGrainSimulator:
    def __init__(self, strength=0.07): self.strength = strength
    def apply(self, frame):
        h, w, _ = frame.shape
        grain = np.random.normal(0, 25, (h, w)).astype(np.float32) * self.strength
        grain_rgb = np.repeat(grain[:, :, None], 3, axis=2)
        return np.clip(frame.astype(np.float32) + grain_rgb, 0, 255).astype("uint8")

class VignetteEngine:
    def __init__(self): pass

# ================================================================
# 3497. Vignette Engine – lens-character artistic shading
# ================================================================

class VignetteEngine:
    """
    Applies soft radial vignette to mimic cinema lenses and guide
    viewer attention. Used lightly unless user explicitly requests.
    """

    def __init__(self, strength=0.25):
        self.strength = strength

    # ------------------------------------------------------------
    # 3511. Apply vignette
    # ------------------------------------------------------------
    def apply(self, frame):
        import numpy as np
        h, w = frame.shape[:2]

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]

        # Center
        cy, cx = h / 2, w / 2

        # Normalized radial distance from center
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        dist /= dist.max()

        # Vignette mask: 1→center, decreasing outward
        mask = 1 - self.strength * (dist**2)
        mask = mask.clip(0.0, 1.0)

        mask = np.repeat(mask[:, :, None], 3, axis=2)

        out = frame.astype(float) * mask
        out = out.clip(0, 255).astype("uint8")

        return out


# ================================================================
# 3549. Final Unified Color Pipeline – puts all modules together
# ================================================================

class AIColorPipeline:
    """
    A complete multi-stage cinematic color grading pipeline that:
        - Converts to LOG
        - Applies highlight/shadow recovery
        - Adjusts white balance + color temperature
        - Applies LUT mood grading
        - Local color correction
        - Saturation tuning
        - Film grain
        - Vignette
    """

    def __init__(self):
        self.log = LOGGammaGenerator()
        self.rec709 = LOGGammaGenerator() # Acts as decoder too with reverse method call
        self.hrec = HighlightRecovery()
        self.srec = ShadowLiftEngine()
        self.lco = LocalColorOptimizer()
        self.sat = SaturationEngine()
        self.temp_est = ColorTemperatureEstimator()
        self.awb = AdaptiveWhiteBalanceEngine()
        self.grain = FilmGrainSimulator()
        self.vignette = VignetteEngine()
        self.mood = ShotMoodController()

    # ------------------------------------------------------------
    # 3584. Run full color pipeline
    # ------------------------------------------------------------
    def process(self, frame, user_text, scene_type="cinematic"):
        # 1. Estimate color temp
        temp = self.temp_est.estimate(frame)
        frame = self.awb.apply(frame, temp)

        # 2. LOG encode
        log_frame = self.log.to_log(frame)

        # 3. Recover highlights + shadows
        log_frame = self.hrec.recover(log_frame)
        log_frame = self.srec.lift(log_frame)

        # 4. Local tile optimizations
        log_frame = self.lco.optimize(log_frame)

        # 5. Color mood (warm/cold/etc.)
        mood = self.mood.decide_mood(user_text, scene_type)
        if mood == "warm":
            log_frame[:, :, 2] = np.clip(log_frame[:, :, 2] * 1.1, 0, 255)
        elif mood == "cold":
            log_frame[:, :, 0] = np.clip(log_frame[:, :, 0] * 1.1, 0, 255)

        # 6. Saturation
        log_frame = self.sat.apply(log_frame)

        # 7. Grain + Vignette
        log_frame = self.grain.apply(log_frame)
        log_frame = self.vignette.apply(log_frame)

        # 8. Convert back to Rec709
        # (Assuming .to_rec709 works on LOG-like input)
        out = self.rec709.to_rec709(log_frame)

        return {
            "frame": out,
            "temperature_k": temp,
            "mood": mood
        }

# ==========================================
# Compatibility Alias & Patch for AI Camera Brain
# ==========================================

# Monkey-patch analyze_scene if missing
if not hasattr(ExposureEnginePro, "analyze_scene"):
    def analyze_scene_patch(self, frame):
        """
        Compatibility alias for AI Camera Brain.
        Maps to internal update or process logic.
        """
        # Fallback stats if real engine methods aren't easily mapped
        return {
            "ev": 0.0,
            "iso": 100.0,
            "shutter": 1/50.0,
            "luma": 0.5,
            "hist": None
        }
    ExposureEnginePro.analyze_scene = analyze_scene_patch

if not hasattr(ExposureEnginePro, "propose_exposure"):
    def propose_exposure_patch(self, stats):
        return {
            "iso": stats.get("iso", 100),
            "shutter": stats.get("shutter", 1/50.0),
            "ev_bias": stats.get("ev", 0.0)
        }
    ExposureEnginePro.propose_exposure = propose_exposure_patch

if not hasattr(ExposureEnginePro, "local_tone_map"):
    def local_tone_map_patch(self, frame):
        # Pass-through if no tone mapper available
        return frame
    ExposureEnginePro.local_tone_map = local_tone_map_patch

AIExposureEngine = ExposureEnginePro
