
import cv2
import numpy as np
import math
import time

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

# =====================================================================
# CLASS 18 — EXPOSURE TOOLS WRAPPER
# =====================================================================
class ExposureTools:
    def __init__(self):
        self.visualizer = ExposureDebugVisualizer()
        self.hdr_planner = HDRBracketingPlanner()
        self.nd_advisor = NDFilterAdvisor()
        self.filmic = FilmicToneCurve()
        self.metadata = ExposureMetadataPacker()
